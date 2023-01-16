# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is modified version of the original `run_align.py` script
# modified by Maksim Tikhonov

from typing import List, Tuple
import io
import random
import itertools
import os
import sys
import shutil
import tempfile
import numpy as np
import torch
from scipy.sparse import csr_matrix
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
import pyonmttok

from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel


def set_seed(seed):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class LineByLineTextDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file, offsets=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.file = file
        self.offsets = offsets

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None

        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()

        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], \
                               [self.tokenizer.tokenize(word) for word in sent_tgt]

        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], \
                           [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                            max_length=self.tokenizer.max_len)['input_ids'], \
                           self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                            max_length=self.tokenizer.max_len)['input_ids']

        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)


    def __iter__(self):
        if self.offsets is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id
            offset_start = self.offsets[worker_id]
            offset_end = self.offsets[worker_id + 1] if worker_id + 1 < len(self.offsets) else None
        else:
            offset_start = 0
            offset_end = None
            worker_id = 0

        self.file.seek(offset_start)
        line = self.file.readline()
        while line:
            processed = self.process_line(worker_id, line)
            if processed is None:
                print(f'Line "{line.strip()}" (offset in bytes: {self.file.tell()}) is not in the correct format. Skipping...')
                empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                empty_sent = ''
                yield (worker_id, empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
            else:
                yield processed
            if offset_end is not None and self.file.tell() >= offset_end:
                    break
            line = self.file.readline()


class AwesomeAligner:

    def __init__(
        self,
        output_file=None,
        output_prob_file=None,
        output_word_file=None,
        extraction='softmax',
        align_layer=8,
        softmax_threshold=0.000,
        model_name_or_path="bert-base-multilingual-cased",
        config_name=None,
        tokenizer_name=None,
        seed=42,
        batch_size=32,
        cache_dir=None,
        no_cuda=True,
        num_workers=1
    ):

        self.data_file = None
        self.output_file = None
        self.output_prob_file = None
        self.output_word_file = None
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.extraction = extraction
        self.align_layer = align_layer
        self.softmax_threshold = softmax_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.tokenizer = None

        set_seed(seed)
        config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
        if config_name:
            config = config_class.from_pretrained(config_name, cache_dir=cache_dir)
        elif model_name_or_path:
            config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        else:
            config = config_class()

        if tokenizer_name:
            self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        elif model_name_or_path:
            self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
            )

        modeling.PAD_ID, modeling.CLS_ID, modeling.SEP_ID = self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        if model_name_or_path:
            self.model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model_class(config=config)

        self.model.to(self.device)
        self.model.eval()

    def open_writer_list(self, writers_attr):
        writer = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        writer.seek(0)
        writers = [writer]
        if self.num_workers > 1:
            writers.extend([tempfile.TemporaryFile(mode='w+', encoding='utf-8') for i in range(1, self.num_workers)])
        setattr(self, writers_attr, writers)

    def find_offsets(self):
        if self.num_workers <= 1:
            return None
        chunk_size = os.fstat(self.data_file.fileno()).st_size // self.num_workers
        offsets = [0]
        for i in range(1, self.num_workers):
            self.data_file.seek(chunk_size * i)
            pos = self.data_file.tell()
            while True:
                try:
                    l = self.data_file.readline()
                    break
                except UnicodeDecodeError:
                    pos -= 1
                    self.data_file.seek(pos)
            offsets.append(self.data_file.tell())
        return offsets

    def merge_files(self, writers_name):
        writers = getattr(self, writers_name)
        if len(writers) == 1:
            return
        for i, writer in enumerate(writers[1:], 1):
            writer.seek(0)
            shutil.copyfileobj(writer, writers[0])
            writer.close()

    def word_align(self, src, tgt):

        def tokenize(data):
            return ' '.join(tokenizer.tokenize(data.strip())[0])

        def to_parallel(sentence):
            return f'{tokenize(sentence[0])} ||| {tokenize(sentence[1])}'

        def format_data(src, tgt):
            data = [sents for sents in zip(src, tgt)]
            return '\n'.join([*map(to_parallel, data)])

        def collate(examples):
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
            ids_src = pad_sequence(ids_src, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

        tokenizer = pyonmttok.Tokenizer(mode="aggressive")
        self.data_file = tempfile.TemporaryFile(mode='w+t', encoding='utf-8')

        if type(self.data_file) is io.TextIOWrapper:
            self.data_file.write(format_data(src, tgt))
            self.data_file.seek(0)

        offsets = self.find_offsets()
        dataset = LineByLineTextDataset(self.tokenizer, file=self.data_file, offsets=offsets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate, num_workers=self.num_workers)
        tqdm_iterator = trange(0, desc="Extracting")

        self.open_writer_list("writers")
        if self.output_prob_file is not None:
            self.open_writer_list("prob_writers")
        if self.output_word_file is not None:
            self.open_writer_list("word_writers")

        out_matrices, out_indices, out_probs = [], [], []
        for batch in dataloader:
            with torch.no_grad():
                worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
                word_aligns_list = self.model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, self.device,
                                                          0, 0, align_layer=self.align_layer, extraction=self.extraction,
                                                          softmax_threshold=self.softmax_threshold, test=True,
                                                          output_prob=True)
                # revamped alignment indices assignment
                for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                    key_aligns = np.array(list(word_aligns.keys())).T
                    values_aligns = np.array(list(word_aligns.values()))
                    # matrix format:
                    # each row represents probabilities of target_sent[row_idx] being aligned to source_sent[column_idx]
                    try:
                        matrix = csr_matrix((values_aligns, (key_aligns[1], key_aligns[0])),
                                            shape=(len(sent_tgt), len(sent_src)), dtype=float).toarray()
                        out_indices.append(np.argmax(matrix, axis=1))
                        out_probs.append(np.max(matrix, axis=1))
                        out_matrices.append(matrix)
                    except ValueError:  # for empty lines, mostly
                        out_indices.append(np.array([]))
                        out_probs.append(np.array([]))
                        out_matrices.append(csr_matrix((0, 0)).toarray())
                    # tgt_indices = np.argmax(matrix, axis=1)
                    # [print(f'[{idx_src}] {sent_src[idx_src]} -> [{idx_tgt}] {sent_tgt[idx_tgt]}', file=sys.stderr) for idx_tgt, idx_src in enumerate(tgt_indices)]

                tqdm_iterator.update(len(ids_src))

        self.merge_files("writers")
        self.writers[0].seek(0)
        self.writers[0].close()
        # return tgt indices, src indices, align matrices, most probable weights
        return [list(range(len(indices))) for indices in out_indices], \
               [out_idx.tolist() for out_idx in out_indices], \
               [matrix.tolist() for matrix in out_matrices], \
               [prob.tolist() for prob in out_probs]

    def add_quotes(src: List, tgt: List) -> Tuple[List]:
        pass

src_sents_euen = [
    'Krisi politikoak, ordea, jarraituko du.', 'Lehenengo klasea bezeroarentzako zerbitzua – gomendatzen produktu handi bat;',
    'Niri jendea margotzea gustatzen zait.', 'Federazioarekin borrokatu behar izan genuen lehiaketa hartan parte hartzeko baimena eman ziezadaten.',
    '7 * 24 online zerbitzuak.', '"It feels so scary.'
]

tgt_sents_euen = [
    'So, the political fight will continue.', 'Delivering a first class service or an amazing product.',
    'I like to paint people.', 'We even had to get a federal court order in order to be allowed to hold the rally.',
    'We have 24/7 online service.', '- "Oso beldurgarria dirudi.'
]

def test():
    """
    Example of usage
    """

    def tokenize(data):
        return ' '.join(tokenizer.tokenize(data.strip())[0])

    def to_parallel(sentence):
        return f'{tokenize(sentence[0])} ||| {tokenize(sentence[1])}'

    def format_data(src, tgt):
        data = [sents for sents in zip(src, tgt)]
        return '\n'.join([*map(to_parallel, data)])

    def split_data(data):
        splitted_data = [f_sent.split(' ||| ') for f_sent in data.split('\n')]
        return [[f_sent[0].split(), f_sent[1].split()] for f_sent in splitted_data]

    def fancy_print(bench : str, lan_1 : str, lan_2 : str):
        for idx, splitted in enumerate(splitted_data):
            print(splitted)
            print(f'src:\n{src_alignments[idx]}\n')
            print(f'tgt:\n{tgt_alignments[idx]}\n')
            [
                print(f'[{src_alignment}] {splitted[0][src_alignment]} -> [{tgt_alignment}] {splitted[1][tgt_alignment]} {prob_matrix[idx][jdx]}')
                for jdx, (src_alignment, tgt_alignment) in enumerate(zip(src_alignments[idx], tgt_alignments[idx]))
            ]
            print(f'{align_matrix[idx]}\n')

    tokenizer = pyonmttok.Tokenizer(mode="aggressive")
    aligner = AwesomeAligner(model_name_or_path='models/awesome-align')
    formatted = format_data(src_sents_euen, tgt_sents_euen)
    splitted_data = split_data(formatted)
    tgt_alignments, src_alignments, align_matrix, prob_matrix = aligner.word_align(src_sents_euen, tgt_sents_euen)
    print(f'align matrix:\n{align_matrix}\n')
    fancy_print('bench', 'en', 'cz')

