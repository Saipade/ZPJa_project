import argparse
import sys
from typing import List, Tuple
import os
import string
import static
import numpy as np
from similarity import SimilarityChecker
import subprocess
import pandas as pd
import fasttext as ft
from normalize import normalize
from unbake import unbake
from markup import replace_markup
from spelling import fix_spelling, fix_numerical_expressions, fix_punctuation
from profanity import remove_profanity
import itertools
from difflib import SequenceMatcher

TOKENIZED_SENTENCE_LENGTH_RATIO_THRESHOLD = 2
LABSE_SIMILARITY_THRESHOLD = 0.63
# set of parallel duplicates
duplicates = set()

class DataCleaner:

    def __init__(self, src_lang: str, tgt_lang: str, delim: str, deduplicate: bool, verify_langs: bool, **kwargs):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.delim = delim
        self._deduplicate = deduplicate
        self.verify_langs = verify_langs

        self.download_models()
        self.vl_model = ft.load_model(os.path.join(static.fasttext_path, 'lid.176.bin'))

    @staticmethod
    def download_models():
        """Static function for models downloading
        """
        subprocess.run(['python', '-m', 'laserembeddings', 'download-models', static.laser_path]) \
            if not os.listdir(static.laser_path) else None
        subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', '-P', static.fasttext_path]) \
            if not os.listdir(static.fasttext_path) else None


    def clean_data(self, src_file, tgt_file, **kwargs) -> None:
        """Main function that applies all cleaning methods
        """
        src_file = open(src_file, 'r') if src_file is not sys.stdin else sys.stdin
        self.tgt_file = open(f'{tgt_file}{".fixed" if args.filter_by_sims else ""}', 'w+') if tgt_file is not sys.stdout \
            else sys.stdout
        self.tgt_filename = tgt_file
        # deduplicate
        if self._deduplicate and self.tgt_file is not sys.stdout:
            print('Deduplicating the dataset...', file=sys.stderr)
            dedup_filename = f'{self.tgt_filename}.dedup'
            dedup_tmp = open(dedup_filename, 'w+')
            while (lines := src_file.readlines(100000)):
                src, tgt = self.format_lines(lines)
                src, tgt = self.deduplicate(src, tgt)
                for s, t in zip(src, tgt):
                    dedup_tmp.write(f'{s}\t{t}\n')
            dedup_tmp.close()
            src_file.close()
            src_file = open(dedup_filename, 'r')   # use deduplicated variant as a new source file

        # clean
        print('Cleaning the dataset...', file=sys.stderr)
        for line in src_file:
            fixed_src, fixed_tgt = [sent.strip() for sent in line.split(self.delim)]
            # verify languages
            if self.verify_langs and not self.verify_languages(fixed_src, fixed_tgt):
                continue
            # apply cleaning functions
            for cleaning_fun in [normalize, unbake, replace_markup, fix_spelling, fix_numerical_expressions, fix_punctuation, remove_profanity]:
                fixed_src, fixed_tgt = cleaning_fun(
                    fixed_src, fixed_tgt,
                    src_lang=self.src_lang, tgt_lang=self.tgt_lang
                )
            self.tgt_file.write(f'{fixed_src}{self.delim}{fixed_tgt}\n') if all([fixed_src, fixed_tgt]) \
                and all([True if len(sent)>=10 else False for sent in [fixed_src, fixed_tgt]]) \
                else None

        if src_file is not sys.stdin:
            src_file.close()
            os.remove(dedup_filename) if os.path.exists(dedup_filename) else None
        if self.tgt_file is not sys.stdout:
            self.tgt_file.close()

    def deduplicate(self, src: List, tgt: List) -> Tuple[List]:
        """Removes duplicates in current source/target sentences
        """
        def find_duplicates(sent_list: List, tag) -> None:
            for sent_1, sent_2 in itertools.combinations(sent_list, 2):
                idx = non_punct_src.index(sent_1) if tag == 'src' and sent_1 in non_punct_src \
                    else non_punct_tgt.index(sent_1) if tag == 'tgt' and sent_1 in non_punct_tgt \
                    else None
                if not idx or not sent_1 or not sent_2:
                    continue
                if sent_1 in duplicates:
                    non_punct_src[idx], non_punct_tgt[idx] = False, False
                    continue
                if SequenceMatcher(None, sent_1, sent_2).ratio() > 0.9:
                    non_punct_src[idx], non_punct_tgt[idx] = False, False
                    duplicates.add(sent_1)

        non_punct_src, non_punct_tgt = [], []
        for idx, (s, t) in enumerate(zip(src, tgt)):
            non_punct_src.append(s.lower())
            non_punct_tgt.append(t.lower())
            for c in [string.punctuation, ' ', '\t', '\n']:
                non_punct_src[idx], non_punct_tgt[idx] = non_punct_src[idx].replace(c, ''), non_punct_tgt[idx].replace(c, '')
        # deduplicate
        find_duplicates(non_punct_src, 'src')
        find_duplicates(non_punct_tgt, 'tgt')
        return list(itertools.compress(src, non_punct_src)), list(itertools.compress(tgt, non_punct_tgt))

    def verify_languages(self, src: str, tgt: str) -> bool:
        """Removes lines that do not match the language on either
        of sides
        """
        def format_language_label(src: tuple) -> str:
            return src[0][0][-2:]

        return format_language_label(self.vl_model.predict(src)) == self.src_lang \
            and format_language_label(self.vl_model.predict(tgt)) == self.tgt_lang

    def get_similarities(self, sim_filename: str, **kwargs) -> None:
        """Creates files with parallel sentences labeled with similarity scores
        """
        self.src_filename = f'{self.tgt_filename}{".fixed" if args.filter_by_sims else ""}'
        self.sim_filename = sim_filename
        similarity_checker = SimilarityChecker(self.src_lang, self.tgt_lang)
        src_file = open(self.src_filename, 'r')
        sim_file = open(sim_filename, 'w+')
        sim_file.write(f's\tt\tlaser\tlabse\ttok_n\n')
        while (lines := src_file.readlines(10000)):
            src, tgt = self.format_lines(lines)
            stats = similarity_checker(src, tgt)
            for s, t, laser, labse, tok_n in zip(src, tgt, *stats):
                sim_file.write(f'{s}\t{t}\t{laser}\t{labse}\t{tok_n}\n')
        src_file.close()
        sim_file.close()

    def filter_by_similarities(self, min_size) -> None:
        sims = pd.read_csv(self.sim_filename, sep=self.delim, on_bad_lines='skip', engine='python')
        convert_dict = {'labse': float, 'laser': float, 'tok_n': float}
        sims = sims.astype(convert_dict)
        sims['mul'] = np.where(
            (sims['tok_n'] < TOKENIZED_SENTENCE_LENGTH_RATIO_THRESHOLD) | (sims['tok_n'] > 1/TOKENIZED_SENTENCE_LENGTH_RATIO_THRESHOLD),
            1, 0.8
        )
        sims['heu'] = (sims['labse']*2 + sims['laser']) * sims['mul']
        sims = sims.sort_values('heu', ascending=False)
        sims_head = sims.head(min_size)
        sims_tail = sims.iloc[min_size:]
        sims_tail = sims_tail.loc[sims['labse'] > LABSE_SIMILARITY_THRESHOLD]
        sims = pd.concat([sims_head, sims_tail])
        sims.to_csv(self.tgt_filename, columns=['s', 't'], sep='\t', index=False, header=False)

    def format_lines(self, lines: List) -> List[List]:
        lines = [line.strip().split(self.delim) for line in lines]
        return list(list(pair) for pair in zip(*lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', default='eu', help='Source language')
    parser.add_argument('--tgt_lang', default='en', help='Target language')
    parser.add_argument('-s', '--src', dest='src_file', default=sys.stdin)
    parser.add_argument('-t', '--tgt', dest='tgt_file', default=sys.stdout)
    parser.add_argument('-d', '--delim', default='\t')
    parser.add_argument('--no_dedup', dest='deduplicate', action='store_false')
    parser.add_argument('--verify_langs', default=False, action='store_true')
    parser.add_argument('--get_sims', default=False, action='store_true')
    parser.add_argument('--min_size', default=100000, type=int, help='Minimum size of fixed data')
    parser.add_argument('--filter_by_sims', default=False, action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cleaner = DataCleaner(**args.__dict__)
    # get original dataset's similarities
    cleaner.get_similarities(f'{args.src_file}.sims') if args.get_sims else None
    # get <data>.fixed
    cleaner.clean_data(**args.__dict__)
    # fitler by similarity
    cleaner.get_similarities(f'{args.tgt_file}.sims') if args.get_sims or args.filter_by_sims else None
    cleaner.filter_by_similarities(args.min_size) if args.filter_by_sims else None
    if not args.get_sims:
        os.remove(f'{args.src_file}.sims') if os.path.exists(f'{args.src_file}.sims') else None
        os.remove(f'{args.tgt_file}.sims') if os.path.exists(f'{args.tgt_file}.sims') else None
