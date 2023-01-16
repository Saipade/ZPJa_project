from transformers import BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
from typing import List, Tuple
import pyonmttok
import os
import torch
import static
from torch import Tensor
from torch.nn import functional as f, CosineSimilarity as Cos
import torch.nn
from laserembeddings import Laser

class SimilarityChecker:

    def __init__(self, src_lang: str, tgt_lang: str) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.pyo_tokenizer = pyonmttok.Tokenizer(mode='aggressive')
        # initialize labse tokenizer and labse model
        self.labse_tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.labse_model = BertModel.from_pretrained("setu4993/LaBSE")
        self.labse_model = self.labse_model.eval()
        # initialize laser model
        encoder_file = open(os.path.join(static.laser_path, 'bilstm.93langs.2018-12-26.pt'), 'rb')
        self.laser = Laser(os.path.join(static.laser_path, '93langs.fcodes'), os.path.join(static.laser_path, '93langs.fvocab'), encoder_file)
        encoder_file.close()

    @staticmethod
    def get_similarity(src_embeddings: Tensor, tgt_embeddings: Tensor) -> List[float]:
        normalized_src_embeddings = f.normalize(src_embeddings, p=2)
        normalized_tgt_embeddings = f.normalize(tgt_embeddings, p=2)
        cos = Cos()
        return [float(sim) for sim in cos(normalized_src_embeddings, normalized_tgt_embeddings)]

    def get_number_of_tokens_ratios(self, src: List, tgt: List) -> List[int]:
        return list(map(
            lambda length_src, length_tgt: length_src/length_tgt,
            [len(self.pyo_tokenizer.tokenize(s)[0]) for s in src], [len(self.pyo_tokenizer.tokenize(t)[0]) for t in tgt]
        ))

    def labse_similarity(self, src: List, tgt: List) -> List[bool]:
        """Get sentence pairs similarity from labse
        """
        src_inputs = self.labse_tokenizer(src, return_tensors='pt', padding=True)
        tgt_inputs = self.labse_tokenizer(tgt, return_tensors='pt', padding=True)
        with torch.no_grad():
            src_outputs = self.labse_model(**src_inputs)
            tgt_outputs = self.labse_model(**tgt_inputs)
        return list(
            self.get_similarity(src_outputs.pooler_output, tgt_outputs.pooler_output)
        )

    def laser_similarity(self, src: List, tgt: List) -> List[bool]:
        """Get sentence pairs similarity from laser
        """
        return list(
            self.get_similarity(Tensor(self.laser.embed_sentences(src, self.src_lang)), Tensor(self.laser.embed_sentences(tgt, self.tgt_lang)))
        )

    def __call__(self, src: List, tgt: List) -> Tuple[List]:
        return self.laser_similarity(src, tgt), self.labse_similarity(src, tgt), self.get_number_of_tokens_ratios(src, tgt)

# example
if __name__ == "__main__":
    sc = SimilarityChecker('eu', 'en')
    print(sc(
        ['Hiru hartzak »', 'Mila eta bostehun karaktere euskarari forma emateko.', '2012-08-05 eguneratua',
        'Krisi politikoak, ordea, jarraituko du.', 'Lehenengo klasea bezeroarentzako zerbitzua – gomendatzen produktu handi bat;',
         'Niri jendea margotzea gustatzen zait.', 'Federazioarekin borrokatu behar izan genuen lehiaketa hartan parte hartzeko baimena eman ziezadaten.',
         '7 * 24 online zerbitzuak.'],
        ['“the three bears”', 'One hundred cents form a guilder.', 'Amendment proposed by 52008PC0815 Repeal', 'So, the political fight will continue.', 'Delivering a first class service or an amazing product.',
         'I like to paint people.', 'We even had to get a federal court order in order to be allowed to hold the rally.',
         'We have 24/7 online service.']
    ))

