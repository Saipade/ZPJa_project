from textblob import TextBlob
from textblob._text import Spelling as BaseSpelling, _read
import sys
import string
from typing import List
from collections import Counter
import regex
import itertools
from difflib import SequenceMatcher
import pyonmttok
import static
import os
import fasttext as ft

class Spelling(BaseSpelling):
    # basque alphabet
    ALPHA = 'abcçdefghijklmnñopqrstuvwxyz'

    def __init__(self, path, lang=None, lm=None):
        self._path = path
        self.lm = lm
        self.lang = lang

    def load(self):
        """Loads the vocabulary
        """
        #if not self._path.endswith('.words'):
        #    words = Counter(regex.findall(r'[{}]+'.format(self.ALPHA), open(self._path).read().lower()))
        #    self._path = f'{self._path.rsplit(".", 1)[0]}.words'
        #    with open(self._path, 'w') as words_file:
        #        [words_file.write(f'{word} {words[word]}\n') for word in words]
        self._path = open(self._path, 'r')
        for x in _read(self._path):
            x = x.split()
            dict.__setitem__(self, x[0], int(x[1]))
        self._path.close()

    def check_language(self, word):
        def format_language_label(src: tuple) -> str:
            return src[0][0][-2:]

        return format_language_label(self.lm.predict(word)) == self.lang

    def create_vocab(self, vocab_path):
        words = Counter(regex.findall(r'[{}]+'.format(self.ALPHA), open(self._path).read().lower()))
        with open(vocab_path, 'w+') as words_file:
            [words_file.write(f'{word} {words[word]}\n') for word in words if self.check_language(word) and words[word] != 1]


def fix_spelling(src: str, tgt: str, src_lang: str, tgt_lang: str) -> List[str]:
    """Searches for spelling errors in source and target sentences
    returns corrected sentences
    """

    def correct_english_word(token):
        if token.surface[0].upper():
            return token.surface
        tb_words = TextBlob(token.surface).words
        return basque_spelling.suggest(tb_words[0])[0][0] if tb_words else token.surface

    def correct_basque_word(token):
        if token.surface[0].upper():
            return token.surface
        tb_words = TextBlob(token.surface).words
        return tb_words[0].correct() if tb_words else token.surface

    fixed_src, fixed_tgt = [tokenizer.tokenize(sent, as_token_objects=True) for sent in [src, tgt]]
    # fix src
    for idx, token in enumerate(fixed_src):
        fixed_src[idx].surface = correct_basque_word(token) if src_lang == 'en' else correct_english_word(token)
        fixed_src[idx].casing = pyonmttok.Casing.LOWERCASE if token.casing == pyonmttok.Casing.MIXED else token.casing
    # fix tgt
    for idx, token in enumerate(fixed_tgt):
        fixed_tgt[idx].surface = correct_english_word(token) if tgt_lang == 'eu' else correct_basque_word(token)
        fixed_tgt[idx].casing = pyonmttok.Casing.LOWERCASE if token.casing == pyonmttok.Casing.MIXED else token.casing
    fixed_src, fixed_tgt = tokenizer.detokenize(fixed_src), tokenizer.detokenize(fixed_tgt)
    return fixed_src, fixed_tgt

NUMERICAL_EXPRESSIONS_RE = regex.compile(r'[\[\(\{]?([+-~]?(?:\d+)\s*[+-=\^%\/:\*@]?\s*)+[\]\)\}]?')

def fix_numerical_expressions(src: str, tgt: str, **kwargs) -> List[str]:
    """Replaces numerical expressions in source sentences with corresponding
    ones from target sentences or deletes unmatched numerical expressions that
    only exist in one of paired sentences
    """
    def string_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    fixed_src, fixed_tgt = src, tgt
    src_expressions = [expression.group() for expression in NUMERICAL_EXPRESSIONS_RE.finditer(fixed_src)]
    tgt_expressions = [expression.group() for expression in NUMERICAL_EXPRESSIONS_RE.finditer(fixed_tgt)]
    # remove all numerical expressions
    if not src_expressions or not tgt_expressions:
        if not src_expressions and not tgt_expressions:
            return [fixed_src, fixed_tgt]
        expressions = src_expressions if src_expressions else tgt_expressions
        for expression in expressions:
            fixed_src, fixed_tgt = [regex.sub(regex.escape(expression), '', sent) for sent in [fixed_src, fixed_tgt]]
        return [fixed_src, fixed_tgt]
    similarities = [
        [*expression_pair, string_similarity(expression_pair[0].replace(' ', ''), expression_pair[1].replace(' ', ''))]
        for expression_pair in itertools.product(src_expressions, tgt_expressions)
    ]
    for expression in tgt_expressions:
        expression_pairs = list(filter(lambda sim: sim[1]==expression, similarities))
        expression_pairs.sort(key=lambda sim: sim[2], reverse=True)
        src_expr, tgt_expr, prob = expression_pairs[0]
        if prob > 0.3:  # enough similarity
            # add trailing space if it is present in source expression
            tgt_expr += ' ' if tgt_expr[-1] != ' ' and src_expr[-1] == ' ' else ''
            fixed_src = regex.sub(regex.escape(src_expr), tgt_expr, fixed_src, count=1)
            try:
                src_expressions.remove(src_expr)
            except Exception:
                pass
    # remove unmatched numerical expressions
    for expression in src_expressions:
        fixed_src = fixed_src.replace(expression, '')
    return [fixed_src, fixed_tgt]


ANY_PUNCT_BEG_RE = (regex.compile(r'^[{0}\s]+(?=[^{0}])'.format(regex.escape(string.punctuation))), r'')
ANY_PUNCT_END_RE = (regex.compile(r'(?<=[^{0}])[{0}\s]+$'.format(regex.escape(string.punctuation))), r'.')
CONVENTIONAL_PUNCTUATION_BEG_RE = regex.compile(r'^(?:[\(\"\'])?(\.{1}|[\"\'\(])(?=[^{0}])'.format(regex.escape(string.punctuation), '{2,3}'))
CONVENTIONAL_PUNCTUATION_END_RE = regex.compile(r'(?<=[^{0}])(\.{1}|[\?\!]{2}|[\"\'])(?:[\)\"\'])?$'.format(regex.escape(string.punctuation), '{1,3}', '{1,2}'))

def fix_punctuation(src: str, tgt: str, **kwargs):
    """Corrects punctuation of sentence pairs at the beginning/end by
    either replacing it with one of conventional variants or by removing it
    """
    # ad hoc solution for parentheses and quotes/apostrophes
    for symbol in ['"', '\'', ')']:
        fixed_src, fixed_tgt = [f'{sent[:-2]}{symbol}.' if sent.endswith(f'.{symbol}') else sent for sent in [src, tgt]]
    # fix punctuation at the sentence beginning
    extracted_punct_src_beg, extracted_punct_tgt_beg = ANY_PUNCT_BEG_RE[0].search(fixed_src), ANY_PUNCT_BEG_RE[0].search(fixed_tgt)
    if any([extracted_punct_src_beg, extracted_punct_tgt_beg]):
        extracted_conv_src = CONVENTIONAL_PUNCTUATION_BEG_RE.search(fixed_src) if extracted_punct_src_beg \
            else None
        extracted_conv_tgt = CONVENTIONAL_PUNCTUATION_BEG_RE.search(fixed_tgt) if extracted_punct_tgt_beg \
            else None
        # both have punctuation at the beginning
        if all([extracted_punct_src_beg, extracted_punct_tgt_beg]):
            # punctuation at the sentence start is valid on both sides -> make identical on both sides
            if all([extracted_conv_src, extracted_conv_tgt]):
                fixed_src = f'{extracted_conv_tgt.group()}{fixed_src[extracted_conv_src.end():]}'
            elif any([extracted_conv_src, extracted_conv_tgt]):
                fixed_src = f'{extracted_conv_tgt.group()}{fixed_src[extracted_punct_src_beg.end():]}' if extracted_conv_tgt \
                    else fixed_src
                fixed_tgt = f'{extracted_conv_src.group()}{fixed_tgt[extracted_punct_tgt_beg.end():]}' if extracted_conv_src \
                    else fixed_tgt
            else:
                fixed_src = fixed_src[extracted_punct_src_beg.end():]
                fixed_tgt = fixed_tgt[extracted_punct_tgt_beg.end():]
                #fixed_src, fixed_tgt = [ANY_PUNCT_BEG_RE[0].sub(ANY_PUNCT_BEG_RE[1], sent) for sent in [fixed_src, fixed_tgt]]
        else:
            fixed_src, fixed_tgt = [ANY_PUNCT_BEG_RE[0].sub(ANY_PUNCT_BEG_RE[1], sent) for sent in [fixed_src, fixed_tgt]]
    # fix punctuation at the sentence end
    extracted_punct_src_end, extracted_punct_tgt_end = ANY_PUNCT_END_RE[0].search(fixed_src), ANY_PUNCT_END_RE[0].search(fixed_tgt)
    if any([extracted_punct_src_end, extracted_punct_tgt_end]):
        extracted_conv_src = CONVENTIONAL_PUNCTUATION_END_RE.search(fixed_src) if extracted_punct_src_end \
            else None
        extracted_conv_tgt = CONVENTIONAL_PUNCTUATION_END_RE.search(fixed_tgt) if extracted_punct_tgt_end \
            else None
        # both have punctuation at the end
        if all([extracted_punct_src_end, extracted_punct_tgt_end]):
            # punctuation at the sentence end is valid on both sides -> make identical on both sides
            if all([extracted_conv_src, extracted_conv_tgt]):
                fixed_src = f'{fixed_src[:extracted_conv_src.start()]}{extracted_conv_tgt.group()}'
            elif any([extracted_conv_src, extracted_conv_tgt]):
                fixed_src = f'{fixed_src[:extracted_punct_src_end.start()]}{extracted_conv_tgt.group()}' if extracted_conv_tgt \
                    else fixed_src
                fixed_tgt = f'{fixed_tgt[:extracted_punct_tgt_end.start()]}{extracted_conv_src.group()}' if extracted_conv_src \
                    else fixed_tgt
            else:
                fixed_src = fixed_src[:extracted_punct_src_end.start()+1]
                fixed_tgt = fixed_tgt[:extracted_punct_tgt_end.start()+1]
                #fixed_src, fixed_tgt = [ANY_PUNCT_END_RE[0].sub(ANY_PUNCT_END_RE[1], sent) for sent in [fixed_src, fixed_tgt]]
        else:
            fixed_src, fixed_tgt = [ANY_PUNCT_END_RE[0].sub(ANY_PUNCT_END_RE[1], sent) for sent in [fixed_src, fixed_tgt]]
    else:
        fixed_src += '.'
        fixed_tgt += '.'
    return [fixed_src, fixed_tgt]


# use this to create vocab
if __name__ == '__main__':
    lm = ft.load_model(os.path.join(static.fasttext_path, 'lid.176.bin'))
    spelling = Spelling(sys.argv[1], 'eu', lm)
    spelling.create_vocab(os.path.join(static.spelling_dict_path, 'spelling_correction.words'))
else:
    basque_spelling = Spelling(os.path.join(static.spelling_dict_path, 'spelling_correction.words'), 'eu')
    basque_spelling.load()
    tokenizer = pyonmttok.Tokenizer(mode='aggressive')
