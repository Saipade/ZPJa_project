import regex
from typing import List

# List of regexes with their replacements

# apostrophes
# with contractions e.g. I've, I'm ...
DOUBLE_APOSTROPHE_CONTRACTIONS_RE = (regex.compile(r'(?<=[[:alnum:]])\s*\'\'\s*(?=(s|m|d|ll|re|ve|t))', flags=regex.IGNORECASE), r"'")
# general
DOUBLE_APOSTROPHE_RE = (regex.compile(r'[\u2018-\u201B\']{2,}'), r'"')
# normalize apostrophe
NORMALIZE_APOSTROPHE_RE = (regex.compile(r'[\u2018-\u201B\`]+'), r"'")

# normalize quotes; guillemets are not standard symbols for quotation in basque
# https://download.microsoft.com/download/c/6/0/c608b32d-8d04-44a8-a42a-4b0401bc3115/eus-esp-StyleGuide.pdf
NORMALIZE_QUOTES_RE = (regex.compile(r'([\u00AB\u00BB\u2039\u203A\u201C-\u201F\"]+|\<\<|\>\>)'), r'"')

# normalize hyphen
# https://jkorpela.fi/dashes.html
HORMALIZE_HYPHEN_RE = (regex.compile(r'[\u058A\u05BE\u1400\u1806\u2010\u2011\u2013\u207B\u208B\u2212\u2E17\uFE58\uFE63\uFF0D]+'), r'-')

# normalize dash
# https://jkorpela.fi/dashes.html
NORMALIZE_DASH_RE = (regex.compile(r'[\u2013\u2015\u2053\u2E3A\u2E3B]+'), r'—')

# enumeration or dash/hyphen at the start of sentence. e.g.
# a) Hi, Mark! -> Hi, Mark!
# - Hello! -> Hello!
LIST_ENUMERATION_RE = (regex.compile(r'(^\s*([\(\[\{]+)?\s*(\w|\d){1,4}\s*[\)\]\}:\.;]+|^\s*[\-—\*]+\s*)+'), r'')

# spacing
# excessive spacing before/after punctuation
PUNCTUATION_SPACING_NORMALIZATION_RE = (regex.compile(r'(\s+(?=[\)\]\}\.\,\:\;\!\?])|(?<=[\(\[\{])\s+)'), r'')
# empty stuff (different kinds of spaces)
EMPTY_RE = (regex.compile(r'[\u2000-\u200F\u00A0\u00AD]'), r'')

# remove all numbers and timestamps that are inside parentheses
NUMBER_INSIDE_PARENS = (regex.compile(r'([\(\[\{]+)([\d\/\:\-]+)([\)\}\]]+)'), r'')
# part of word or word is inside parens e.g.
# (h)ello -> hello
# (hello) -> hello
PART_INSIDE_PARENS = (regex.compile(r'(([\(\[\{]+)([a-zA-Z]+)([\)\}\]]+)(?=\w)|([\(\[\{]+)([a-zA-Z]+)([\)\}\]]+))'), r'\2')
# some text inside double (or more) parenthesis e.g.
# Zorionekoak ikusi gabe sinesten dutenak [[Joanen Ebanjelioa]] -> Zorionekoak ikusi gabe sinesten dutenak
TEXT_INSIDE_DOUBLE_PARENS = (regex.compile(r'([\(\[\{]{2,})([^\(\[\{]*)([\)\}\]]{2,})'), r'')

# entire sentence inside parenthesis e.g
# (Hello, Mark!) -> Hello, Mark!
SENTENCE_INSIDE_PARENS_RE = (regex.compile(r'^\s*([\(\[\{]*)([^)\]\}]+)([\)\]\}]+)?\s*$'), r'\2')

# ellipsis
ELLIPSIS_RE = (regex.compile(r'(…\s*)+'), r'...')
EXCESSIVE_ELEPSIS_RE = (regex.compile(r'(\.{3,}\s*)+'), r'...')

# trash symbols
REDUNDANT_SYMBOLS_RE = (regex.compile(r'[→←¿¡•§·_个♪º°�▲\|]+'), r'')

# stolen from https://www.kaggle.com/code/takanorihasebe/text-cleaning-bert-and-transformer#Removing-emoji-tags
EMOJI_RE = (regex.compile(
    "(["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+|[%:;]-?[\]\)D\^])", flags=regex.UNICODE
), r'')

# all substitutions listed
substisutions = [
    ELLIPSIS_RE,
    EXCESSIVE_ELEPSIS_RE,
    NORMALIZE_APOSTROPHE_RE,
    NORMALIZE_QUOTES_RE,
    DOUBLE_APOSTROPHE_CONTRACTIONS_RE,
    HORMALIZE_HYPHEN_RE,
    NORMALIZE_DASH_RE,
    REDUNDANT_SYMBOLS_RE,
    LIST_ENUMERATION_RE,
    EMOJI_RE,
    EMPTY_RE,
    NUMBER_INSIDE_PARENS,
    PART_INSIDE_PARENS,
    SENTENCE_INSIDE_PARENS_RE,
    TEXT_INSIDE_DOUBLE_PARENS,
    PUNCTUATION_SPACING_NORMALIZATION_RE,
    DOUBLE_APOSTROPHE_RE,
]

def apply_substitutions(src: str, tgt: str) -> List[str]:
    """Applies substitutions provided by `substisutions`
    """
    fixed_src, fixed_tgt = src, tgt
    for substitution in substisutions:
        fixed_src, fixed_tgt = [substitution[0].sub(substitution[1], sent).strip() for sent in [fixed_src, fixed_tgt]]
    return [fixed_src, fixed_tgt]


paired_symbols = [
    ('(', ')'),
    ('[', ']'),
    ('{', '}'),
    ('"', '"')
]

WORDS_INSIDE_PARENS_RE = (regex.compile(r'([\[\(\{])([a-zA-Z\t\.\s]+)([\]\)\}])'), r'\2')


def process_unpaired_symbols(src: str, tgt: str) -> List[str]:
    """If any of sentences in pair has unpaired, removes
    corresponding signs from both sentences.
    """
    fixed_src, fixed_tgt = src, tgt
    for symbols in paired_symbols:
        # replace multiple concurrent symbols
        fixed_src, fixed_tgt = [regex.sub(regex.escape(symbols[0]+'{2,}'), symbols[0], sent) for sent in [fixed_src, fixed_tgt]]
        fixed_src, fixed_tgt = [regex.sub(regex.escape(symbols[1]+'{2,}'), symbols[1], sent) for sent in [fixed_src, fixed_tgt]]

        r = regex.compile(r'(^[^{0}]*[{0}][^{1}]+$|^[^{0}]*[{1}]+$)' \
            .format(symbols[0], symbols[1]))
        if any([r.search(sent) for sent in [fixed_src, fixed_tgt]]):
            repl = regex.compile(r'({}|{})'.format(regex.escape(symbols[0]), regex.escape(symbols[1])))
            fixed_src, fixed_tgt = [repl.sub('', sent) for sent in [fixed_src, fixed_tgt]]

    found = [WORDS_INSIDE_PARENS_RE[0].search(sent) for sent in [fixed_src, fixed_tgt]]
    if any(found):
        if all(found):
            return [fixed_src, fixed_tgt]
        fixed_src = WORDS_INSIDE_PARENS_RE[0].sub(WORDS_INSIDE_PARENS_RE[1], fixed_src) if found[0] else fixed_src
        fixed_tgt = WORDS_INSIDE_PARENS_RE[0].sub(WORDS_INSIDE_PARENS_RE[1], fixed_tgt) if found[1] else fixed_tgt
    return [fixed_src, fixed_tgt]


FIRST_ALPHA_RE = regex.compile(r'([^[:alpha:]]*)([[:alpha:]])')

def capitalize_first_alpha(src: str, tgt: str) -> List[str]:
    """Capitalizes first alpha character in src/tgt sentecnes
    """
    pair = [src, tgt]
    if any(FIRST_ALPHA_RE.search(sent).group(2).isupper() for sent in pair if FIRST_ALPHA_RE.search(sent)):
        for idx, sent in enumerate(pair):
            if (res := FIRST_ALPHA_RE.search(sent)) is None:
                continue
            pair[idx] = f'{sent[:res.span(2)[0]]}{res.group(2).capitalize()}{sent[res.span(2)[1]:]}'
    return pair


def lowercase_uppercase(src: str, tgt: str, **kwargs) -> List[str]:
    """Lowercases the words fully written in uppercase
    """
    fixed_src = ' '.join([word.lower() if word.isupper() else word for word in src.split()])
    fixed_tgt = ' '.join([word.lower() if word.isupper() else word for word in tgt.split()])
    return [fixed_src, fixed_tgt]


def normalize(src: str, tgt: str, **kwargs) -> List[str]:
    """Normalizes both source and target sentences
    """
    normalized_src, normalized_tgt = src.strip(), tgt.strip()
    for normalization_fun in [apply_substitutions, process_unpaired_symbols, capitalize_first_alpha, lowercase_uppercase]:
        normalized_src, normalized_tgt = [sent.strip() for sent in normalization_fun(normalized_src, normalized_tgt)]
    return [normalized_src, normalized_tgt]



# example
if __name__ == "__main__":
    src_sents_euen = [
        'zer nahi den den (zernahi',
        '"Kaixo mundua!.... . !"', "((Donald Sutherland: ''[Hemen] zuekin egoteagatik poz handia sentitzen dut''))", '(...) Jo. batez ere, Jainkoaren Erreinuaren eta haren justiziaren bila; gainerako gehigarritzat emango zaizue.',
        'Krisi politikoak, ordea, jarraituko du.', 'Lehenengo (    klase¡a be¡zeroarentzako )ze¡rbitzua – ( gomendatzen produktu handi bat );',
        'Niri jendea margotzea gustatzen zait.', 'Federazioarekin borrokatu behar izan genuen lehiaketa hartan parte hartzeko baimena eman ziezadaten.',
        '7 * 24 online zerbitzuak.', 'Apustu bat irabazi dut.', '● Beharrizan bat ase nahi izateak', ' Ezin dut sinetsi oraindik ez nuela errezeta goxo eta errez hau publikatu 😁'
    ]

    tgt_sents_euen = [
        'Allāh will do whatever He Wishes.',
        '"Hello world!"', "((Donald Sutherland: ''It is a great joy to be here with you''))", 'First seek God\'s kingdom and his righteousness, and all these things will be given to you as well.'
        'so, the political fight will continue…', 'I\'\'ve Delivering a first class service or an amazing product...',
        '¡ i like to paint p¿eop¿le.', 'we even had to get a federal court order in order to be allowed to hold the rally.',
        'We have 24/7 online service.', 'In fact, I cannot believe that I have not already posted this recipe.'
    ]
    for sents in zip(src_sents_euen, tgt_sents_euen):
        print(normalize(sents[0], sents[1]))

