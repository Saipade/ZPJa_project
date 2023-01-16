import html
import regex
from typing import List

# html entities replacements
HTML_NAMED_ENTITIES_REPLACEMENTS = {f'&{entity}': replacement for entity, replacement in html.entities.html5.items() if entity.endswith(';')}
# latex entities replacements
LATEX_ENTITIES_REPLACEMENTS = {
    '\\#': '#',  '\\% ': '%', '\\&': '&', '\\’': '’', '\\.': '.', '\\:': ':', '\\(': '(', '\\)': ')', '\\{': '{',
    '\\}': '}', '\\[': '[', '\\]': ']', '\\_': '_', '\\˜': '˜', '\\|': '|', '\\aa': 'å', '\\AA': 'Å', '\\ae': 'æ',
    '\\AE': 'Æ', '\\aleph': 'ℵ', '\\alpha': 'α', '\\and': ' ', '\\beta': 'β', '\\pi': 'π', '\\Pi': 'Π', '\\phi': 'φ', '\\partial': '∂',
    '\\psi': 'ψ', '\\Psi': 'Ψ', '\\sim': '~', '\\n': '', '\\t': '', '\\r': '', '\\\\': '\\'
}

LATEX_TEXT_MODIFIERS = [
    r'\\textmd', r'\\textbf', r'\\textup', r'\\textit', r'\\textsl', r'\\textsc',
    r'\\textrm', r'\\textsf', r'\\texttt',
    r'\\tiny', r'\\scriptsize', r'\\footnotesize', r'\\small', r'\\normalsize', r'\\large', r'\\Large', r'\\LARGE', r'\\huge', r'\\Huge'
]

LATEX_TEXT_MODIFIERS_RE = regex.compile(f'({"|".join([mod for mod in LATEX_TEXT_MODIFIERS])})')

# general html entity regex (either numeric or named)
def unescape_html(match: regex.Match) -> str:
    """Unescape html callback function
    """
    match_group = match.group(0)
    return HTML_NAMED_ENTITIES_REPLACEMENTS[match_group] if match_group in HTML_NAMED_ENTITIES_REPLACEMENTS \
        else html.unescape(match_group) if match_group.startswith('&#') \
        else match_group

HTML_ENTITY_RE = (regex.compile(r'&#?[[:alnum:]]+;'), unescape_html)
# regex for markup languages tags (html, xml, ...)
MARKUP_TAGS_RE = (regex.compile(r'<[[:alnum:]\/]+>'), r'')

# general regex for latex expression
def process_markup(match: regex.Match) -> str:
    """Process markup callback function
    """
    match_group = match.group(0)
    return LATEX_ENTITIES_REPLACEMENTS[match_group] if match_group in LATEX_ENTITIES_REPLACEMENTS \
        else LATEX_TEXT_MODIFIERS_RE.sub(r'', match_group).strip('{}').strip() if LATEX_TEXT_MODIFIERS_RE.search(match_group) \
        else '' if LATEX_GENERAL_RE[0].search(match_group) \
        else match_group

LATEX_GENERAL_RE = (regex.compile(r'\\[[:alnum:]]+(\{[[:alnum:]]*\})*'), process_markup)
# hyperlink regex
HYPERLINK_RE = (regex.compile(r'(https?://)?(www\.)(\w+\.)+[a-zA-Z0-9#\?%]{2,10}/?'), r'')

def replace_markup(src: str, tgt: str, **kwargs) -> List[str]:
    """Replaces latex escape sequences and html entities
    """
    fixed_src, fixed_tgt = src, tgt
    for replacement in [HYPERLINK_RE, MARKUP_TAGS_RE, HTML_ENTITY_RE, LATEX_GENERAL_RE]:
        fixed_src, fixed_tgt = [replacement[0].sub(replacement[1], sent) for sent in [fixed_src, fixed_tgt]]
    return [fixed_src, fixed_tgt]

