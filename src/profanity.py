from typing import List, Tuple
from profanity_check import predict as predict_profanity

def remove_profanity(src: str, tgt: str, src_lang, **kwargs) -> Tuple[List]:
    """Removes pairs with profane content on English side from dataset
    """
    to_check = src if src_lang == 'en' else tgt
    return ['', ''] if predict_profanity([to_check]) else [src, tgt]
