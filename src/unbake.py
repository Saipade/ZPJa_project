from typing import List, Tuple
from ftfy import fix_encoding
from typing import Tuple, List

def unbake(src: str, tgt: str, **kwargs) -> List[str]:
    """Applies ftfy's fix encoding function on pairs of sentences
    """
    return [fix_encoding(sent) for sent in [src, tgt]]
