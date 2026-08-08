from typing import List

from baseline_solution import extract_pii as _extract_pii
from pii_item import PIIItem


def extract_pii(text: str) -> List[PIIItem]:
    return _extract_pii(text)
