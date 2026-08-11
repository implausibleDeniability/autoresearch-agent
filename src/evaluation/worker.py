import importlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Mapping

from src.evaluation.models import PIIItem

MAX_CONCURRENT_DOCUMENTS = 8


def extract_documents(
    texts: Mapping[str, str],
    *,
    module_name: str,
) -> Dict[str, List[PIIItem]]:
    extract_pii: Callable[[str], List[PIIItem]] = importlib.import_module(module_name).extract_pii
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOCUMENTS) as executor:
        predictions = executor.map(extract_pii, texts.values())
        return dict(zip(texts, predictions))
