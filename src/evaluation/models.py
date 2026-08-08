from dataclasses import dataclass
from typing import Tuple

PIIValues = Tuple[str, ...]


@dataclass(frozen=True)
class PIIItem:
    first_name: PIIValues = ()
    middle_name: PIIValues = ()
    last_name: PIIValues = ()
    age: PIIValues = ()
    birthdate: PIIValues = ()
    phone: PIIValues = ()
    email: PIIValues = ()
    social_network_identifier: PIIValues = ()
    location: PIIValues = ()
    ssn: PIIValues = ()
