"""Companies House company numbers.

A company number is 8 characters: either 8 digits (England and Wales, e.g. `00445790`) or a
2-letter prefix plus 6 digits (e.g. `SC123456` Scotland, `NI012345` Northern Ireland,
`OC300001` LLPs). Sources disagree on leading zeros (`566433` vs `00566433`), so everything is
normalised before it is used as a join key.
"""

import re

_COMPANY_NUMBER_RE = re.compile(r"^[A-Z0-9]{8}$")


def normalise_company_number(raw: str) -> str:
    """Upper-case, strip whitespace and left-pad purely numeric numbers to 8 digits.

    Raises ValueError for anything that cannot be a company number.
    """
    value = raw.strip().upper()
    if value.isdigit():
        value = value.zfill(8)
    if not _COMPANY_NUMBER_RE.match(value):
        raise ValueError(f"Not a valid company number: {raw!r}")
    return value
