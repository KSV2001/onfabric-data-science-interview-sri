"""
Configuration for preprocessing search / event history.
"""

from typing import Final

# URL detection
URL_REGEX_PATTERN: Final[str] = r"https?://\S+"

# Tokenization
TOKEN_SPLIT_PATTERN: Final[str] = r"\W+"

# URL normalization
GOOGLE_REDIRECT_DOMAIN: Final[str] = "google.com"
GOOGLE_REDIRECT_PATH: Final[str] = "/url"

# Document frequency filtering
MAX_DF_RATIO: Final[float] = 0.3
