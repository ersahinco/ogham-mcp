"""Date and entity extraction for memory enrichment.

Pure regex, no LLM calls. Runs at store_memory time to enrich
metadata (dates) and tags (entities) automatically.
"""

import logging
import re
from datetime import datetime

import parsedatetime
from stop_words import AVAILABLE_LANGUAGES, get_stop_words

logger = logging.getLogger(__name__)

# parsedatetime calendar for relative date parsing (zero deps, English)
_PDT_CAL = parsedatetime.Calendar()

# --- Stopwords (34 languages, loaded once) ---

_STOP_WORDS: set[str] = set()
for _lang in AVAILABLE_LANGUAGES:
    _STOP_WORDS.update(get_stop_words(_lang))

# --- Date extraction ---

_ISO_DATE = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_NATURAL_DATE = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}\b"
    rf"|\b\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}\b",
    re.IGNORECASE,
)

# Patterns for relative date phrases to feed to parsedatetime
_RELATIVE_PHRASES = re.compile(
    r"\b((?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|week|month|year)"
    r"|yesterday|today|tomorrow"
    r"|\d+\s+(?:days?|weeks?|months?|years?)\s+ago"
    r"|in\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)

TEMPORAL_KEYWORDS = frozenset(
    {
        "when",
        "date",
        "time",
        "ago",
        "last",
        "before",
        "after",
        "during",
        "since",
        "until",
        "yesterday",
        "tomorrow",
        "week",
        "month",
        "year",
    }
)


def extract_dates(content: str) -> list[str]:
    """Extract date strings from content. Returns sorted normalised ISO dates."""
    dates: set[str] = set()

    for match in _ISO_DATE.finditer(content):
        dates.add(match.group(1).replace("/", "-"))

    for match in _NATURAL_DATE.finditer(content):
        try:
            raw = re.sub(r"(st|nd|rd|th)", "", match.group(0))
            raw = raw.replace(",", "").strip()
            for fmt in ("%B %d %Y", "%d %B %Y", "%b %d %Y", "%d %b %Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    dates.add(dt.strftime("%Y-%m-%d"))
                    break
                except ValueError:
                    continue
        except Exception:
            logger.debug("Failed to parse natural date: %s", match.group(0))
            continue

    # Relative dates via parsedatetime ("last Tuesday", "yesterday", "two weeks ago")
    # Only attempt if no absolute dates found and content has temporal keywords
    if not dates:
        for phrase in _RELATIVE_PHRASES.findall(content):
            try:
                result, status = _PDT_CAL.parse(phrase)
                if status:
                    dt = datetime(*result[:6])
                    dates.add(dt.strftime("%Y-%m-%d"))
            except Exception:
                logger.debug("Failed to parse relative date: %s", phrase)
                continue

    return sorted(dates)


def has_temporal_intent(query: str) -> bool:
    """Check if a query has temporal intent (when, date, time, etc.)."""
    words = set(query.lower().split())
    return bool(words & TEMPORAL_KEYWORDS)


# --- Shared regex patterns (used by both importance and entity extraction) ---

_CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+\b")
_FILE_PATH = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")
_ERROR_TYPE = re.compile(r"\b\w*(?:Error|Exception)\b")

# --- Importance scoring ---

_DECISION_RE = re.compile(
    r"\b(?:decided|chose|choosing|switched|migrated|selected|picked|opted)\b",
    re.IGNORECASE,
)
_ARCHITECTURE_RE = re.compile(
    r"\b(?:design|pattern|refactor|architecture|restructur|modular|decouple)\b",
    re.IGNORECASE,
)


def compute_importance(content: str, tags: list[str] | None = None) -> float:
    """Score content importance based on signals. Returns 0.0-1.0.

    No LLM needed -- pure regex on the text itself.
    """
    score = 0.2  # base score

    if _DECISION_RE.search(content):
        score += 0.3
    if _ERROR_TYPE.search(content):
        score += 0.2
    if _ARCHITECTURE_RE.search(content):
        score += 0.2
    if _FILE_PATH.search(content):
        score += 0.1
    if "```" in content or "`" in content:
        score += 0.1
    if len(content) > 500:
        score += 0.1
    if tags and len(tags) >= 3:
        score += 0.1

    return min(score, 1.0)


# --- Entity extraction ---
_PUNCT = str.maketrans("", "", ".,!?:;\"'()")


def extract_entities(content: str) -> list[str]:
    """Extract named entities from content for tagging.

    Returns sorted list of prefixed tags, capped at 15:
      person:FirstName LastName
      entity:CamelCaseName
      file:path/to/file.ext
      error:SomeError
    """
    entities: set[str] = set()

    for m in _CAMEL_CASE.finditer(content):
        entities.add(f"entity:{m.group(0)}")

    for i, m in enumerate(_FILE_PATH.finditer(content)):
        if i >= 5:
            break
        entities.add(f"file:{m.group(0)}")

    for m in _ERROR_TYPE.finditer(content):
        entities.add(f"error:{m.group(0)}")

    # Person names: two consecutive capitalised words not in stopwords
    words = content.split()
    for i in range(len(words) - 1):
        w1 = words[i].translate(_PUNCT)
        w2 = words[i + 1].translate(_PUNCT)
        if (
            w1
            and w2
            and w1[0].isupper()
            and w2[0].isupper()
            and w1.isalpha()
            and w2.isalpha()
            and w1.lower() not in _STOP_WORDS
            and w2.lower() not in _STOP_WORDS
            and len(w1) > 1
            and len(w2) > 1
        ):
            entities.add(f"person:{w1} {w2}")

    return sorted(entities)[:15]
