from typing import List, Tuple
import re

# ---------------------------
# Dictionaries / Vocab
# ---------------------------

MONTHS = {
    "january","february","march","april","may","june","july",
    "august","september","october","november","december","jan","feb","mar",
    "apr","jun","jul","aug","sep","sept","oct","nov","dec"
}

EVENT_KEYWORDS = {
    "resign","resigns","resigned","resignation",
    "convict","convicted","acquitted","appeal","sentenced","arrested",
    "killed","dies","death","attack","ban","law","election","vote","verdict",
    "scandal","claim","hoax","rumor","false","misleading","fake",
}

STOPWORDS = {
    "a","an","the","of","to","for","in","on","and","or","but","with","by","from",
    "is","are","was","were","be","being","been","that","this","these","those",
    "at","as","it","its","their","his","her","him","them","they","you","we","i",
    "end","over","under","after","before","during","amid","across","about",
    "years","year","decade","decades","months","month","weeks","week","days","day"
}

# ---------------------------
# Internals
# ---------------------------

def _try_spacy_ner(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Attempt spaCy NER. If spaCy or model is unavailable, return empty signals
    so callers can fall back to regex heuristics.
    Returns: (persons, entities[ORG/GPE/LOC/NORP], dates, verbs_lemmas)
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Graceful fallback trigger for callers
        return [], [], [], []

    doc = nlp(text)
    persons: List[str] = []
    ents: List[str] = []
    dates: List[str] = []

    for e in doc.ents:
        if e.label_ == "PERSON":
            persons.append(e.text.strip())
        elif e.label_ in ("ORG", "GPE", "LOC", "NORP"):
            ents.append(e.text.strip())
        elif e.label_ == "DATE":
            dates.append(e.text.strip())

    verbs = [t.lemma_ for t in doc if t.pos_ in ("VERB", "AUX")]
    return persons, ents, dates, verbs


def _regex_backoff(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Lightweight heuristic fallback when spaCy isn't available.
    - PERSONS ~ first one or two multi-token capitalized phrases
    - ENTS    ~ remaining capitalized phrases (limited)
    - DATES   ~ {year, month}; year regex fixed to capture full year
    - VERBS   ~ event-like keywords (from EVENT_KEYWORDS)
    """
    # Capitalized phrases (up to 4 tokens)
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)

    # crude split: assume the first capitalized chunk with a space is likely PERSON
    persons = [c for c in caps if " " in c][:2]
    other_ents = [c for c in caps if c not in persons][:3]

    lower = text.lower()

    # FIX: capture full years (e.g., "2018"), not only the "19"/"20" group
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    months = [w for w in re.findall(r"\b[a-zA-Z]{3,9}\b", lower) if w in MONTHS]

    dates: List[str] = []
    if years:
        # keep the most recent year lexically; good enough here
        dates.append(max(years))
    if months:
        dates.append(months[-1])

    tokens = re.findall(r"\b[a-zA-Z]+\b", lower)
    verbs = [t for t in tokens if t in EVENT_KEYWORDS]

    return persons, other_ents, dates, verbs


def _clean_terms(parts: List[str]) -> List[str]:
    """
    Split on whitespace, drop stopwords, keep simple alpha/hyphen tokens,
    and de-duplicate while preserving order.
    """
    out: List[str] = []
    for p in parts:
        for w in re.split(r"\s+", p.strip()):
            wl = w.lower()
            if wl and wl not in STOPWORDS and re.match(r"[a-zA-Z\-]+$", w):
                out.append(w)

    # unique, preserve order
    seen, uniq = set(), []
    for w in out:
        wl = w.lower()
        if wl not in seen:
            uniq.append(w)
            seen.add(wl)
    return uniq

# ---------------------------
# Public API
# ---------------------------

def expand_queries_general(claim_text: str, max_terms: int = 8) -> List[str]:
    """
    Build a short list of general-purpose search queries for open-web retrieval.
    Strategy (kept as-is, with safer NER fallback and fixed year extraction):
      1) Prefer PERSON (quoted) or leading ORG/GPE + short key terms
      2) Add an unquoted short form
      3) Add quoted PERSON-only variant (if not already present)
      4) Backoff: "PERSON verb" or "ENTITY verb"
      5) Final backoff: first 3 non-stopword tokens from the claim
    """
    persons: List[str]; ents: List[str]; dates: List[str]; verbs: List[str]
    pedsv: Tuple[List[str], List[str], List[str], List[str]]

    # Try spaCy; gracefully fallback to regex
    pedsv = _try_spacy_ner(claim_text)
    if not any(pedsv):
        pedsv = _regex_backoff(claim_text)
    persons, ents, dates, verbs = pedsv

    # prefer PERSON first; then an ORG/GPE; then verb; then a date
    lead = " ".join(persons[:1]) if persons else " ".join(ents[:1])
    verb = verbs[0] if verbs else ""
    date = dates[0] if dates else ""

    key_items = [lead, verb, date]
    key = _clean_terms([x for x in key_items if x])[:max_terms]
    short = " ".join(key)

    queries: List[str] = []
    if lead:
        queries.append(f"\"{lead}\" {short}".strip())
    if short:
        queries.append(short)
    if lead and f"\"{lead}\"" not in queries:
        queries.append(f"\"{lead}\"")

    # backoff: person + verb, or entity + verb
    if persons and verb:
        queries.append(f"\"{persons[0]}\" {verb}")
    elif ents and verb:
        queries.append(f"\"{ents[0]}\" {verb}")

    # final backoff: first 3 non-stopword tokens from the claim
    tokens = _clean_terms(re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", claim_text))
    if tokens:
        head = " ".join(tokens[:3])
        if head and head not in queries:
            queries.append(head)

    # unique, order-preserving
    seen, uniq = set(), []
    for q in queries:
        qn = q.strip()
        if qn and qn.lower() not in seen:
            uniq.append(qn)
            seen.add(qn.lower())
    return uniq


def expand_queries_fc(claim_text: str, max_terms: int = 6) -> List[str]:
    """
    Build short, unquoted queries tailored for FactCheck-like tools/search.
    Strategy (kept as-is, with safer NER fallback and fixed year extraction):
      - person + action + year (if available)
      - person + action (no year)
      - final backoff: top 2 non-stopword tokens
    """
    persons: List[str]; ents: List[str]; dates: List[str]; verbs: List[str]
    pedsv: Tuple[List[str], List[str], List[str], List[str]]

    # Try spaCy; gracefully fallback to regex
    pedsv = _try_spacy_ner(claim_text)
    if not any(pedsv):
        pedsv = _regex_backoff(claim_text)
    persons, ents, dates, verbs = pedsv

    lead = persons[:1] or ents[:1]
    parts: List[str] = []
    if lead:
        parts.append(lead[0])
    if verbs:
        parts.append(verbs[0])
    if dates:
        parts.append(dates[0])

    cleaned = _clean_terms(parts)[:max_terms]

    # two forms: person action year, and person action (no year)
    q1 = " ".join(cleaned)
    q2 = " ".join(cleaned[:-1]) if len(cleaned) > 1 else q1
    queries = [q for q in (q1, q2) if q]

    # final backoff: top 2 tokens from the claim
    tokens = _clean_terms(re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", claim_text))[:2]
    if tokens:
        q3 = " ".join(tokens)
        if q3 and q3 not in queries:
            queries.append(q3)

    # unique, order-preserving
    seen, uniq = set(), []
    for q in queries:
        qn = q.strip()
        if qn and qn.lower() not in seen:
            uniq.append(qn)
            seen.add(qn.lower())
    return uniq
