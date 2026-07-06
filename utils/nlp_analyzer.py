"""
NLP/ML analysis layer.
Extracts keywords, key phrases, named entities, and semantic features
from Wikipedia concept data.
Uses: NLTK, scikit-learn (TF-IDF), spaCy (optional), sentence-transformers.
Falls back gracefully if heavy dependencies are absent.
"""
import re
import math
import string
from collections import Counter

# ── Optional imports with graceful fallback ─────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    # Download required NLTK data silently
    for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    _NLTK = True
except ImportError:
    _NLTK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY = True
except Exception:
    _SPACY = False


# ── Helpers ─────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "of","in","on","at","to","for","with","by","from","up","about","into","through",
    "it","its","this","that","these","those","and","or","but","if","when","where",
    "which","who","what","how","all","each","every","both","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","just","because","as","until","while","he","she","they","we","you","i",
    "also","however","therefore","thus","hence","whereas","although","though",
}

def _clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)           # remove citation markers
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _simple_tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _tfidf_keywords(text: str, top_n: int = 15) -> list[str]:
    """TF-IDF over sentences to pull out salient single tokens."""
    if not _SKLEARN:
        return []
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) < 3:
        return []
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=200,
            token_pattern=r"\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        )
        vec.fit(sentences)
        scores = zip(vec.get_feature_names_out(), vec.idf_)
        # lower IDF = more important
        ranked = sorted(scores, key=lambda x: x[1])
        return [w for w, _ in ranked[:top_n]]
    except Exception:
        return []


def _extract_entities_spacy(text: str) -> list[str]:
    if not _SPACY:
        return []
    doc = _nlp(text[:5000])
    entities = [ent.text for ent in doc.ents if ent.label_ not in ("CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY")]
    return list(dict.fromkeys(entities))[:10]


def _extract_entities_nltk(text: str) -> list[str]:
    if not _NLTK:
        return []
    try:
        tokens = word_tokenize(text[:3000])
        tagged = nltk.pos_tag(tokens)
        entities = [word for word, pos in tagged if pos in ("NNP", "NNPS") and word.lower() not in _STOPWORDS]
        return list(dict.fromkeys(entities))[:10]
    except Exception:
        return []


def _extract_noun_phrases(text: str) -> list[str]:
    """Simple regex-based noun phrase extractor as fallback."""
    # Pattern: optional adj + nouns
    pattern = r"\b([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,2})\b"
    candidates = re.findall(pattern, text[:4000])
    freq = Counter(candidates)
    return [phrase for phrase, _ in freq.most_common(10) if len(phrase.split()) <= 3 and phrase.lower() not in _STOPWORDS]


def _lemmatize(tokens: list[str]) -> list[str]:
    if _NLTK:
        try:
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            pass
    return tokens


def _extract_domain(text: str, title: str) -> str:
    """Heuristic domain classification."""
    text_lower = (title + " " + text[:500]).lower()
    domains = {
        "physics": ["quantum","particle","energy","force","wave","field","relativity","atom","photon"],
        "biology": ["cell","organism","dna","gene","protein","evolution","species","enzyme","tissue"],
        "computer science": ["algorithm","software","data structure","network","computing","binary","processor","code"],
        "mathematics": ["theorem","proof","function","equation","topology","algebra","calculus","manifold"],
        "economics": ["market","trade","capital","labor","price","supply","demand","currency","inflation"],
        "history": ["war","empire","century","revolution","civilization","dynasty","colony","treaty"],
        "philosophy": ["ethics","logic","epistemology","ontology","metaphysics","consciousness","reason"],
        "chemistry": ["molecule","reaction","element","compound","bond","acid","base","catalyst"],
    }
    scores = {}
    for domain, keywords in domains.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze_concept(concept_data: dict) -> dict:
    """
    Analyzes Wikipedia concept data and returns a features dict:
    {
        keywords: list[str],
        entities: list[str],
        noun_phrases: list[str],
        domain: str,
        core_functions: list[str],
        structural_components: list[str],
        summary_sentences: list[str],
    }
    """
    title = concept_data.get("title", "")
    summary = _clean_text(concept_data.get("summary", ""))
    full_text = _clean_text(concept_data.get("full_text", ""))

    combined = summary + " " + full_text[:3000]

    # Keywords via TF-IDF (preferred) or frequency
    tfidf_kw = _tfidf_keywords(combined, top_n=12)
    if not tfidf_kw:
        tokens = _simple_tokenize(combined)
        tfidf_kw = [w for w, _ in Counter(tokens).most_common(12)]

    # Named entities
    entities = _extract_entities_spacy(combined) or _extract_entities_nltk(combined)

    # Noun phrases
    noun_phrases = _extract_noun_phrases(combined)

    # Domain
    domain = _extract_domain(combined, title)

    # Summary sentences (first 5 informative ones)
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    summary_sentences = [s for s in sentences if len(s.split()) > 6][:5]

    # Structural components: nouns in summary that are likely "parts"
    part_words = re.findall(r"\b(?:system|component|element|part|layer|stage|phase|step|process|mechanism|structure|module|unit|node|link|channel|signal|state|function|role|property)\b", combined.lower())
    component_context = []
    for match in re.finditer(r"(\w[\w\s]{0,20}(?:system|component|element|layer|stage|phase|mechanism|structure|module|unit|node|link|channel|signal|state|function|property)\b)", combined, re.IGNORECASE):
        phrase = match.group(0).strip()
        if 2 <= len(phrase.split()) <= 4:
            component_context.append(phrase)
    structural_components = list(dict.fromkeys(component_context))[:8]

    # Core functions: verb-heavy phrases
    func_patterns = re.findall(r"\b(?:used to|enables|allows|provides|performs|converts|transmits|processes|stores|generates|measures|controls|regulates|connects|maps|transforms)\b.{5,60}", combined, re.IGNORECASE)
    core_functions = [re.sub(r"\s+", " ", f.strip()) for f in func_patterns[:6]]

    return {
        "keywords": tfidf_kw,
        "entities": entities,
        "noun_phrases": noun_phrases,
        "domain": domain,
        "core_functions": core_functions,
        "structural_components": structural_components,
        "summary_sentences": summary_sentences,
        "title": title,
    }
