"""
Concept Mapping Layer

Maps structural and functional elements of Concept 2 (reference)
onto Concept 1 (target) to build an analogy scaffold.

No blending is performed — this is purely a directional alignment:
    Concept 2 element → explains → Concept 1 element
"""
import re
from difflib import SequenceMatcher


# ── Similarity helpers ───────────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    ta = set(re.findall(r"\b\w{3,}\b", a.lower()))
    tb = set(re.findall(r"\b\w{3,}\b", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _sim(a: str, b: str) -> float:
    return max(_token_overlap(a, b), _seq_sim(a, b) * 0.5)


# ── Role templates ──────────────────────────────────────────────────────────
#  Maps broad semantic categories to common roles so we can find cross-concept alignments.

_ROLE_PATTERNS = {
    "carrier / medium": re.compile(r"\b(medium|carrier|channel|wire|cable|path|wave|signal|conduit|pipe|vessel|track|road|highway)\b", re.I),
    "sender / source": re.compile(r"\b(sender|source|emitter|transmitter|producer|generator|origin|caller|encoder)\b", re.I),
    "receiver / destination": re.compile(r"\b(receiver|destination|target|decoder|consumer|listener|sink|endpoint|recipient)\b", re.I),
    "controller / regulator": re.compile(r"\b(controller|regulator|manager|governor|scheduler|router|switch|gate|valve|hub)\b", re.I),
    "storage": re.compile(r"\b(storage|memory|database|cache|register|repository|store|buffer|accumulator)\b", re.I),
    "processor / transformer": re.compile(r"\b(processor|transform|convert|encode|decode|compute|process|translate|interpret)\b", re.I),
    "input": re.compile(r"\b(input|stimulus|query|request|signal|trigger|data|feed)\b", re.I),
    "output": re.compile(r"\b(output|result|response|product|reply|answer|yield)\b", re.I),
}


def _assign_roles(phrases: list[str]) -> dict[str, list[str]]:
    """Tag each phrase with a semantic role category."""
    role_map: dict[str, list[str]] = {role: [] for role in _ROLE_PATTERNS}
    for phrase in phrases:
        for role, pattern in _ROLE_PATTERNS.items():
            if pattern.search(phrase):
                role_map[role].append(phrase)
    return role_map


def _keywords_to_phrases(features: dict) -> list[str]:
    """Combine all textual features into a flat phrase list."""
    items = []
    items.extend(features.get("keywords", []))
    items.extend(features.get("structural_components", []))
    items.extend(features.get("noun_phrases", []))
    items.extend(features.get("core_functions", []))
    return [i for i in items if i and len(i.strip()) > 1]


# ── Main mapping function ────────────────────────────────────────────────────

def map_concepts(features1: dict, features2: dict) -> dict:
    """
    Build a directional mapping: Concept 2 → Concept 1.

    Returns:
    {
        "pairs": [{"reference": str, "target": str, "similarity": float}],
        "role_alignment": {role: {"reference": [...], "target": [...]}},
        "shared_domain": bool,
        "anchor_summary": str,
    }
    """
    phrases1 = _keywords_to_phrases(features1)
    phrases2 = _keywords_to_phrases(features2)

    title1 = features1.get("title", "Concept 1")
    title2 = features2.get("title", "Concept 2")
    domain1 = features1.get("domain", "general")
    domain2 = features2.get("domain", "general")

    # ── Direct similarity pairing ─────────────────────────────────────────
    pairs = []
    used_targets = set()

    for ref_phrase in phrases2:
        best_score = 0.0
        best_target = None
        for tgt_phrase in phrases1:
            if tgt_phrase in used_targets:
                continue
            score = _sim(ref_phrase, tgt_phrase)
            if score > best_score:
                best_score = score
                best_target = tgt_phrase
        if best_target and best_score > 0.05:
            pairs.append({
                "reference": ref_phrase,
                "target": best_target,
                "similarity": round(best_score, 3),
            })
            used_targets.add(best_target)

    # Sort by similarity descending, keep top 8
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    pairs = pairs[:8]

    # ── Role-based alignment ───────────────────────────────────────────────
    roles1 = _assign_roles(phrases1)
    roles2 = _assign_roles(phrases2)

    role_alignment = {}
    for role in _ROLE_PATTERNS:
        r1 = roles1.get(role, [])
        r2 = roles2.get(role, [])
        if r1 or r2:
            role_alignment[role] = {"target": r1[:3], "reference": r2[:3]}

    # ── Anchor summary ────────────────────────────────────────────────────
    top_pairs_text = "; ".join(
        f"{p['reference']} ↔ {p['target']}" for p in pairs[:4]
    )
    anchor_summary = (
        f"Using {title2} as a lens to explain {title1}. "
        f"Key analogical anchors: {top_pairs_text}."
    ) if top_pairs_text else f"Structural analogy between {title2} and {title1}."

    return {
        "pairs": pairs,
        "role_alignment": role_alignment,
        "shared_domain": domain1 == domain2,
        "anchor_summary": anchor_summary,
        "domain1": domain1,
        "domain2": domain2,
    }
