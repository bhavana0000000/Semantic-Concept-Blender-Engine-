"""
LLM Generation Layer — OpenRouter API integration.

Sends a structured analogy-explanation prompt to a capable model and
returns a parsed dict with the four output sections.
"""
import json
import re
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"

LENGTH_TOKENS = {
    "brief": 600,
    "detailed": 1100,
    "comprehensive": 1800,
}

STYLE_GUIDANCE = {
    "intuitive": "Use vivid everyday language. Prioritize intuition over precision. Speak to a curious layperson.",
    "analytical": "Use precise, structured language. Highlight mechanisms and logical relationships.",
    "educational": "Use clear, step-by-step language suitable for a student encountering this for the first time.",
}


def _build_prompt(
    concept1: str, data1: dict, features1: dict,
    concept2: str, data2: dict, features2: dict,
    mapping: dict, style: str, length: str,
) -> str:
    pairs_text = "\n".join(
        f"  • {p['reference']} (from {concept2}) → {p['target']} (in {concept1})"
        for p in mapping.get("pairs", [])[:6]
    ) or "  (no direct mappings found — use broad structural analogy)"

    kw1 = ", ".join(features1.get("keywords", [])[:8])
    kw2 = ", ".join(features2.get("keywords", [])[:8])

    summary1 = data1.get("summary", "")[:400]
    summary2 = data2.get("summary", "")[:400]

    style_note = STYLE_GUIDANCE.get(style, STYLE_GUIDANCE["intuitive"])
    length_note = {
        "brief": "Keep the full response concise — roughly 200–300 words total.",
        "detailed": "Write a thorough explanation — roughly 400–600 words total.",
        "comprehensive": "Write a comprehensive, deep explanation — roughly 700–1000 words total.",
    }[length]

    return f"""You are an expert educator specializing in analogy-based explanations.

CONCEPT 1 — TARGET (what must be explained):
  Name: {concept1}
  Summary: {summary1}
  Key Features: {kw1}

CONCEPT 2 — REFERENCE (used ONLY as an explanatory lens):
  Name: {concept2}
  Summary: {summary2}
  Key Features: {kw2}

MAPPED RELATIONSHIPS (Concept 2 → Concept 1):
{pairs_text}

TASK:
Explain {concept1} using {concept2} as an analogy.

STRICT RULES — follow these exactly:
1. DO NOT combine, merge, or blend the two concepts.
2. DO NOT create a hybrid or new concept.
3. {concept1} is ALWAYS the subject being explained.
4. {concept2} is ONLY a tool/lens for explanation.
5. The analogy must illuminate {concept1} — not redefine it.

STYLE: {style_note}
LENGTH: {length_note}

OUTPUT — respond ONLY with a valid JSON object, no preamble, no markdown fences:
{{
  "core_explanation": "<Explain {concept1} using {concept2} as analogy. 2–4 paragraphs.>",
  "element_mapping": "<Walk through 3–5 specific element mappings from {concept2} to {concept1}. One sentence each.>",
  "key_insight": "<The single most powerful insight this analogy reveals about {concept1}. 1–2 sentences.>",
  "limits": "<Where does the analogy break down? What does {concept2} fail to capture about {concept1}? 2–3 sentences.>"
}}"""


def generate_explanation(
    concept1: str, data1: dict, features1: dict,
    concept2: str, data2: dict, features2: dict,
    mapping: dict, style: str, length: str,
    api_key: str,
) -> dict:
    """
    Call OpenRouter and return parsed output dict.
    Falls back to a structured error dict on failure.
    """
    prompt = _build_prompt(
        concept1, data1, features1,
        concept2, data2, features2,
        mapping, style, length,
    )

    max_tokens = LENGTH_TOKENS.get(length, 900)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://concept-explainer.app",
        "X-Title": "Concept Explainer Engine",
    }

    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json()

        content = raw["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        parsed = json.loads(content)

        # Validate required keys
        required = {"core_explanation", "element_mapping", "key_insight", "limits"}
        missing = required - set(parsed.keys())
        if missing:
            # Try to recover partial results
            for key in missing:
                parsed[key] = "(Not available)"

        return parsed

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        msg = e.response.text[:200] if e.response else str(e)
        return _error_response(f"OpenRouter API error {status}: {msg}")
    except json.JSONDecodeError as e:
        return _error_response(f"Failed to parse LLM response as JSON: {e}")
    except Exception as e:
        return _error_response(str(e))


def _error_response(msg: str) -> dict:
    return {
        "core_explanation": f"⚠️ Error generating explanation: {msg}",
        "element_mapping": "",
        "key_insight": "",
        "limits": "",
    }
