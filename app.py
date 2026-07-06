import streamlit as st
import time
from utils.wikipedia_fetcher import fetch_concept_data
from utils.nlp_analyzer import analyze_concept
from utils.concept_mapper import map_concepts
from utils.llm_generator import generate_explanation

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concept Explainer Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0d0d0f;
    --surface: #161618;
    --border: #2a2a2e;
    --accent: #e8c97e;
    --accent2: #7eb8e8;
    --text: #e8e6e0;
    --muted: #888882;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}

.concept-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    margin: 0.5rem 0;
}

.concept-card.target { border-left: 3px solid var(--accent); }
.concept-card.reference { border-left: 3px solid var(--accent2); }

.concept-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    margin: 0 0 0.5rem 0;
}
.concept-card.target h4 { color: var(--accent); }
.concept-card.reference h4 { color: var(--accent2); }

.concept-card p {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0;
}

.tag-pill {
    display: inline-block;
    background: #222226;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: var(--muted);
    margin: 2px;
}

.mapping-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #1e1e22;
    font-size: 0.88rem;
}

.mapping-row:last-child { border-bottom: none; }

.map-ref {
    flex: 1;
    color: var(--accent2);
    text-align: right;
}

.map-arrow {
    color: var(--muted);
    font-size: 0.9rem;
    flex-shrink: 0;
}

.map-target {
    flex: 1;
    color: var(--accent);
}

.output-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem;
    margin: 0.8rem 0;
}

.output-section h4 {
    font-family: 'Playfair Display', serif;
    color: var(--accent);
    margin: 0 0 0.7rem 0;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.output-section p, .output-section li {
    color: var(--text);
    line-height: 1.75;
    font-size: 0.93rem;
}

.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1c1c20;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0.2rem 0;
}

.step-badge.done { color: #7ec880; border-color: #2a3e2b; }
.step-badge.active { color: var(--accent); border-color: #3e3520; }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #d4a832) !important;
    color: #0d0d0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

label, .stSelectbox label { color: var(--muted) !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Concept Explainer Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Understand anything — by mapping it to what you already know.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Input Section ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    concept1 = st.text_input("🎯 Concept 1 — Target (what to explain)", placeholder="e.g. Quantum Entanglement")

with col2:
    concept2 = st.text_input("🔍 Concept 2 — Reference (analogy source)", placeholder="e.g. Telephone Network")

col3, col4, col5 = st.columns([1, 1, 1])
with col3:
    style = st.selectbox("Explanation Style", ["intuitive", "analytical", "educational"])
with col4:
    length = st.selectbox("Output Length", ["brief", "detailed", "comprehensive"])
with col5:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("✦ Generate Explanation", use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Main Logic ─────────────────────────────────────────────────────────────
if run_btn:
    if not concept1.strip() or not concept2.strip():
        st.warning("Please enter both concepts to proceed.")
        st.stop()

    if not st.session_state.get("openrouter_key"):
        st.error("⚠️ OpenRouter API key not set. Add it in the sidebar under Settings.")
        st.stop()

    progress_placeholder = st.empty()
    results_placeholder = st.empty()

    steps = [
        ("📡", "Fetching Wikipedia data…", False),
        ("🔬", "Running NLP analysis…", False),
        ("🗺️", "Mapping concepts…", False),
        ("🤖", "Generating explanation…", False),
    ]

    def render_steps(current_idx):
        html = ""
        for i, (icon, label, _) in enumerate(steps):
            if i < current_idx:
                css = "done"
                icon_shown = "✓"
            elif i == current_idx:
                css = "active"
                icon_shown = icon
            else:
                css = ""
                icon_shown = icon
            html += f'<span class="step-badge {css}">{icon_shown} {label}</span>&nbsp;'
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    # Step 1: Wikipedia
    render_steps(0)
    data1 = fetch_concept_data(concept1)
    data2 = fetch_concept_data(concept2)

    if data1.get("error") or data2.get("error"):
        err = data1.get("error") or data2.get("error")
        st.error(f"Wikipedia error: {err}")
        st.stop()

    # Step 2: NLP
    render_steps(1)
    features1 = analyze_concept(data1)
    features2 = analyze_concept(data2)

    # Step 3: Mapping
    render_steps(2)
    mapping = map_concepts(features1, features2)

    # Step 4: LLM
    render_steps(3)
    explanation = generate_explanation(
        concept1, data1, features1,
        concept2, data2, features2,
        mapping, style, length,
        api_key=st.session_state["openrouter_key"]
    )

    progress_placeholder.empty()

    # ─── Results ──────────────────────────────────────────────────────────
    st.markdown("## Results")
    rc1, rc2 = st.columns(2)

    with rc1:
        kw1 = " ".join([f'<span class="tag-pill">{k}</span>' for k in features1.get("keywords", [])[:8]])
        st.markdown(f"""
        <div class="concept-card target">
            <h4>🎯 {concept1}</h4>
            <p>{data1.get('summary','')[:280]}…</p>
            <div style="margin-top:0.6rem">{kw1}</div>
        </div>""", unsafe_allow_html=True)

    with rc2:
        kw2 = " ".join([f'<span class="tag-pill">{k}</span>' for k in features2.get("keywords", [])[:8]])
        st.markdown(f"""
        <div class="concept-card reference">
            <h4>🔍 {concept2}</h4>
            <p>{data2.get('summary','')[:280]}…</p>
            <div style="margin-top:0.6rem">{kw2}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Mapping visual
    st.markdown("### Concept Mapping")
    if mapping.get("pairs"):
        map_html = '<div class="concept-card" style="border-left:3px solid #444;">'
        map_html += f'<div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;"><span style="color:var(--accent2);font-size:0.78rem;font-weight:500;">{concept2.upper()}</span><span style="color:var(--accent);font-size:0.78rem;font-weight:500;">{concept1.upper()}</span></div>'
        for pair in mapping["pairs"][:7]:
            map_html += f'<div class="mapping-row"><span class="map-ref">{pair["reference"]}</span><span class="map-arrow">→</span><span class="map-target">{pair["target"]}</span></div>'
        map_html += "</div>"
        st.markdown(map_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # LLM Output sections
    st.markdown("### Explanation")
    sections = {
        "Core Explanation": explanation.get("core_explanation", ""),
        "Element Mapping": explanation.get("element_mapping", ""),
        "Key Insight": explanation.get("key_insight", ""),
        "Limits of the Analogy": explanation.get("limits", ""),
    }

    for title, body in sections.items():
        if body:
            st.markdown(f"""
            <div class="output-section">
                <h4>{title}</h4>
                <p>{body}</p>
            </div>""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    key_input = st.text_input("OpenRouter API Key", type="password",
                               value=st.session_state.get("openrouter_key", ""),
                               placeholder="sk-or-…")
    if key_input:
        st.session_state["openrouter_key"] = key_input

    st.markdown("---")
    st.markdown("""
    **How it works**
    1. Enter a target concept to explain
    2. Enter a familiar reference concept
    3. The engine maps their structures
    4. An LLM explains the target using the reference as analogy
    
    **Rule**: Concepts are never blended — the reference is *only* a lens.
    """)
