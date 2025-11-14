# app.py (baseline)

import os, io, base64, json, ast
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage
from dotenv import load_dotenv
from find_in_store import open_find_in_store, render_store_panel, _render_store_block


# Load .env BEFORE importing OpenAI client or feature modules
load_dotenv()

# ---- Optional sanity: confirm image-search keys are visible to Streamlit ----
serp = os.getenv("SERPAPI_KEY")
bing = os.getenv("BING_IMAGE_SEARCH_KEY")
if not (serp or bing):
    st.warning("No image-search API key found (SERPAPI_KEY or BING_IMAGE_SEARCH_KEY). "
               "Celebrity reference images will be disabled.")
else:
    st.caption(f"Image search ready → SERPAPI: {'yes' if serp else 'no'} • BING: {'yes' if bing else 'no'}")

# ---- Optional sanity: confirm OpenAI key is present before creating the client ----
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it to your .env or export it before running Streamlit.")
    st.stop()

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Create one shared OpenAI client for the whole app
client = OpenAI()   # env var already loaded by load_dotenv()

# after this, import your feature module(s)
from celeb_personal_shopper import celeb_personal_shopper
from find_in_store import open_find_in_store, render_store_panel

# ----------------------------
# Environment & client
# ----------------------------
load_dotenv()
client = OpenAI()

GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"

DATA_CSV = "data/sample_clothes/sample_styles_with_embeddings.csv"
IMAGES_DIR = "data/sample_clothes/sample_images"  # optional preview

# ----------------------------
# Data loading (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_catalog():
    df = pd.read_csv(DATA_CSV, on_bad_lines="skip")
    if isinstance(df["embeddings"].iloc[0], str):
        df["embeddings"] = df["embeddings"].apply(ast.literal_eval)
    return df

styles_df = load_catalog()
unique_subcategories = styles_df["subCategory"].astype(str).unique().tolist()

# --- Ensure a price column exists (mock deterministic pricing for demo) ---
def ensure_price_column(df: pd.DataFrame) -> None:
    if "price" not in df.columns:
        # Deterministic "nice" prices based on id, e.g. 31.99–150.99
        base = (df["id"].astype(int) * 947) % 120 + 30
        df["price"] = (base.round(0) + 0.99).astype(float)

ensure_price_column(styles_df)

# ----------------------------
# Cookbook-style matching (Section 3)
# ----------------------------
def cosine_similarity_manual(vec1, vec2):
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-12)
    return float(np.dot(vec1, vec2) / denom)

def find_similar_items(input_embedding, embeddings, threshold=0.3, top_k=10):
    sims = [cosine_similarity_manual(input_embedding, emb) for emb in embeddings]
    order = np.argsort(sims)[::-1]
    scores_sorted = np.sort(sims)[::-1]
    return [(int(i), float(s)) for i, s in zip(order, scores_sorted) if s >= threshold][:top_k]

def find_matching_items_with_rag(df_items, item_descs, threshold=0.3, top_k=10):
    query_text = item_descs[0] if isinstance(item_descs, list) and item_descs else str(item_descs)
    emb_inp = client.embeddings.create(input=query_text, model=EMBEDDING_MODEL).data[0].embedding
    return find_similar_items(emb_inp, df_items["embeddings"].to_list(), threshold=threshold, top_k=top_k)

# ----------------------------
# Vision analysis (Section 4)
# ----------------------------
def encode_image_to_base64_from_bytes(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def analyze_image(encoded_image, unique_subcategories):
    system_prompt = (
        "You are a fashion analysis assistant. From the outfit image, extract:\n"
        "- subCategory (choose from the provided list when possible)\n"
        "- gender (Men/Women/Unisex if inferable)\n"
        "- baseColour (basic color family)\n"
        "- season (if inferable)\n"
        "Return STRICT JSON with keys: subCategory, gender, baseColour, season."
    )
    user_content = [
        {"type": "text",
         "text": "Analyze this outfit image and return JSON only. Known subcategories (subset): "
                 f"{sorted(set(map(str, unique_subcategories)))[:200]}"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
    ]
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_content}],
        temperature=0.2,
    )
    return resp.choices[0].message.content  # JSON or text containing JSON

def parse_json_lenient(text: str):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{"); end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                return {}
        return {}

# ----------------------------
# Guardrails (Section 6)
# ----------------------------
def format_item(row: pd.Series) -> str:
    parts = [f"name: {row.get('productDisplayName','(unknown)')}"]
    for c in ["articleType","subCategory","masterCategory","baseColour","gender","season","year","usage"]:
        if c in row and str(row[c]) != "nan":
            parts.append(f"{c}: {row[c]}")
    return " | ".join(parts)

# ----------------------------
# AI Styling Tips helpers
# ----------------------------

def _styling_key_base(row: pd.Series, source_tag: str) -> str:
    """Build a stable key prefix for styling tips for this row and section."""
    try:
        product_id = int(row.get("id", -1))
    except Exception:
        product_id = -1
    if product_id < 0:
        # Fallback to something deterministic if id is missing
        try:
            product_id = hash(str(row.to_dict()))
        except Exception:
            product_id = hash(str(row))
    return f"styling-{source_tag}-{product_id}"


def _generate_styling_tips_for_row(row: pd.Series, occasion_context: str = "") -> str:
    """
    Call the LLM to generate styling tips for a single product row.
    Returns markdown text in the required format.
    """
    name = str(row.get("productDisplayName", "this item"))

    details = []
    for col in ["articleType", "subCategory", "masterCategory", "baseColour", "gender", "season", "usage", "year"]:
        if col in row and str(row[col]) != "nan":
            details.append(f"{col}: {row[col]}")
    price_val = row.get("price", None)
    if price_val is not None:
        try:
            details.append(f"price: ${float(price_val):.2f}")
        except Exception:
            pass
    details_block = "\n".join(details) if details else "No additional catalog details."

    occ_line = occasion_context or "No specific occasion context provided."

    user_prompt = f"""
You are RetailNext's AI Stylist. Given one fashion item from a retail catalog, suggest concise,
practical styling ideas.

Write your answer in **plain markdown** in exactly this structure (no extra headings):

AI Styling Tips for [{name}]
How to Style
- bullet 1
- bullet 2
- bullet 3

Perfect Pairings
- bullet 1
- bullet 2
- bullet 3

Occasion Tips
- bullet 1
- bullet 2
- bullet 3

Rules:
- Keep each bullet short, concrete, and easy to understand.
- Refer to the item as "this blouse", "this dress", etc. instead of repeating the full name.
- Use realistic combinations a high-street retailer would sell.
- If the occasion context suggests casual, do not propose very formal looks, and vice versa.

Item details:
{details_block}

Occasion context:
{occ_line}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert fashion stylist for a large high-street retailer. "
                        "You respond with concise markdown styling advice only."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fail gracefully but preserve the structure
        return (
            f"AI Styling Tips for [{name}]\n"
            "How to Style\n"
            "- Sorry, I couldn't generate tips for this item right now.\n\n"
            "Perfect Pairings\n"
            "- Please try again in a moment.\n\n"
            "Occasion Tips\n"
            f"- Internal error: {e}"
        )


def render_ai_styling_tips_panel(row: pd.Series, source_tag: str, occasion_context: str = "") -> None:
    """
    Render ONLY the AI Styling Tips pane (expander) for a single product card.

    The toggle button is handled in render_product_card so this expander can be
    placed full-width under the card rather than inside a narrow column.
    """
    base_key = _styling_key_base(row, source_tag)
    open_key = f"{base_key}-open"
    content_key = f"{base_key}-content"

    if open_key not in st.session_state:
        st.session_state[open_key] = False
    if content_key not in st.session_state:
        st.session_state[content_key] = None

    # If the pane is not open, nothing to render
    if not st.session_state[open_key]:
        return

    # Full-width panel (called from card scope, not from a button column)
    with st.expander("AI Styling Tips", expanded=True):
        if st.session_state[content_key] is None:
            with st.spinner("Creating styling ideas for this item..."):
                st.session_state[content_key] = _generate_styling_tips_for_row(
                    row, occasion_context=occasion_context
                )

        st.markdown(st.session_state[content_key])

        # Extra button inside the panel (no-op for now)
        st.button(
            "Find items similar to these tips",
            key=f"{base_key}-find-similar",
            use_container_width=True,
        )


# ----------------------------
# Inline Find in Store helpers
# ----------------------------

def _store_key_base(row: pd.Series, source_tag: str) -> str:
    """Build a stable key prefix for inline Find in Store state."""
    try:
        product_id = int(row.get("id", -1))
    except Exception:
        product_id = -1
    if product_id < 0:
        try:
            product_id = hash(str(row.to_dict()))
        except Exception:
            product_id = hash(str(row))
    return f"store-inline-{source_tag}-{product_id}"


def render_inline_store_panel(row: pd.Series, source_tag: str) -> None:
    """
    Render the Find in Store pane inline (full width under the results row)
    for a single product, if its toggle is open.
    """
    base_key = _store_key_base(row, source_tag)
    open_key = f"{base_key}-open"

    # If this product's panel isn't open, nothing to draw
    if not st.session_state.get(open_key, False):
        return

    product_name = str(row.get("productDisplayName", "(unknown)"))
    try:
        product_id = int(row.get("id", -1))
    except Exception:
        product_id = -1

    with st.container(border=True):
        # Reuse the existing store rendering from find_in_store.py
        _render_store_block(product_name, product_id)

        cols = st.columns([1, 6])
        with cols[0]:
            if st.button("Close", key=f"{base_key}-close"):
                st.session_state[open_key] = False

# --- Product card renderer (price + three buttons) ---
def render_product_card(row: pd.Series, images_dir: str, source_tag: str) -> None:
    # image (if available)
    img_path = ""
    try:
        candidate = os.path.join(images_dir, f"{int(row['id'])}.jpg")
        if os.path.exists(candidate):
            img_path = candidate
    except Exception:
        pass

    # keys for styling tips open/close state
    styling_base = _styling_key_base(row, source_tag)
    styling_open_key = f"{styling_base}-open"
    if styling_open_key not in st.session_state:
        st.session_state[styling_open_key] = False

    # keys for inline Find in Store open/close state
    store_base = _store_key_base(row, source_tag)
    store_open_key = f"{store_base}-open"
    if store_open_key not in st.session_state:
        st.session_state[store_open_key] = False

    card = st.container()
    with card:
        # Image at the top of the card
        if img_path:
            st.image(img_path, width=180)

        # --- Fixed-height text block: title + meta + price ---
        name = row.get("productDisplayName", "(unknown)")
        meta = f"{row.get('articleType','')} • {row.get('baseColour','')} • {row.get('gender','')}"
        price_val = float(row.get("price", 0))

        card_text_html = f"""
        <div class="rn-card-text">
            <div class="rn-card-title">{name}</div>
            <div class="rn-card-meta">{meta}</div>
            <div class="rn-card-price">${price_val:.2f}</div>
        </div>
        """
        st.markdown(card_text_html, unsafe_allow_html=True)

        # -------- Top row: AI Style Tips + Find in Store --------
        top_left, top_right = st.columns(2)

        # AI Style Tips toggle (left)
        with top_left:
            if st.button(
                "AI Style Tips",
                key=f"{styling_base}-button",
                use_container_width=True,
            ):
                # toggle the styling pane open/closed for this product
                st.session_state[styling_open_key] = not st.session_state[styling_open_key]

        # Find in Store toggle (right)
        with top_right:
            if st.button(
                "Find in Store",
                key=f"{store_base}-button",
                use_container_width=True,
            ):
                # toggle the inline store pane open/closed for this product
                st.session_state[store_open_key] = not st.session_state[store_open_key]

        # -------- Bottom row: wide "+ Add to Outfit" --------
        if st.button(
            "+ Add to Outfit",
            key=f"add-{source_tag}-{row['id']}",
            use_container_width=True,
            type="primary",
        ):
            add_to_outfit(row)

# --- Grid renderer (N cards per row) ---
def render_results_grid(df: pd.DataFrame, images_dir: str, per_row: int = 3, source_tag: str = "text") -> None:
    if df is None or df.empty:
        st.info("No results to display.")
        return

    # Occasion context for styling tips (same across this section)
    try:
        occ_context = occasion if (occasion and occasion != "(none)") else ""
    except NameError:
        occ_context = ""

    # render cards in rows of `per_row`
    for start in range(0, len(df), per_row):
        chunk = df.iloc[start:start + per_row]

        # 1) Row of product cards (3 columns)
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                render_product_card(row, images_dir, source_tag)

        # 2) Row-wide panels for any items in this chunk whose toggles are open
        for _, row in chunk.iterrows():
            # AI Styling Tips pane (full width)
            render_ai_styling_tips_panel(
                row,
                source_tag=source_tag,
                occasion_context=occ_context,
            )

            # Inline Find in Store pane (full width)
            render_inline_store_panel(
                row,
                source_tag=source_tag,
            )

def check_match(reference_desc: str, candidate_desc: str, occasion_text: str = "") -> bool:
    """
    YES/NO guardrail with optional occasion context.
    Say YES unless the candidate is clearly inappropriate for the stated occasion or mismatched to the reference.
    """
    sys_prompt = "You are a strict fashion validator. Answer only 'YES' or 'NO'."
    occ = f"\nContext: The outfit should be appropriate for {occasion_text}." if occasion_text else ""
    user_prompt = (
        "Reference item description:\n"
        f"{reference_desc}\n\n"
        "Candidate item description:\n"
        f"{candidate_desc}\n\n"
        "Answer only YES or NO. "
        "Say YES unless the candidate is clearly inappropriate for the stated occasion or mismatched to the reference."
        f"{occ}"
    )

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")

# ----------------------------
# UI (baseline)
# ----------------------------
st.set_page_config(page_title="RetailNext AI Outfit Assistant", page_icon="🛍️", layout="wide")

# ---- RetailNext theming & hero header ----
def _load_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

_bg_base64 = _load_base64("RetailNext background.png")
_logo_base64 = _load_base64("RetailNext logo.png")

bg_style = ""
if _bg_base64:
    bg_style = (
        "background-image: url('data:image/png;base64,"
        + _bg_base64
        + "'); background-size: cover; background-position: center;"
    )

st.markdown(
    f"""
<style>

/* ============================
   RetailNext Overall Page Style
   ============================ */
.stApp {{
    background-color: #e1f4fb;   /* light turquoise */
}}

/* ============================
   Hero Section
   ============================ */
.rn-hero {{
    background-color: #ffffff;
    border-radius: 18px;
    padding: 1.75rem 2.25rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 1.25rem;
    {bg_style}
    background-repeat: no-repeat;
    background-size: cover;
    background-position: top center;
    min-height: 300px;
}}

.rn-hero-inner {{
    display: flex;
    flex-direction: column;
    gap: 1rem;
}}

.rn-hero-logo-row {{
    max-width: 220px;
    margin-bottom: 0.75rem;
}}

.rn-logo-img {{
    width: 100%;
    height: auto;
}}

.rn-hero-title {{
    font-size: 2.4rem;
    font-weight: 700;
    color: #111111;
    margin-bottom: 0.25rem;
}}

.rn-hero-sub {{
    font-weight: 600;
    font-style: italic;
    color: #009fc4;  /* RetailNext turquoise */
    margin-bottom: 0.75rem;
}}

.rn-hero-explainer {{
    font-size: 0.95rem;
    color: #222222;
    line-height: 1.5;
}}
.rn-hero-explainer b {{
    color: #111111;
}}

/* ============================
   Universal Text Colour Fixes
   ============================ */
body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
.stTextInput label, .stSelectbox label, .stSlider label,
.stCheckbox label, .stCaption {{
    color: #111 !important;
}}

/* Section main titles */
h3 {{
    color: #009fc4 !important;
    font-weight: 700 !important;
}}

/* Section subtitles */
h4 {{
    color: #009fc4 !important;
    font-weight: 700 !important;
}}

/* --- Celeb 'Choose a look' radio group --- */
/* Dark teal background behind the options */
[data-testid="stRadio"] > div {{
    background-color: #003b5c;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
}}

/* Make the look names high-contrast white on dark teal */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] div[role="radiogroup"] label,
[data-testid="stRadio"] div[role="radiogroup"] span {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

/* File uploader label ("Upload outfit image (JPG/PNG)") */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label span {{
    color: #009fc4 !important;
    font-weight: 600 !important;
}}

/* Yellow callout boxes (replacing green success/info) */
.rn-callout-yellow {{
    background-color: #ffcc33;
    color: #000000;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.9rem;
    margin-top: 0.25rem;
}}

/* ============================
   Button Styling
   ============================ */

/* Primary (Add to Outfit) */
.stButton > button[kind="primary"] {{
    background-color: #009fc4 !important;
    color: white !important;
    border-radius: 6px;
    font-weight: 600;
    border: none;
}}

/* Secondary (AI Style Tips, Find In Store, Generate looks, etc.) */
.stButton > button[kind="secondary"] {{
    background-color: #ffcc33 !important;
    color: black !important;
    border: 1px solid black !important;
    border-radius: 6px;
    font-weight: 600;
}}

/* ============================
   Product Card Styling
   ============================ */
.rn-product-card {{
    background-color: white;
    border-radius: 18px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}}

.rn-card-text {{
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}

.rn-card-title {{
    font-weight: 700;
    margin-bottom: 0.25rem;
}}

.rn-card-meta {{
    font-size: 0.85rem;
    color: #333333;
    margin-bottom: 0.5rem;
}}

.rn-card-price {{
    font-weight: 700;
    margin-top: auto;
}}

</style>
""",
    unsafe_allow_html=True,
)

hero_html = f"""
<div class="rn-hero">
  <div class="rn-hero-inner">
    <div class="rn-hero-logo-row">
      <img src="data:image/png;base64,{_logo_base64}" alt="RetailNext" class="rn-logo-img" />
    </div>
    <div class="rn-hero-title">RetailNext AI Outfit Assistant</div>
    <div class="rn-hero-sub"><strong><em>Build your perfect outfit</em></strong></div>
    <div class="rn-hero-explainer">
      <p><b>Know exactly what item you need?</b> Describe the item and have AI find the most similar items available, plus complimentary items and styling tips.</p>
      <p><b>Got part of an outfit already but not sure what goes with it?</b> Upload a photo of your own clothing and get AI-powered outfit matching recommendations, complimentary items, and styling tips.</p>
      <p><b>Struggling for outfit inspiration?</b> Tell us a celebrity whose style you like and have AI study their looks, finding similar and complimentary items for you to make your own.</p>
    </div>
  </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

with st.sidebar:
    st.header("Search")

    top_k = st.slider(
        "Top-K results",
        3, 12, 8,
        key="sidebar_top_k",
    )
    sim_threshold = st.slider(
        "Similarity threshold",
        0.1, 0.6, 0.3, 0.05,
        key="sidebar_sim_threshold",
    )
    enable_guardrails = st.checkbox(
        "Enable Guardrails (YES/NO validator)",
        value=True,
        key="sidebar_enable_guardrails",
    )
    reference_choice = st.selectbox(
        "Reference Choice (guardrails)",
        ["top_candidate", "analysis"],
        key="sidebar_reference_choice",
    )

    st.markdown("---")
    st.caption(f"Catalog size: **{len(styles_df)}** items")

# Default active_mode
if "_active_mode" not in st.session_state:
    st.session_state["_active_mode"] = "Text"

# --- Input & Results columns ---
cols = st.columns([1, 1.2])

with cols[0]:
    st.subheader("Your Ideas")
    st.markdown("*Describe what you need or upload a photo of your own clothing to start with*")

    # --- TEXT INPUT (always visible) ---
    default_text = st.session_state.get("text_query_text", "")
    raw_text = st.text_input(
        "Describe what you need",
        value=default_text,
        key="text_query_box",
        placeholder="white cotton shirt for men",
    )

    # --- Occasion dropdown (moved from sidebar) ---
    occasion = st.selectbox(
        "Occasion",
        [
            "(none)",
            "Black tie",
            "Job interview",
            "Smart-casual dinner",
            "Outdoor wedding",
            "Office formal",
            "Festival",
            "Back-to-school",
        ],
        index=0,
        key="occasion_select_main",
    )

    # Process text query if provided
    if raw_text:
        query_text = raw_text
        if occasion and occasion != "(none)":
            query_text = f"{raw_text} for {occasion.lower()}"
        st.session_state["text_query_text"] = query_text
        st.session_state["_active_mode"] = "Text"
        st.markdown(
            f'<div class="rn-callout-yellow">Current query: {query_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- IMAGE INPUT (always visible) ---
    uploaded = st.file_uploader(
        "Upload outfit image (JPG/PNG)",
        type=["jpg","jpeg","png"],
        key="image_upload"
    )

    if uploaded:
        img = PILImage.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

        if st.button("Analyze image"):
            with st.spinner("Analyzing image…"):
                # Only call the vision model and quietly parse the JSON;
                # do not show raw JSON or parsed dict
                raw = analyze_image(encoded, unique_subcategories)
                analysis = parse_json_lenient(raw)

                st.session_state["image_analysis"] = analysis

                parts = [
                    str(analysis.get(k, ""))
                    for k in ["subCategory", "baseColour", "season", "gender"]
                    if analysis.get(k)
                ]
                query_text = " ".join(parts) if parts else "versatile casual outfit"
                if occasion and occasion != "(none)":
                    query_text += f" for {occasion.lower()}"

                st.session_state["image_query_text"] = query_text
                st.session_state["_active_mode"] = "Image"

                st.markdown(
                    f'<div class="rn-callout-yellow">Query from analysis: {query_text}</div>',
                    unsafe_allow_html=True,
                )

    # Show prior image query if exists and we’re in image mode
    if st.session_state.get("image_query_text") and st.session_state["_active_mode"] == "Image":
        st.markdown(
            f'<div class="rn-callout-yellow">Current image query: {st.session_state["image_query_text"]}</div>',
            unsafe_allow_html=True,
        )

# ----------------------------
# Occasion-based prefilter (optional)
# ----------------------------
def apply_occasion_prefilter(df: pd.DataFrame, occasion: str) -> pd.DataFrame:
    """
    Lightly biases the shortlist based on occasion keywords.
    Example: 'Job interview' → prefer shirts and formal items.
    """
    if not occasion or occasion == "(none)":
        return df
    o = occasion.lower()

    df2 = df.copy()
    # Keep it simple for demo dataset
    if "job interview" in o or "office" in o:
        mask = (
            df2["articleType"].astype(str).str.lower().str.contains("shirt")
            | df2["usage"].astype(str).str.lower().str.contains("formal")
        )
        df2 = df2[mask]

    if df2.empty:
        return df  # fallback to original shortlist if filter removes all
    return df2

# ----------------------------
# Outfit helpers (My Outfit basket)
# ----------------------------
def add_to_outfit(row: pd.Series) -> None:
    """
    Add the given product row to the My Outfit basket in session_state.
    Uses the product's 'id' as the key; avoids duplicates.
    """
    product_id = int(row.get("id", -1))
    if product_id < 0:
        return
    outfit_ids = st.session_state.get("outfit_ids", [])
    if product_id not in outfit_ids:
        outfit_ids.append(product_id)
        st.session_state["outfit_ids"] = outfit_ids

def _get_outfit_df(styles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of the current outfit items, based on outfit_ids in session_state.
    """
    outfit_ids = st.session_state.get("outfit_ids", [])
    if not outfit_ids:
        return pd.DataFrame(columns=styles_df.columns)
    df = styles_df[styles_df["id"].isin(outfit_ids)].copy()
    # Drop duplicates if the same id got added more than once somehow
    df = df.drop_duplicates(subset=["id"])
    return df

def render_outfit_panel(styles_df: pd.DataFrame, images_dir: str) -> None:
    """
    Renders the 'My Outfit' pane at the bottom of the page with:
    - each selected item (image, name, price)
    - total price
    - a 'Buy Online Now' button (no-op for now)
    """
    outfit_df = _get_outfit_df(styles_df)
    if outfit_df.empty:
        return  # nothing to show yet

    st.markdown("## 👗 My Outfit")

    # Show each item in the outfit
    for _, row in outfit_df.iterrows():
        col_img, col_info = st.columns([1, 3])
        with col_img:
            img_path = ""
            try:
                candidate = os.path.join(images_dir, f"{int(row['id'])}.jpg")
                if os.path.exists(candidate):
                    img_path = candidate
            except Exception:
                pass
            if img_path:
                st.image(img_path, width=120)
        with col_info:
            st.markdown(f"**{row.get('productDisplayName','(unknown)')}**")
            st.caption(f"{row.get('articleType','')} • {row.get('baseColour','')} • {row.get('gender','')}")
            price_val = float(row.get("price", 0))
            st.markdown(f"**${price_val:.2f}**")

    # Total price
    total_price = float(outfit_df["price"].sum())
    st.markdown("---")
    st.markdown(f"### Total: **${total_price:.2f}**")

    # Action buttons (non-functional for now)
    cols = st.columns([2, 2, 3])
    with cols[0]:
        st.button("Buy Online Now")
    with cols[1]:
        if st.button("Clear Outfit"):
            st.session_state["outfit_ids"] = []
            st.experimental_rerun()

with cols[1]:
    st.subheader("Possible Outfit Items")
    st.markdown("*Check out the products available that are similar or complimentary to your search*")

    # Determine current query from active mode
    active_mode = st.session_state.get("_active_mode", "Text")
    if active_mode == "Text":
        current_query = st.session_state.get("text_query_text")
    else:
        current_query = st.session_state.get("image_query_text")

    # Initialize containers for search results
    matches = []
    shortlist = None
    final_df = None

    if current_query:
        # Retrieve matches using your existing matcher
        with st.spinner("Retrieving similar items…"):
            matches = find_matching_items_with_rag(styles_df, current_query)

        if matches:
            # Build shortlist DataFrame from matches
            idxs = [i for i, _ in matches]
            scores = [s for _, s in matches]
            shortlist = styles_df.iloc[idxs].copy()
            shortlist.insert(0, "score", scores)

            # Apply occasion prefilter
            shortlist = apply_occasion_prefilter(shortlist, occasion)

            st.markdown("#### Broad Options (pre-guardrails):")
            st.dataframe(shortlist, width="stretch", hide_index=True)

            # ----- Guardrails -----
            final_df = shortlist.copy()
            if enable_guardrails:
                image_analysis = st.session_state.get("image_analysis", {})
                if (
                    active_mode == "Image"
                    and isinstance(image_analysis, dict)
                    and image_analysis
                    and (occasion and occasion != "(none)")
                ):
                    ref_desc = " | ".join(
                        [
                            f"{k}: {image_analysis[k]}"
                            for k in ["subCategory", "baseColour", "season", "gender"]
                            if image_analysis.get(k)
                        ]
                    )
                elif (
                    active_mode == "Image"
                    and reference_choice == "analysis"
                    and isinstance(image_analysis, dict)
                    and image_analysis
                ):
                    ref_desc = " | ".join(
                        [
                            f"{k}: {image_analysis[k]}"
                            for k in ["subCategory", "baseColour", "season", "gender"]
                            if image_analysis.get(k)
                        ]
                    )
                else:
                    ref_idx, _ = matches[0]
                    ref_desc = format_item(styles_df.iloc[ref_idx])

                occ_text = occasion if (occasion and occasion != "(none)") else ""

                kept = []
                for i, score in matches:
                    cand_desc = format_item(styles_df.iloc[i])
                    if check_match(ref_desc, cand_desc, occasion_text=occ_text):
                        kept.append((i, score))

                if kept:
                    kept_idxs, kept_scores = zip(*kept)
                    final_df = styles_df.iloc[list(kept_idxs)].copy()
                    final_df.insert(0, "score", kept_scores)
                    final_df = final_df.reset_index(drop=True)
                else:
                    final_df = pd.DataFrame(columns=shortlist.columns)

            # ----- Final results (cards) -----
            st.markdown("#### Best Options (guardrailed):")
            current_source_tag = "text" if active_mode == "Text" else "image"
            render_results_grid(final_df, IMAGES_DIR, per_row=3, source_tag=current_source_tag)

            # Inline Find-in-Store panel for THIS section
            if (
                st.session_state.get("_store_panel_open")
                and st.session_state.get("_store_panel_source") == current_source_tag
            ):
                render_store_panel()

        else:
            st.info("No matches found for this query.")
    # NOTE: if no current_query, show nothing extra (no info box)

# -------------------------------------------------------
st.markdown("---")
celeb_personal_shopper(
    styles_df,
    find_matching_items_with_rag,
    client,
    IMAGES_DIR,
    add_to_outfit,
    render_results_grid,  # 👈 use the shared grid for celeb matches
)

if st.session_state.get("_store_panel_open") and st.session_state.get("_store_panel_source") == "celeb":
    render_store_panel()


# Finally, show the My Outfit basket (if any items were added)
render_outfit_panel(styles_df, IMAGES_DIR)





