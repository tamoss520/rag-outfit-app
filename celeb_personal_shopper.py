# celeb_personal_shopper.py
import os
import json
import streamlit as st
from io import BytesIO
import requests
from celebrity_image_search import serpapi_search_images
from find_in_store import open_find_in_store


def _init_state():
    """Ensure all module state keys exist in st.session_state."""
    st.session_state.setdefault("celeb_outfits", None)          # list[dict] | None
    st.session_state.setdefault("celeb_query", "")              # str
    st.session_state.setdefault("celeb_selected_idx", 0)        # int
    st.session_state.setdefault("celeb_matches_cache", {})      # dict[(query, idx) -> list[(i,score)]]


def _generate_outfits(client, query: str):
    """Call LLM to produce exactly two outfit candidates for the given celebrity + occasion query."""
    sys_prompt = (
        "You are a celebrity fashion expert with extensive knowledge of celebrity styles. "
        "Return ONLY valid JSON (no markdown, no code fences) with exactly two outfit options, "
        "each containing 'title', 'description', and 'tags' (list of descriptive keywords). "
        "Schema: {\"outfits\":[{\"title\":\"\",\"description\":\"\",\"tags\":[\"\",...]},{...}]}"
    )
    user_prompt = (
        f"Create two distinct outfits for: {query}. "
        "Include styling details, color palette, and vibe keywords."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=800,
    )
    txt = resp.choices[0].message.content.strip()
    cleaned = txt.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    outfits = data.get("outfits", data)  # tolerate top-level list
    if not isinstance(outfits, list) or len(outfits) < 2:
        raise ValueError("Model did not return two outfits.")
    return outfits

def _render_celeb_cards(df, images_dir, add_to_outfit, render_ai_styling_tips_panel, occasion_context=None, per_row: int = 3):
    """Render small product cards (price + buttons) for the celeb shopper results."""
    if df is None or df.empty:
        st.info("No similar items found for this look.")
        return

    for start in range(0, len(df), per_row):
        chunk = df.iloc[start:start + per_row]
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                # image
                img_path = ""
                try:
                    candidate = os.path.join(images_dir, f"{int(row['id'])}.jpg")
                    if os.path.exists(candidate):
                        img_path = candidate
                except Exception:
                    pass
                if img_path:
                    st.image(img_path, width=180)

                # product info
                st.markdown(f"**{row.get('productDisplayName','(unknown)')}**")
                st.caption(
                    f"{row.get('articleType','')} • {row.get('baseColour','')} • {row.get('gender','')}"
                )

                # price (mock/demo)
                price_val = float(row.get("price", 0))
                st.markdown(f"**${price_val:.2f}**")

                # action buttons
                c1, c2, c3 = st.columns(3)
                # AI Styling Tips panel (reuses the same helper as text/image cards)
                with c1:
                    render_ai_styling_tips_panel(
                        row,
                        source_tag="celeb",
                        occasion_context=occasion_context or "",
                    )

                # Find in Store
                with c2:
                    if st.button("Find in Store", key=f"store-celeb-{row['id']}"):
                        st.session_state["_store_panel_source"] = "celeb"
                        open_find_in_store(row)

                # Add to Outfit
                with c3:
                    # Use a composite key (section + id) so keys don't clash
                    if st.button("Add to Outfit", key=f"add-celeb-{row['id']}"):
                        add_to_outfit(row)

def celeb_personal_shopper(
    styles_df,
    find_matching_items_with_rag,
    client,
    images_dir: str,
    add_to_outfit,
    render_results_grid,
):
    """
    Main UI entrypoint. Renders a Celeb Personal Shopper section, persists generated looks
    and selection in session_state, and shows top catalog matches (with images).

    `render_results_grid` is passed in from app.py so celeb results use the same
    card layout and panels as text/image results.
    """
    _init_state()

    st.markdown("### ⭐ Celeb Personal Shopper")
    st.markdown(
        "*Share a celebrity whose style you like, and the occasion. AI will study their looks, and give you two options, which you can make your own*"
    )
    query = st.text_input(
        "Celebrity + occasion",
        value=st.session_state.get("celeb_query", ""),
        placeholder="e.g., Taylor Swift on a date night",
    )

    left, right = st.columns([1, 3])
    with left:
        gen = st.button("Generate celebrity looks", use_container_width=True)

    # Generate new looks only when the button is pressed
    if gen and query:
        with st.spinner("Finding celebrity looks..."):
            try:
                outfits = _generate_outfits(client, query)
                st.session_state["celeb_outfits"] = outfits
                st.session_state["celeb_query"] = query
                st.session_state["celeb_selected_idx"] = 0
                st.session_state["celeb_matches_cache"] = {}  # clear cache for a new query
            except Exception as e:
                st.error(f"Could not generate looks: {e}")
                return

    outfits = st.session_state.get("celeb_outfits")
    if not outfits:
        st.info("Enter a celebrity + occasion and click **Generate celebrity looks**.")
        return

    # Choose which of the two looks to view
    titles = [o.get("title", f"Look {i+1}") for i, o in enumerate(outfits)]
    idx = st.radio(
        "Choose a look",
        options=list(range(len(outfits))),
        index=st.session_state.get("celeb_selected_idx", 0),
        format_func=lambda i: titles[i],
        horizontal=True,
        key="celeb_radio_key",
    )
    if idx != st.session_state.get("celeb_selected_idx", 0):
        st.session_state["celeb_selected_idx"] = idx

    chosen = outfits[st.session_state["celeb_selected_idx"]]
    title = chosen.get("title", "")
    desc = chosen.get("description", "")
    tags = chosen.get("tags", [])

    st.markdown(
        f'<div class="rn-callout-yellow">{desc}</div>',
        unsafe_allow_html=True,
    )
    if tags:
        st.caption("Tags: " + ", ".join(tags))

    # ---------- Celebrity reference images (robust, with diagnostics) ----------
    with st.expander("Show reference celebrity images"):
        def _fetch_bytes(url: str):
            """Fetch image bytes with a browser-like UA to avoid 403 hotlink blocks."""
            try:
                r = requests.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        )
                    },
                    timeout=10,
                )
                r.raise_for_status()
                return r.content, None
            except Exception as ex:
                return None, str(ex)

        key_present = bool(os.getenv("SERPAPI_KEY"))
        st.caption(f"SERPAPI key loaded: {'yes' if key_present else 'no'}")

        celeb_prompt = st.session_state.get("celeb_query", "")
        primary = f"{celeb_prompt} {title}".strip() if title else celeb_prompt
        fallback = f"{celeb_prompt} {desc[:120]}".strip()

        st.caption(f"Search 1 → “{primary}”")
        imgs, err = serpapi_search_images(primary, count=4)
        if err or not imgs:
            st.caption(f"(Primary returned no images; trying fallback)  Search 2 → “{fallback}”")
            imgs, err = serpapi_search_images(fallback, count=4)

        if err:
            st.error(f"Image search error: {err}")
        elif not imgs:
            st.info("No images found for this look.")
        else:
            st.caption(f"Results: {len(imgs)}")
            cols = st.columns(len(imgs))
            for col, im in zip(cols, imgs):
                with col:
                    url = im.get("thumbnailUrl") or im.get("contentUrl")
                    name = im.get("name") or "Reference"
                    src = im.get("source") or "Source"
                    page = im.get("hostPageUrl") or im.get("contentUrl")

                    shown = False
                    if url:
                        try:
                            st.image(url, width=180)
                            shown = True
                            st.caption("loaded: direct URL")
                        except Exception:
                            pass
                        if not shown:
                            data, e = _fetch_bytes(url)
                            if data:
                                st.image(BytesIO(data), width=180)
                                st.caption("loaded: fetched bytes")
                            else:
                                st.caption(f"could not load: {e or 'unknown error'}")

                    if page:
                        st.markdown(f"[{name}]({page})")
                    st.caption(src)

    # ---------- Build retrieval query and show catalog matches ----------
    text_query = desc or st.session_state.get("celeb_query", "")

    cache_key = (st.session_state.get("celeb_query", ""), st.session_state["celeb_selected_idx"])
    matches_cache = st.session_state.get("celeb_matches_cache", {})

    if cache_key in matches_cache:
        matches = matches_cache[cache_key]
    else:
        with st.spinner("Searching catalog for similar items…"):
            matches = find_matching_items_with_rag(styles_df, text_query)
        matches_cache[cache_key] = matches
        st.session_state["celeb_matches_cache"] = matches_cache

    if not matches:
        st.info("No similar items found for this look.")
        return

    st.markdown("**Top matches:**")
    top_idx = [i for i, _ in matches[:9]]
    top_df = styles_df.iloc[top_idx].copy()
    if "price" not in top_df.columns:
        base = (top_df["id"].astype(int) * 947) % 120 + 30
        top_df["price"] = (base.round(0) + 0.99).astype(float)

    # Use the shared grid renderer from app.py so celeb matches use the same:
    # - AI Style Tips + Find in Store + Add to Outfit buttons
    # - AI Styling Tips pane behavior
    # - Find in Store behavior
    render_results_grid(
        top_df,
        images_dir,
        per_row=3,
        source_tag="celeb",
    )


