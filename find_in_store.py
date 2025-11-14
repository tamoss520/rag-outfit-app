# find_in_store.py
import streamlit as st
import pandas as pd

# Static hours + reserve instructions (demo)
STORE_HOURS_TEXT = """\
**Store Hours:**
Monday - Friday: 10 AM - 9 PM  
Saturday: 10 AM - 10 PM  
Sunday: 11 AM - 7 PM
"""

RESERVE_TEXT = """\
**Reserve & Pickup:**  
Call **1-800-RETAIL** or use our mobile app to reserve this item for in-store pickup within **2 hours**.
"""

# Mock stores (demo)
MOCK_STORES = [
    {"store_name": "Downtown Store",   "city": "London", "phone": "020-1111-2222"},
    {"store_name": "Fashion District", "city": "London", "phone": "020-3333-4444"},
    {"store_name": "Westfield Mall",   "city": "London", "phone": "020-5555-6666"},
]

def _stock_level_label(on_hand: int) -> str:
    if on_hand >= 8:
        return "In Stock"
    if on_hand >= 3:
        return "Limited"
    if on_hand >= 1:
        return "Low Stock"
    return "Out of Stock"

def _deterministic_on_hand(product_id: int, idx: int) -> int:
    """Deterministic pseudo-random stock per store for demo."""
    base = (product_id * (idx + 7)) % 11  # 0..10
    return base if base != 0 else 2

def _build_inventory_table(product_id: int) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(MOCK_STORES):
        on_hand = _deterministic_on_hand(product_id, i)
        rows.append({
            "Store": s["store_name"],
            "City": s["city"],
            "Status": _stock_level_label(on_hand),
            "On Hand": on_hand,
            "Phone": s["phone"],
        })
    df = pd.DataFrame(rows)
    rank = {"In Stock": 0, "Limited": 1, "Low Stock": 2, "Out of Stock": 3}
    df["__rank"] = df["Status"].map(rank)
    df = df.sort_values(["__rank", "On Hand"], ascending=[True, False]).drop(columns=["__rank"])
    return df

def _render_store_block(product_name: str, product_id: int):
    st.markdown(f"### Find **{product_name}** in Store")
    inv_df = _build_inventory_table(product_id)
    st.dataframe(inv_df, width="stretch", hide_index=True)
    st.markdown("---")
    st.markdown(STORE_HOURS_TEXT)
    st.markdown(RESERVE_TEXT)

def open_find_in_store(row: pd.Series) -> None:
    """
    Opens a 'find in store' UI for the given product.
    If Streamlit provides st.modal, use it; otherwise set session_state so the
    main app can render a bottom panel via render_store_panel().
    """
    product_name = str(row.get("productDisplayName", "(unknown)"))
    product_id = int(row.get("id", -1))

    # Newer Streamlit builds have st.modal; older ones don't.
    if hasattr(st, "modal"):
        with st.modal(f"Find {product_name} in Store"):
            _render_store_block(product_name, product_id)
            st.button("Close")
    else:
        # Fallback: push into session_state; main app will render a sticky panel.
        st.session_state["_store_panel_row"] = {
            "productDisplayName": product_name,
            "id": product_id,
        }
        # Force a rerun so the panel appears immediately.
        st.session_state["_store_panel_open"] = True

def render_store_panel() -> None:
    """
    Call once near the bottom of app.py.
    Renders a non-modal panel when st.modal is unavailable and a product was requested.
    """
    if getattr(st.session_state, "_store_panel_open", False):
        data = st.session_state.get("_store_panel_row")
        if not data:
            return
        with st.container(border=True):
            _render_store_block(data["productDisplayName"], int(data["id"]))
            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button("Close Store Panel"):
                    st.session_state["_store_panel_open"] = False
                    st.session_state.pop("_store_panel_row", None)
                    st.experimental_rerun()
