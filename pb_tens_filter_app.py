import streamlit as st
import pandas as pd
import ast
from typing import List

st.set_page_config(page_title="PB Tens Filter App", layout="wide")

# --- Digit settings for Tens ---
DIGITS = "0123456"

# --- CSV loaders ---
def load_filters(uploaded_file, is_zone=False):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    return df

# --- Parse override digits ---
def parse_list(txt: str) -> List[int]:
    if not txt:
        return []
    parts = txt.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
    return [int(x.strip()) for x in parts if x.strip().isdigit() and 0 <= int(x.strip()) <= 9]

# --- Hot/Cold/Due calculation ---
def auto_hot_cold_due(seed: str, prevs: List[str]):
    history = "".join(prevs)
    digits = list(DIGITS)
    counts = {d: history.count(d) for d in digits}
    sorted_digits = sorted(counts, key=counts.get, reverse=True)
    hot = sorted_digits[:3]
    cold = sorted_digits[-3:]
    due_candidates = set(digits) - set("".join(prevs[:2]))
    due = list(due_candidates)
    return [int(x) for x in hot], [int(x) for x in cold], [int(x) for x in due]

# --- Sidebar Inputs ---
st.title("PB Tens Filter App (mirrored from Ones app)")

st.subheader("Upload Filters")
manual_file = st.file_uploader("Upload Tens Filters CSV", type=["csv"])
zone_file = st.file_uploader("Upload Zone Filters CSV (optional)", type=["csv"])

seed = st.text_input("Seed (previous winning number)")
prevs = st.text_area("Prev draws (comma separated)").strip()
prevs_list = [p.strip() for p in prevs.split(",") if p.strip()]

hot_override = st.text_input("Hot digits override (optional)")
cold_override = st.text_input("Cold digits override (optional)")
due_override = st.text_input("Due digits override (optional)")

preserve_tracked = st.checkbox("Preserve tracked combos", value=False)
inject_tracked = st.checkbox("Inject tracked combos even if not generated", value=False)
select_all_toggle = st.checkbox("Select/Deselect all filters (shown)", value=False)
hide_zero = st.checkbox("Hide filters with 0 initial cuts", value=True)

if seed and not all(ch.isdigit() for ch in seed):
    st.warning("Seed must be numeric.")

# --- Load CSV ---
manual_filters = load_filters(manual_file, is_zone=False)
zone_filters = load_filters(zone_file, is_zone=True)

# --- Auto hot/cold/due if no override ---
auto_hot, auto_cold, auto_due = auto_hot_cold_due(seed, prevs_list)
hot = parse_list(hot_override) or auto_hot
cold = parse_list(cold_override) or auto_cold
due = parse_list(due_override) or auto_due

st.sidebar.markdown(f"**Auto →** Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}")
st.sidebar.markdown(f"**Using →** Hot {hot} | Cold {cold} | Due {due}")

# --- Show Manual Filters ---
st.subheader("🛠️ Manual Filters")

if manual_filters.empty:
    st.info("Upload a CSV to see filters.")
else:
    if hide_zero and "stat" in manual_filters.columns:
        manual_filters = manual_filters[~manual_filters["stat"].astype(str).str.startswith("0/")]
    if select_all_toggle:
        selected = st.multiselect("Select filters to apply", manual_filters["filter_id"].tolist(), manual_filters["filter_id"].tolist())
    else:
        selected = st.multiselect("Select filters to apply", manual_filters["filter_id"].tolist())

    st.dataframe(manual_filters)

    if selected:
        st.success(f"{len(selected)} filters selected.")

st.write("This PB Tens Filter App is fully mirrored from your Ones app — now using digits 0–6.")
