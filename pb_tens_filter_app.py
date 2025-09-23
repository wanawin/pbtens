import streamlit as st
import pandas as pd
import ast

# ----------------
# Config
# ----------------
FILTER_CSV = "pb_tens_filters_adapted.csv"

st.set_page_config(page_title="Powerball Tens Filter App", layout="wide")

# ----------------
# Helpers
# ----------------
def load_filters(csv_path):
    df = pd.read_csv(csv_path)
    filters = []
    for _, row in df.iterrows():
        filters.append({
            "id": row["filter_id"],
            "variant": row["variant"],
            "expression": row["expression"],
            "layman": row.get("layman_explanation", ""),
            "stat": row.get("hist", ""),
        })
    return filters


def run_filters(combos, filters, ctx_base, selected_ids):
    survivors = combos[:]
    cut_counts = {f["id"]: 0 for f in filters}

    for flt in filters:
        if flt["id"] not in selected_ids:
            continue
        expr = flt["expression"]
        survivors_tmp = []
        for c in survivors:
            ctx = ctx_base.copy()
            ctx["winner"] = c
            ctx["combo_digits"] = [int(x) for x in str(c)]
            try:
                if eval(expr, {}, ctx):
                    cut_counts[flt["id"]] += 1
                else:
                    survivors_tmp.append(c)
            except Exception as e:
                survivors_tmp.append(c)
        survivors = survivors_tmp
    return survivors, cut_counts


def generate_combos(seed, method="1-digit"):
    # stub: just digits 0–6, length 5
    import itertools
    return ["".join(p) for p in itertools.product("0123456", repeat=5)]


# ----------------
# Sidebar controls
# ----------------
st.sidebar.header("Inputs")

seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "11344")
prev2 = st.sidebar.text_input("Draw 2-back (optional):", "")
prev3 = st.sidebar.text_input("Draw 3-back (optional):", "")
prev4 = st.sidebar.text_input("Draw 4-back (optional):", "")
prev5 = st.sidebar.text_input("Draw 5-back (optional):", "")
prev6 = st.sidebar.text_input("Draw 6-back (optional):", "")

method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit"])
hot = [int(x) for x in st.sidebar.text_input("Hot digits (override, comma-separated):", "").split(",") if x.strip().isdigit()]
cold = [int(x) for x in st.sidebar.text_input("Cold digits (override, comma-separated):", "").split(",") if x.strip().isdigit()]
due = [int(x) for x in st.sidebar.text_input("Due digits (override, comma-separated):", "").split(",") if x.strip().isdigit()]

track_combo = st.sidebar.text_input("Track/Test combo (e.g., 00123):", "").strip()

# ----------------
# Generate combos
# ----------------
combos = generate_combos(seed, method)

# ----------------
# Load filters
# ----------------
filters = load_filters(FILTER_CSV)

# ----------------
# Run with none selected
# ----------------
ctx_base = {
    "seed": [int(c) for c in seed],
    "hot": hot, "cold": cold, "due": due,
    "tracked": track_combo,
}

survivors, cut_counts = run_filters(combos, filters, ctx_base, [])

# ----------------
# Sidebar pipeline summary
# ----------------
st.sidebar.markdown("### Pipeline")
st.sidebar.write(f"Raw generated: {len(combos)}")
st.sidebar.write(f"Unique enumeration: {len(set(combos))}")
st.sidebar.write(f"Remaining after filters: {len(survivors)}")

if track_combo:
    if track_combo not in combos:
        st.sidebar.warning("Tracked combo was **NOT generated**.")
    elif track_combo in survivors:
        st.sidebar.success("Tracked combo **survived all filters**.")
    else:
        st.sidebar.error("Tracked combo **eliminated**.")

# ----------------
# Main filter panel
# ----------------
st.header("🛠 Manual Filters")
st.write(f"Applicable filters: {len(filters)}")

selected_ids = []
for flt in filters:
    key = f"flt_{flt['id']}"
    checked = st.checkbox(
        f"{flt['id']}: {flt['layman']} | hist {flt['stat']}",
        key=key, value=False
    )
    if checked:
        selected_ids.append(flt["id"])

# ----------------
# Re-run with selected filters
# ----------------
survivors, cut_counts = run_filters(combos, filters, ctx_base, selected_ids)

st.subheader(f"✅ Final Survivors: {len(survivors)}")
with st.expander("Show survivors"):
    st.write(survivors)

if survivors:
    df = pd.DataFrame(survivors, columns=["combo"])
    st.download_button("Download survivors (CSV)", df.to_csv(index=False), file_name="survivors.csv", mime="text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(survivors), file_name="survivors.txt", mime="text/plain")
