import streamlit as st
import pandas as pd
from collections import Counter
import itertools

# Constants
TENS_DOMAIN = set("0123456")

def digits_from_str(s):
    return [int(x) for x in s if x in TENS_DOMAIN] if s else []

def generate_combos(seed, method="1-digit"):
    # Simple example logic; adapt to your actual generator
    base = digits_from_str(seed)
    if method == "1-digit":
        return {"".join(map(str, combo)) for combo in itertools.product(range(7), repeat=5) if any(d in base for d in combo)}
    return set()

def apply_filters(combos, filters, hot_digits, cold_digits):
    survivors = []
    for c in combos:
        # Example integration: require at least 1 hot digit
        if not any(int(d) in hot_digits for d in c):
            continue
        # Example integration: require at least 1 cold digit
        if not any(int(d) in cold_digits for d in c):
            continue
        survivors.append(c)
    return survivors

# --- Streamlit UI ---
st.set_page_config(page_title="Powerball Tens-Only — Manual Filter Runner", layout="wide")

st.title("🎯 Powerball Tens-Only — Manual Filter Runner")

with st.sidebar:
    st.subheader("Run control")
    if st.button("Run / Refresh"):
        st.experimental_rerun()

    st.subheader("Inputs")
    seed = st.text_input("Seed tens (Draw 1-back, 5 digits 0–6):", "")
    prevs = []
    for i in range(2, 8):  # Prev up to 6 draws back
        prevs.append(st.text_input(f"Prev {i-1}-back tens (optional):", ""))

    method = st.selectbox("Generation Method:", ["1-digit", "other"])

    st.subheader("Filters")
    use_default = st.checkbox("Use default adapted filters", value=True)
    uploaded = st.file_uploader("Upload additional filter CSV (optional)", type="csv")

# --- Hot/Cold Calculation ---
all_prev_digits = []
for p in [seed] + prevs:
    if p:
        all_prev_digits.extend(digits_from_str(p))

counts = Counter(all_prev_digits)
freq = {d: counts.get(d, 0) for d in range(7)}
df_freq = pd.DataFrame({"digit": list(freq.keys()), "count": list(freq.values())})
df_freq = df_freq.sort_values("count", ascending=False).reset_index(drop=True)

st.subheader("🔥 Hot / ❄ Cold Digits from Seed + Previous Draws")
st.dataframe(df_freq, use_container_width=True)

top_hot = df_freq.head(3)["digit"].tolist()
bottom_cold = df_freq.tail(3)["digit"].tolist()
st.write(f"**Hot digits (auto):** {top_hot}")
st.write(f"**Cold digits (auto):** {bottom_cold}")

# --- Generate combos ---
if seed:
    combos = generate_combos(seed, method)
    st.write(f"Generated {len(combos)} combos.")

    # Apply filters with hot/cold integrated
    survivors = apply_filters(combos, None, top_hot, bottom_cold)

    st.subheader("✅ Survivors")
    st.write(f"Remaining after filters: {len(survivors)}")
    st.dataframe(pd.DataFrame({"combo": survivors}))

    st.download_button("Download survivors (CSV)", pd.DataFrame({"combo": survivors}).to_csv(index=False), "survivors.csv", "text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(survivors), "survivors.txt", "text/plain")
