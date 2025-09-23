import streamlit as st
import pandas as pd
import itertools
from pathlib import Path

# -----------------------------
# File constants
# -----------------------------
MANUAL_FILTER_CSV = "pb_tens_filters_adapted.csv"
PERCENTILE_FILTER_CSV = "pb_tens_percentile_filters.csv"

# -----------------------------
# Helpers
# -----------------------------
def load_filter_csv(path):
    if Path(path).exists():
        return pd.read_csv(path).to_dict("records")
    return []

def generate_tens(seed_draws, method="1-digit"):
    """Generate baseline combos from seed digits."""
    # For Powerball tens: only use digits 0–6
    base_digits = [str(i) for i in range(0, 7)]
    combos = list(itertools.product(base_digits, repeat=5))
    return ["".join(c) for c in combos]

def sum_category(combo):
    """Compute the tens sum (sum of digits)."""
    return sum(int(d) for d in combo)

def apply_percentile_filters(raw_combos, zone_filters):
    """Apply percentile filters to raw combos before deduplication."""
    survivors = []
    for combo in raw_combos:
        s = sum_category(combo)
        keep = any(low <= s <= high for (low, high) in zone_filters)
        if keep:
            survivors.append(combo)
    return survivors

def apply_manual_filters(combos, filters, seed_value, winner_value):
    """Apply manual filters one by one."""
    survivors = combos[:]
    results = []
    for flt in filters:
        expr = flt["expression"]
        fid = flt["filter_id"]
        try:
            # Safe eval context
            ctx = {
                "combo_digits": [int(d) for d in "".join(survivors[0])] if survivors else [],
                "seed_value": seed_value,
                "winner_value": winner_value,
                "winner_structure": 5,
            }
            cut = [c for c in survivors if eval(expr, {}, {**ctx, "winner": sum_category(c)})]
            survivors = [c for c in survivors if c not in cut]
            results.append((fid, len(cut), len(survivors)))
        except Exception as e:
            results.append((fid, f"error: {e}", len(survivors)))
    return survivors, results

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🎯 Powerball Tens Filter App")

    # Sidebar inputs
    seed1 = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "")
    seed2 = st.sidebar.text_input("Draw 2-back (optional):", "")
    seed3 = st.sidebar.text_input("Draw 3-back (optional):", "")
    seed4 = st.sidebar.text_input("Draw 4-back (optional):", "")
    seed5 = st.sidebar.text_input("Draw 5-back (optional):", "")
    seed6 = st.sidebar.text_input("Draw 6-back (optional):", "")

    gen_method = st.sidebar.selectbox("Generation Method:", ["1-digit"])

    hot_digits = st.sidebar.text_input("Hot digits (override, comma-separated):")
    cold_digits = st.sidebar.text_input("Cold digits (override, comma-separated):")
    due_digits = st.sidebar.text_input("Due digits (override, comma-separated):")

    tracked_combo = st.sidebar.text_input("Track/Test combo (e.g., 01234):")

    # -----------------------------
    # Load filters
    # -----------------------------
    manual_filters = load_filter_csv(MANUAL_FILTER_CSV)
    percentile_filters = load_filter_csv(PERCENTILE_FILTER_CSV)

    # Parse percentile zones from CSV
    zones = []
    for z in percentile_filters:
        try:
            parts = z["expression"].replace("sum", "").split("-")
            low, high = int(parts[0]), int(parts[1])
            zones.append((low, high))
        except:
            pass

    # -----------------------------
    # Generate combos
    # -----------------------------
    raw_combos = generate_tens([seed1, seed2, seed3, seed4, seed5, seed6], method=gen_method)
    st.sidebar.markdown(f"**Raw generated:** {len(raw_combos)}")

    # -----------------------------
    # Apply percentile filter (pre-dedup)
    # -----------------------------
    pre_dedup = apply_percentile_filters(raw_combos, zones) if zones else raw_combos
    st.sidebar.markdown(f"**Survive percentile pre-dedup:** {len(pre_dedup)}")

    # Deduplicate
    unique = sorted(set(pre_dedup))
    st.sidebar.markdown(f"**Unique enumeration:** {len(unique)}")

    survivors = unique[:]

    # -----------------------------
    # Apply manual filters
    # -----------------------------
    st.header("🛠️ Manual Filters")
    applicable_filters = 0

    for flt in manual_filters:
        fid = flt["filter_id"]
        layman = flt.get("layman_explanation", "")
        expr = flt["expression"]

        # Checkbox for each filter
        if st.checkbox(f"{fid}: {layman or expr}"):
            applicable_filters += 1
            survivors, _ = apply_manual_filters(survivors, [flt], 0, 0)
            st.markdown(f"Remaining: **{len(survivors)}**")

    st.sidebar.markdown(f"**Remaining after filters:** {len(survivors)}")

    # -----------------------------
    # Track combo status
    # -----------------------------
    if tracked_combo:
        if tracked_combo in survivors:
            st.sidebar.success("Tracked combo survived all filters.")
        else:
            st.sidebar.error("Tracked combo was eliminated.")

    # -----------------------------
    # Final survivors
    # -----------------------------
    st.subheader("✅ Final Survivors")
    st.markdown(f"**Final Survivors: {len(survivors)}**")

    if st.checkbox("Show survivors"):
        st.write(survivors)

    # Download options
    df = pd.DataFrame(survivors, columns=["combo"])
    st.download_button("Download survivors (CSV)", df.to_csv(index=False).encode("utf-8"), "survivors.csv", "text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(survivors), "survivors.txt", "text/plain")


if __name__ == "__main__":
    main()
