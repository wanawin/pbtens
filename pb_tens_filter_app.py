import streamlit as st
import csv
import os
from collections import Counter
from itertools import product
import pandas as pd

# ==============================
# Globals
# ==============================
FILTER_CSV = "pb_tens_filters_adapted.csv"
PERCENTILE_CSV = "pb_tens_percentile_filters.csv"
DIGITS = "0123456"   # tens domain only

# ==============================
# Helpers
# ==============================
def safe_id(raw: str, fallback: str) -> str:
    return (raw or fallback).strip()

def load_filters(paths: list[str]) -> list[dict]:
    filters: list[dict] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for idx, raw in enumerate(rdr):
                row = { (k or "").lower().strip(): (v if isinstance(v, str) else v)
                        for k, v in raw.items() }

                fid = safe_id(row.get("id",""), f"row{idx+1}")
                layman = (row.get("layman") or row.get("layman_explanation") or "").strip()
                stat = (row.get("stat") or "").strip()
                expr = (row.get("expression") or "").strip()

                if not expr:
                    continue

                try:
                    code = compile(expr, f"<expr:{fid}>", "eval")
                except SyntaxError as e:
                    filters.append({
                        "id": fid, "layman": layman, "stat": stat,
                        "expr_code": None, "expr_text": expr,
                        "compile_error": str(e),
                        "is_percentile": (os.path.basename(path) == os.path.basename(PERCENTILE_CSV))
                    })
                    continue

                filters.append({
                    "id": fid, "layman": layman, "stat": stat,
                    "expr_code": code, "expr_text": expr,
                    "compile_error": None,
                    "is_percentile": (os.path.basename(path) == os.path.basename(PERCENTILE_CSV))
                })
    return filters

def generate_combos(seed: str, method: str):
    combos = []
    seen = set()
    if method == "1-digit":
        for d in seed:
            for p in product(DIGITS, repeat=4):
                key = "".join(sorted(d + "".join(p)))
                combos.append(key)
                seen.add(key)
    else:  # 2-digit pair
        pairs = { "".join(sorted((seed[i], seed[j])))
                  for i in range(len(seed)) for j in range(i+1, len(seed)) }
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                key = "".join(sorted(pair + "".join(p)))
                combos.append(key)
                seen.add(key)
    return combos, sorted(seen)

def compute_hot_cold_due(draws: list[str]):
    if len([d for d in draws if d]) < 6:
        return [], [], []

    last6 = [d for d in draws if d][:6]
    all_digits = "".join(last6)
    cnt = Counter(all_digits)

    hot = [int(x) for x, _ in cnt.most_common(3)]
    cold = [int(x) for x, _ in cnt.most_common()[-3:]]
    due = [d for d in range(7) if str(d) not in all_digits]

    return hot, cold, due

def run_filter_on_pool(pool, flt, ctx_base):
    survivors = []
    cut = 0
    for combo in pool:
        cdigits = [int(c) for c in combo]
        ctx = ctx_base | {"combo": combo, "cdigits": cdigits}
        try:
            if flt["expr_code"] and eval(flt["expr_code"], {}, ctx):
                cut += 1
            else:
                survivors.append(combo)
        except Exception:
            survivors.append(combo)
    return survivors, cut

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🎯 Powerball Tens Filter App")

    # --- Seed inputs
    seed = st.sidebar.text_input("Draw 1-back (required):", "").strip()
    prev2 = st.sidebar.text_input("Draw 2-back:", "").strip()
    prev3 = st.sidebar.text_input("Draw 3-back:", "").strip()
    prev4 = st.sidebar.text_input("Draw 4-back:", "").strip()
    prev5 = st.sidebar.text_input("Draw 5-back:", "").strip()
    prev6 = st.sidebar.text_input("Draw 6-back:", "").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

    # --- Hot/cold/due
    hot_override = st.sidebar.text_input("Hot digits (override, comma-separated):", "")
    cold_override = st.sidebar.text_input("Cold digits (override, comma-separated):", "")
    due_override = st.sidebar.text_input("Due digits (override, comma-separated):", "")

    # --- Track combo
    track_combo = st.sidebar.text_input("Track combo (e.g. 01234):", "").strip()

    # --- Filter toggles
    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)

    # --- Validate
    if len(seed) != 5 or any(ch not in DIGITS for ch in seed):
        st.warning("⚠️ Enter a valid 5-digit tens seed (digits 0–6).")
        return

    # --- Auto hot/cold/due
    draws = [seed, prev2, prev3, prev4, prev5, prev6]
    auto_hot, auto_cold, auto_due = compute_hot_cold_due(draws)

    hot = [int(x) for x in hot_override.split(",") if x.strip().isdigit()] or auto_hot
    cold = [int(x) for x in cold_override.split(",") if x.strip().isdigit()] or auto_cold
    due = [int(x) for x in due_override.split(",") if x.strip().isdigit()] or auto_due

    st.sidebar.markdown(f"**Auto Hot:** {auto_hot}, **Auto Cold:** {auto_cold}, **Auto Due:** {auto_due}")

    # --- Generate combos
    raw_combos, unique_combos = generate_combos(seed, method)

    # --- Load filters
    filters = load_filters([FILTER_CSV, PERCENTILE_CSV])

    percentile_filters = [f for f in filters if f["is_percentile"]]
    manual_filters = [f for f in filters if not f["is_percentile"]]

    ctx_base = {"seed": [int(c) for c in seed], "hot": hot, "cold": cold, "due": due}

    # --- Phase A: percentile filters pre-dedup
    pool = raw_combos
    for flt in percentile_filters:
        pool, _ = run_filter_on_pool(pool, flt, ctx_base)

    # --- Phase B: dedup
    pool = sorted(set(pool))

    # --- Metrics
    st.sidebar.markdown(f"""
**Pipeline**
- Raw generated: {len(raw_combos)}
- Survivors after percentile: {len(pool)}
- Unique enumeration: {len(unique_combos)}
""")

    # --- Manual filters
    st.header("🛠 Manual Filters")
    cut_counts = {}
    for flt in manual_filters:
        pool_test, cut = run_filter_on_pool(pool, flt, ctx_base)
        cut_counts[flt["id"]] = cut

    # Sort filters by aggressiveness
    sorted_filters = sorted(manual_filters, key=lambda f: -cut_counts[f["id"]])
    display_filters = [f for f in sorted_filters if cut_counts[f["id"]] > 0] if hide_zero else sorted_filters

    st.markdown(f"**Applicable filters: {len(display_filters)}**")

    survivors = pool.copy()
    eliminated_by = None

    for flt in display_filters:
        label = f"{flt['id']}: {flt['layman']} | hist {flt['stat']} | cut {cut_counts[flt['id']]}"
        checked = st.checkbox(label, value=select_all, key=f"flt_{flt['id']}")
        if checked:
            survivors, cut = run_filter_on_pool(survivors, flt, ctx_base)
            if track_combo and eliminated_by is None:
                if "".join(sorted(track_combo)) not in survivors:
                    eliminated_by = flt['id']

        st.markdown(f"Remaining: {len(survivors)}")

    # --- Tracked combo status
    if track_combo:
        norm = "".join(sorted(track_combo))
        if norm not in unique_combos:
            st.sidebar.error("Tracked combo was NOT generated.")
        elif eliminated_by:
            st.sidebar.error(f"Tracked combo eliminated by {eliminated_by}.")
        elif norm in survivors:
            st.sidebar.success("Tracked combo survived all filters.")
        else:
            st.sidebar.warning("Tracked combo eliminated.")

    # --- Survivors
    st.subheader(f"✅ Final Survivors: {len(survivors)}")
    with st.expander("Show survivors"):
        st.write(survivors)

    if survivors:
        df = pd.DataFrame(survivors, columns=["combo"])
        st.download_button("Download survivors (CSV)", df.to_csv(index=False), file_name="survivors.csv", mime="text/csv")
        st.download_button("Download survivors (TXT)", "\n".join(survivors), file_name="survivors.txt", mime="text/plain")

if __name__ == "__main__":
    main()
