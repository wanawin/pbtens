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
DIGITS = "0123456"   # only 0–6 for tens model

# ==============================
# Helpers
# ==============================
def safe_id(raw: str, fallback: str) -> str:
    """Ensure every filter row has an ID."""
    return (raw or fallback).strip()

def load_filters(path: str) -> list[dict]:
    """Load filters from CSV."""
    if not os.path.exists(path):
        st.error(f"Filter file not found: {path}")
        st.stop()

    filters: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for idx, raw in enumerate(rdr):
            row = { (k or "").lower().strip(): (v if isinstance(v, str) else v)
                    for k, v in raw.items() }

            fid = safe_id(row.get("filter_id",""), f"row{idx+1}")
            layman = (row.get("layman") or row.get("layman_explanation") or "").strip()
            stat = (row.get("stat") or row.get("hist") or "").strip()
            expr = (row.get("expression") or row.get("expr") or "").strip()

            if not expr:
                continue

            try:
                code = compile(expr, f"<expr:{fid}>", "eval")
            except SyntaxError as e:
                filters.append({
                    "id": fid, "layman": layman, "stat": stat,
                    "expr_code": None, "expr_text": expr,
                    "compile_error": str(e)
                })
                continue

            filters.append({
                "id": fid, "layman": layman, "stat": stat,
                "expr_code": code, "expr_text": expr,
                "compile_error": None
            })
    return filters

def generate_combos(seed: str, method: str) -> list[str]:
    """Generate candidate tens combos."""
    combos = set()
    if method == "1-digit":
        for d in seed:
            for p in product(DIGITS, repeat=4):
                combos.add("".join(sorted(d + "".join(p))))
    else:  # 2-digit pair
        pairs = { "".join(sorted((seed[i], seed[j])))
                  for i in range(len(seed)) for j in range(i+1, len(seed)) }
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                combos.add("".join(sorted(pair + "".join(p))))
    return sorted(combos)

def compute_hot_cold_due(draws: list[str]) -> tuple[list[int], list[int], list[int]]:
    """Compute hot, cold, due tens digits from last 6 draws."""
    if len(draws) < 6:
        return [], [], []

    last6 = [d for d in draws[:6] if d]
    all_digits = "".join(last6)
    cnt = Counter(all_digits)

    hot = [int(x) for x, _ in cnt.most_common(3)]
    cold = [int(x) for x, _ in cnt.most_common()[-3:]]
    due = [d for d in range(7) if str(d) not in all_digits]

    return hot, cold, due

def run_filters(combos: list[str], filters: list[dict], ctx_base: dict, selected: list[str]) -> tuple[list[str], dict]:
    """Evaluate filters on combos."""
    survivors = []
    cut_counts = {f["id"]: 0 for f in filters}

    for combo in combos:
        cdigits = [int(c) for c in combo]
        ctx = ctx_base | {"combo": combo, "combo_digits": cdigits}
        eliminated = False

        for flt in filters:
            if flt["id"] not in selected:
                continue
            if not flt["expr_code"]:
                continue
            try:
                if eval(flt["expr_code"], {}, ctx):
                    cut_counts[flt["id"]] += 1
                    eliminated = True
                    break
            except Exception:
                continue
        if not eliminated:
            survivors.append(combo)

    return survivors, cut_counts

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🎯 Tens Filter App")

    # ----------------
    # Inputs
    # ----------------
    seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "").strip()
    prev2 = st.sidebar.text_input("Draw 2-back:", "").strip()
    prev3 = st.sidebar.text_input("Draw 3-back:", "").strip()
    prev4 = st.sidebar.text_input("Draw 4-back:", "").strip()
    prev5 = st.sidebar.text_input("Draw 5-back:", "").strip()
    prev6 = st.sidebar.text_input("Draw 6-back:", "").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

    hot_override = st.sidebar.text_input("Hot digits (override, comma-separated):", "")
    cold_override = st.sidebar.text_input("Cold digits (override, comma-separated):", "")
    due_override = st.sidebar.text_input("Due digits (override, comma-separated):", "")

    track_combo = st.sidebar.text_input("Track/Test combo (e.g., 00123):", "").strip()

    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos if not generated", value=False)

    select_all = st.sidebar.checkbox("Select/Deselect all filters (shown)", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=False)

    # ----------------
    # Validate seed
    # ----------------
    if len(seed) != 5 or any(ch not in DIGITS for ch in seed):
        st.warning("⚠️ Enter a valid 5-digit tens seed (digits 0–6).")
        return

    # ----------------
    # Hot/Cold/Due
    # ----------------
    draws = [seed, prev2, prev3, prev4, prev5, prev6]
    auto_hot, auto_cold, auto_due = compute_hot_cold_due(draws)

    hot = [int(x) for x in hot_override.split(",") if x.strip().isdigit()] or auto_hot
    cold = [int(x) for x in cold_override.split(",") if x.strip().isdigit()] or auto_cold
    due = [int(x) for x in due_override.split(",") if x.strip().isdigit()] or auto_due

    st.sidebar.markdown(f"**Auto ➜ Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}**")
    st.sidebar.markdown(f"**Using ➜ Hot {hot} | Cold {cold} | Due {due}**")

    # ----------------
    # Generate combos
    # ----------------
    combos = generate_combos(seed, method)

    if inject_tracked and track_combo:
        combos = sorted(set(combos) | {track_combo})

    # ----------------
    # Load filters
    # ----------------
    filters = load_filters(FILTER_CSV)

    # Determine which filters are selected
    selected_ids = []
    for flt in filters:
        key = f"flt_{flt['id']}"
        checked = st.sidebar.checkbox(
            f"{flt['id']}: {flt['layman']} | hist {flt['stat']}",
            key=key, value=select_all
        )
        if checked:
            selected_ids.append(flt["id"])

    # ----------------
    # Run filters
    # ----------------
    ctx_base = {
        "seed": [int(c) for c in seed],
        "hot": hot, "cold": cold, "due": due,
        "tracked": track_combo,
    }

    survivors, cut_counts = run_filters(combos, filters, ctx_base, selected_ids)

    # ----------------
    # Pipeline summary in sidebar
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
            eliminated_by = None
            for flt in filters:
                try:
                    if eval(flt["expr_code"], {}, ctx_base | {"combo": track_combo, "combo_digits": [int(c) for c in track_combo]}):
                        eliminated_by = flt["id"]
                        break
                except Exception:
                    continue
            if eliminated_by:
                st.sidebar.error(f"Tracked combo **eliminated by {eliminated_by}**.")
            else:
                st.sidebar.error("Tracked combo **eliminated**.")

    # ----------------
    # Main filter panel
    # ----------------
    st.header("🛠 Manual Filters")
    st.write(f"Applicable filters: {len(filters)}")

    # ----------------
    # Survivors
    # ----------------
    st.subheader(f"✅ Final Survivors: {len(survivors)}")

    with st.expander("Show survivors"):
        st.write(survivors)

    if survivors:
        df = pd.DataFrame(survivors, columns=["combo"])
        st.download_button("Download survivors (CSV)", df.to_csv(index=False), file_name="survivors.csv", mime="text/csv")
        st.download_button("Download survivors (TXT)", "\n".join(survivors), file_name="survivors.txt", mime="text/plain")


if __name__ == "__main__":
    main()
