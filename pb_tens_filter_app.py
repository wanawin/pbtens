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
    """Load filters from CSV (no caching to avoid unserializable code objects)."""
    if not os.path.exists(path):
        st.error(f"Filter file not found: {path}")
        st.stop()

    filters: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for idx, raw in enumerate(rdr):
            row = { (k or "").lower().strip(): (v if isinstance(v, str) else v)
                    for k, v in raw.items() }

            fid = safe_id(row.get("id",""), f"row{idx+1}")
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

def run_filters(combos: list[str], filters: list[dict], ctx_base: dict) -> tuple[list[str], dict]:
    """Evaluate filters on combos."""
    survivors = []
    cut_counts = {f["id"]: 0 for f in filters}

    for combo in combos:
        cdigits = [int(c) for c in combo]
        ctx = ctx_base | {"combo": combo, "cdigits": cdigits}
        eliminated = False

        for flt in filters:
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
    st.title("🎯 Powerball Tens Filter App")

    # ----------------
    # Inputs
    # ----------------
    seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "").strip()
    prev2 = st.sidebar.text_input("Draw 2-back (optional):", "").strip()
    prev3 = st.sidebar.text_input("Draw 3-back (optional):", "").strip()
    prev4 = st.sidebar.text_input("Draw 4-back (optional):", "").strip()
    prev5 = st.sidebar.text_input("Draw 5-back (optional):", "").strip()
    prev6 = st.sidebar.text_input("Draw 6-back (optional):", "").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

    hot_override = st.sidebar.text_input("Hot digits (comma-separated 0–6, overrides auto):", "")
    cold_override = st.sidebar.text_input("Cold digits (comma-separated 0–6, overrides auto):", "")
    due_override = st.sidebar.text_input("Due digits (comma-separated 0–6, overrides auto):", "")

    track_combos = st.sidebar.text_area("Track/Test combos (newline or comma-separated):", "")

    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos if not generated", value=False)

    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=True)
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

    st.sidebar.markdown(f"**Auto Hot:** {auto_hot}, **Auto Cold:** {auto_cold}, **Auto Due:** {auto_due}")

    # ----------------
    # Generate combos
    # ----------------
    combos = generate_combos(seed, method)

    # Add tracked combos if missing
    tracked = []
    if track_combos.strip():
        raw = track_combos.replace(",", " ").split()
        tracked = [c.strip() for c in raw if c.strip()]
        if inject_tracked:
            combos = sorted(set(combos) | set(tracked))

    # ----------------
    # Load filters
    # ----------------
    filters = load_filters(FILTER_CSV)

    # ----------------
    # Run filters
    # ----------------
    ctx_base = {
        "seed": [int(c) for c in seed],
        "hot": hot, "cold": cold, "due": due,
        "preserve_tracked": preserve_tracked,
        "tracked": tracked
    }

    survivors, cut_counts = run_filters(combos, filters, ctx_base)

    # ----------------
    # Sidebar filters
    # ----------------
    st.header("🛠 Manual Filters")

    for flt in filters:
        init_cuts = cut_counts.get(flt["id"], 0)
        if hide_zero and init_cuts == 0:
            continue
        label = f"{flt['id']}: {flt['layman']} | hist {flt['stat']} | cut {init_cuts}"
        key = f"flt_{flt['id']}"
        st.sidebar.checkbox(label, key=key, value=select_all)

    # ----------------
    # Survivors
    # ----------------
    st.subheader(f"Remaining after filters: {len(survivors)}")

    with st.expander("Show survivors"):
        st.write(survivors)

    # Download
    if survivors:
        df = pd.DataFrame(survivors, columns=["combo"])
        st.download_button("Download survivors (CSV)", df.to_csv(index=False), file_name="survivors.csv", mime="text/csv")
        st.download_button("Download survivors (TXT)", "\n".join(survivors), file_name="survivors.txt", mime="text/plain")


if __name__ == "__main__":
    main()
