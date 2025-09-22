import streamlit as st
import csv, os
from collections import Counter
from itertools import product
import pandas as pd

# ==============================
# Globals
# ==============================
MANUAL_FILTER_CSV = "pb_tens_filters_adapted.csv"
PERCENTILE_FILTER_CSV = "pb_tens_percentile_filters.csv"
DIGITS = "0123456"   # only 0–6 for tens model

# ==============================
# Helpers
# ==============================
def safe_id(raw: str, fallback: str) -> str:
    return (raw or fallback).strip()

def load_filter_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        st.error(f"Filter file not found: {path}")
        st.stop()

    filters = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for idx, raw in enumerate(rdr):
            row = { (k or "").lower().strip(): (v if isinstance(v, str) else v)
                    for k, v in raw.items() }

            fid = safe_id(row.get("id",""), f"row{idx+1}")
            layman = (row.get("layman") or "").strip()
            stat = (row.get("stat") or row.get("hist") or "").strip()
            expr = (row.get("expression") or row.get("expr") or "").strip()

            if not expr:
                continue

            try:
                code = compile(expr, f"<expr:{fid}>", "eval")
            except SyntaxError:
                filters.append({
                    "id": fid, "layman": layman, "stat": stat,
                    "expr_code": None, "expr_text": expr
                })
                continue

            filters.append({
                "id": fid, "layman": layman, "stat": stat,
                "expr_code": code, "expr_text": expr
            })
    return filters

def generate_tens(seed: str, method: str) -> list[str]:
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

def apply_percentile_filters(raw_combos: list[str], zone_filters: list[dict], base_ctx: dict) -> list[str]:
    survivors = []
    for combo in raw_combos:
        cdigits = [int(c) for c in combo]
        ctx = base_ctx | {"combo": combo, "cdigits": cdigits}
        keep = True
        for flt in zone_filters:
            if not flt["expr_code"]:
                continue
            try:
                if eval(flt["expr_code"], {}, ctx):
                    keep = False
                    break
            except Exception:
                continue
        if keep:
            survivors.append(combo)
    return survivors

# ==============================
# Streamlit App
# ==============================
def main():
    st.title("🎯 Tens — Manual Filter Runner")

    # ----------------
    # Inputs
    # ----------------
    seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "").strip()
    prev2 = st.sidebar.text_input("Draw 2-back (optional):", "").strip()
    prev3 = st.sidebar.text_input("Draw 3-back (optional):", "").strip()
    prev4 = st.sidebar.text_input("Draw 4-back (optional):", "").strip()
    prev5 = st.sidebar.text_input("Draw 5-back (optional):", "").strip()
    prev6 = st.sidebar.text_input("Draw 6-back (optional):", "").strip()

    method = st.sidebar.selectbox("Generation method:", ["1-digit", "2-digit pair"])

    hot_override = st.sidebar.text_input("Hot digits (comma-separated 0–6, overrides auto):", "")
    cold_override = st.sidebar.text_input("Cold digits (comma-separated 0–6, overrides auto):", "")
    due_override = st.sidebar.text_input("Due digits (comma-separated 0–6, overrides auto):", "")

    track_combos = st.sidebar.text_area("Track/Test combos (newline or comma-sep):", "")
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    select_all = st.sidebar.checkbox("Default to selected when shown", value=True)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=False)

    # ----------------
    # Validate seed
    # ----------------
    if len(seed) != 5 or any(ch not in DIGITS for ch in seed):
        st.warning("⚠️ Enter a valid Draw 1-back: exactly 5 digits in 0–6.")
        return

    # ----------------
    # Hot/Cold/Due
    # ----------------
    draws = [seed, prev2, prev3, prev4, prev5, prev6]
    auto_hot, auto_cold, auto_due = compute_hot_cold_due(draws)
    hot = [int(x) for x in hot_override.split(",") if x.strip().isdigit()] or auto_hot
    cold = [int(x) for x in cold_override.split(",") if x.strip().isdigit()] or auto_cold
    due = [int(x) for x in due_override.split(",") if x.strip().isdigit()] or auto_due
    st.sidebar.markdown(f"Auto ➜ Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}")
    st.sidebar.markdown(f"Using ➜ Hot {hot} | Cold {cold} | Due {due}")

    # ----------------
    # Generate combos
    # ----------------
    raw_combos = generate_tens(seed, method)

    # Tracked
    tracked = []
    if track_combos.strip():
        raw = track_combos.replace(",", " ").split()
        tracked = [c.strip() for c in raw if c.strip()]
        if inject_tracked:
            raw_combos = sorted(set(raw_combos) | set(tracked))

    # ----------------
    # Apply percentile filters (pre-dedup)
    # ----------------
    zone_filters = load_filter_csv(PERCENTILE_FILTER_CSV)
    ctx_base = {"seed": [int(c) for c in seed], "hot": hot, "cold": cold, "due": due}
    post_percentile = apply_percentile_filters(raw_combos, zone_filters, ctx_base)

    # Deduplication
    unique_baseline = sorted(set(post_percentile))

    # ----------------
    # Manual filters
    # ----------------
    manual_filters = load_filter_csv(MANUAL_FILTER_CSV)
    survivors, cut_counts = run_filters(unique_baseline, manual_filters, ctx_base)

    # ----------------
    # Manual filters UI
    # ----------------
    st.header("🛠 Manual Filters")
    rows = []
    for flt in manual_filters:
        init_cuts = cut_counts.get(flt["id"], 0)
        if hide_zero and init_cuts == 0:
            continue
        label = f"{flt['id']}: {flt['layman']} | hist {flt['stat']} | cut {init_cuts}"
        key = f"flt_{flt['id']}"
        active = st.checkbox(label, key=key, value=select_all)
        rows.append((flt, active))

    # Apply only selected manual filters sequentially
    live_pool = survivors
    for flt, active in rows:
        if active and flt["expr_code"]:
            new_pool = []
            cut = 0
            for combo in live_pool:
                ctx = ctx_base | {"combo": combo, "cdigits": [int(c) for c in combo]}
                try:
                    if eval(flt["expr_code"], {}, ctx):
                        cut += 1
                    else:
                        new_pool.append(combo)
                except Exception:
                    new_pool.append(combo)
            live_pool = new_pool
            st.markdown(f"Filter {flt['id']} applied → Remaining: {len(live_pool)} (cut {cut})")

    # ----------------
    # Survivors
    # ----------------
    st.subheader(f"Remaining after filters: {len(live_pool)}")
    with st.expander("Show survivors"):
        st.write(live_pool)

    if live_pool:
        df = pd.DataFrame(live_pool, columns=["combo"])
        st.download_button("Download survivors (CSV)", df.to_csv(index=False), file_name="survivors.csv", mime="text/csv")
        st.download_button("Download survivors (TXT)", "\n".join(live_pool), file_name="survivors.txt", mime="text/plain")

if __name__ == "__main__":
    main()
