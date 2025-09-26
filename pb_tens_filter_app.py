import os
import csv
from collections import Counter
from itertools import product
import pandas as pd
import streamlit as st

MANUAL_FILTER_CSV = "pb_tens_filters_adapted.csv"
ZONE_FILTER_CSV_CANDIDATES = [
    "pb_tens_percentile_filters_updated.csv",
    "pb_tens_percentile_filters.csv",
]

DIGITS = "0123456"  # tens-only
LOW_SET = {0, 1, 2, 3, 4}
HIGH_SET = {5, 6}

def _exists(path: str) -> bool: return path and os.path.exists(path)
def _first_existing(paths): return next((p for p in paths if _exists(p)), None)
def _safe_id(raw: str, fallback: str) -> str: return (raw or fallback).strip()

def _compile_row(row, fid):
    expr_txt = (row.get("expression") or row.get("expr") or "").strip()
    if not expr_txt: return None, "empty expression"
    try:
        return compile(expr_txt, f"<expr:{fid}>", "eval"), None
    except SyntaxError as e:
        return None, str(e)

def _normalize_cols(raw: dict) -> dict:
    return {(k or "").strip().lower(): v for k, v in raw.items()}

def load_filters(path: str, is_zone: bool = False) -> list[dict]:
    if not _exists(path): return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for i, raw in enumerate(rdr):
            row = _normalize_cols(raw)
            fid = _safe_id(row.get("id", row.get("filter_id", "")), f"row{i+1}")
            layman = (row.get("layman") or row.get("layman_explanation") or "").strip()
            hist = (row.get("stat") or row.get("hist") or "").strip()
            code, cerr = _compile_row(row, fid)
            out.append(dict(id=fid, layman=f"[syntax error: {cerr}] {layman}" if cerr else layman,
                            hist=hist, code=code, is_zone=is_zone))
    return out

def sum_category(total: int) -> str:
    if 0 <= total <= 10: return "Very Low"
    elif 11 <= total <= 13: return "Low"
    elif 14 <= total <= 17: return "Mid"
    return "High"

def gen_raw_combos(seed: str, method: str) -> list[str]:
    seed_sorted = "".join(sorted(seed))
    raw = []
    if method == "1-digit":
        for d in seed_sorted:
            for p in product(DIGITS, repeat=4):
                raw.append("".join(sorted(d + "".join(p))))
    else:
        pairs = {"".join(sorted((seed_sorted[i], seed_sorted[j]))) for i in range(5) for j in range(i+1,5)}
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                raw.append("".join(sorted(pair + "".join(p))))
    return raw

def make_ctx(seed: str, prevs: list[str], combo: str, hot: list[int], cold: list[int], due: list[int]):
    combo_digits = [int(c) for c in combo] if combo else []
    seed_digits = [int(c) for c in seed]
    prev_digits = [int(c) for c in (prevs[0] or "")] if prevs else []
    prev_prev_digits = [int(c) for c in (prevs[1] or "")] if len(prevs) > 1 else []
    return {
        "variant_name": "tens",
        "combo_digits": combo_digits,
        "seed_digits": seed_digits,
        "prev_seed_digits": prev_digits,
        "prev_prev_seed_digits": prev_prev_digits,
        "winner_value": sum(combo_digits),
        "seed_value": sum(seed_digits),
        "winner_even_count": sum(1 for d in combo_digits if d%2==0),
        "winner_odd_count": sum(1 for d in combo_digits if d%2==1),
        "winner_unique_count": len(set(combo_digits)),
        "winner_range": max(combo_digits)-min(combo_digits) if combo_digits else 0,
        "winner_low_count": sum(1 for d in combo_digits if d in LOW_SET),
        "winner_high_count": sum(1 for d in combo_digits if d in HIGH_SET),
        "hot_digits": hot, "cold_digits": cold, "due_digits": due,
        "Counter": Counter, "sum_category": sum_category, "abs": abs, "len": len, "set": set,
    }

def auto_hot_cold_due(seed: str, prevs: list[str]) -> tuple[list[int], list[int], list[int]]:
    # hot/cold from up to 10 draws if available
    seq10 = "".join([seed] + [p for p in prevs if p])
    hot = cold = []
    if len(seq10) >= 50:
        cnt = Counter(int(ch) for ch in seq10)
        hot = [d for d,_ in cnt.most_common(3)]
        cold = [d for d,_ in cnt.most_common()[-3:]]
    # due always just 2 draws
    seq2 = "".join([s for s in [seed] + prevs[:1] if s])
    due = []
    if len(seq2) >= 10:
        cnt2 = Counter(int(ch) for ch in seq2)
        due = [d for d in range(7) if d not in cnt2]
    return hot, cold, due

# ------------------- UI -------------------
st.set_page_config(page_title="Powerball Tens Filter App", layout="wide")
st.sidebar.header("Inputs")

seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):").strip()
prevs = [st.sidebar.text_input(f"Draw {i}-back:", value="").strip() for i in range(2,11)]
method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

hot_override = st.sidebar.text_input("Hot digits override:", value="")
cold_override = st.sidebar.text_input("Cold digits override:", value="")
due_override = st.sidebar.text_input("Due digits override:", value="")

track = st.sidebar.text_input("Track/Test combo (e.g., 01234):").strip()
track_norm = "".join(sorted("".join(ch for ch in track if ch.isdigit()))) if track else None

if len(seed)!=5 or any(ch not in DIGITS for ch in seed):
    st.info("Enter a valid 5-digit seed first.")
    st.stop()

manual_filters = load_filters(MANUAL_FILTER_CSV, is_zone=False)
zone_filters = load_filters(_first_existing(ZONE_FILTER_CSV_CANDIDATES), is_zone=True)

auto_hot, auto_cold, auto_due = auto_hot_cold_due(seed, prevs)
parse_list = lambda txt: [int(t) for t in txt.replace(","," ").split() if t.isdigit() and 0<=int(t)<=6]
hot = parse_list(hot_override) or auto_hot
cold = parse_list(cold_override) or auto_cold
due = parse_list(due_override) or auto_due

st.sidebar.markdown(f"Auto ➜ Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}")
st.sidebar.markdown(f"Using ➜ Hot {hot} | Cold {cold} | Due {due}")

raw = gen_raw_combos(seed, method)
base_ctx = {"seed_str": seed, "prev_strs": prevs, "hot": hot, "cold": cold, "due": due}
raw_after_zones = [c for c in raw if all(not f["code"] or not eval(f["code"], {}, base_ctx | make_ctx(seed, prevs, c, hot, cold, due)) for f in zone_filters)]
unique_baseline = sorted(set(raw_after_zones))

st.sidebar.write(f"Generated: {len(raw)} | After zones: {len(raw_after_zones)} | Unique: {len(unique_baseline)}")

# Manual filter rendering
init_counts = {f["id"]: sum(1 for c in unique_baseline if f.get("code") and eval(f["code"], {}, base_ctx | make_ctx(seed, prevs, c, hot, cold, due))) for f in manual_filters if f.get("code")}
display_filters = sorted(manual_filters, key=lambda f: -init_counts.get(f["id"],0))
selection_state = {f["id"]: st.checkbox(f"{f['id']} | {f['layman']} | init cut {init_counts.get(f['id'],0)}") for f in display_filters}

# Apply sequential filters
pool = list(unique_baseline)
for f in display_filters:
    if selection_state.get(f["id"]):
        pool = [c for c in pool if not (f.get("code") and eval(f["code"], {}, base_ctx | make_ctx(seed, prevs, c, hot, cold, due)))]

st.write(f"### ✅ Survivors: {len(pool)}")
if pool:
    df = pd.DataFrame({"tens_combo": pool})
    st.download_button("Download CSV", df.to_csv(index=False), "pb_tens_survivors.csv")
    st.download_button("Download TXT", "\n".join(pool), "pb_tens_survivors.txt")
