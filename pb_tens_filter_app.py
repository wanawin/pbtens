import os
import csv
from collections import Counter
from itertools import product
import pandas as pd
import streamlit as st

# -----------------------------
# Files
# -----------------------------
MANUAL_FILTER_CSV_CANDIDATES = [
    "filters_tens_cleaned_FINAL_BALANCED_DEDUPED.csv",
    "filters_tens_cleaned_FINAL_BALANCED_PLAIN4.csv",
    "filters_tens_cleaned_FINAL_BALANCED.csv",
    "filters_tens_cleaned.csv",
]
ZONE_FILTER_CSV_CANDIDATES = [
    "pb_tens_percentile_filters.csv",
    "filters_tens_cleaned.csv",
]

# -----------------------------
# Constants
# -----------------------------
DIGITS = "0123456"  # tens-only model
LOW_SET  = {0, 1, 2, 3, 4}
HIGH_SET = {5, 6}   # <-- fix: only 5 & 6 are "high" in 0–6 world

# Mirror digit map (0↔5, 1↔6, 2↔7, 3↔8, 4↔9)
MIRROR_MAP = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}

# V-Trac groups (keep mapping identical to ones app)
VTRAC_MAP = {
    0: 0, 5: 0,
    1: 1, 6: 1,
    2: 2, 7: 2,
    3: 3, 8: 3,
    4: 4, 9: 4
}

# -----------------------------
# Utilities
# -----------------------------
def _exists(path: str) -> bool:
    return path and os.path.exists(path)

def _first_existing(paths):
    return next((p for p in paths if _exists(p)), None)

def _safe_id(raw: str, fallback: str) -> str:
    return (raw or fallback).strip()

def _compile_row(row, fid):
    expr_txt = (row.get("expression") or row.get("expr") or "").strip()
    if not expr_txt:
        return None, "empty expression"
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
            out.append(dict(
                id=fid,
                layman=f"[syntax error: {cerr}] {layman}" if cerr else layman,
                hist=hist,
                code=code,
                is_zone=is_zone
            ))
    return out

def sum_category(total: int) -> str:
    if 0 <= total <= 10: return "Very Low"
    elif 11 <= total <= 13: return "Low"
    elif 14 <= total <= 17: return "Mid"
    return "High"

def _valid_draw(s: str) -> bool:
    """5 chars, each in 0–6."""
    return isinstance(s, str) and len(s) == 5 and all(ch in DIGITS for ch in s)

def gen_raw_combos(seed: str, method: str) -> list[str]:
    seed_sorted = "".join(sorted(seed))
    raw = []
    if method == "1-digit":
        for d in seed_sorted:
            for p in product(DIGITS, repeat=4):
                raw.append("".join(sorted(d + "".join(p))))
    else:
        pairs = {"".join(sorted((seed_sorted[i], seed_sorted[j])))
                 for i in range(5) for j in range(i+1, 5)}
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                raw.append("".join(sorted(pair + "".join(p))))
    return raw

# ---------- Mirror helpers ----------
def mirror_digits(seq):
    """Return the digit-wise mirrors for a sequence of ints."""
    return [MIRROR_MAP[int(d)] for d in seq]

def mirror(x):
    """
    Flexible mirror helper for eval():
    - If x is a list/tuple of digits -> return mirrored list
    - If x is a single int/str digit -> return mirrored digit
    """
    if isinstance(x, (list, tuple)):
        return [MIRROR_MAP[int(d)] for d in x]
    try:
        return MIRROR_MAP[int(x)]
    except Exception:
        # Fallback: try to iterate
        return [MIRROR_MAP[int(d)] for d in x]

# ---------- Hot / Cold / Due (robust) ----------
def auto_hot_cold_due(seed: str, prevs: list[str], hot_n: int = 3, cold_n: int = 3,
                      hot_window: int = 10, due_window: int = 2):
    """
    Compute hot/cold from up to `hot_window` draws (seed + prevs).
    Compute due from the last `due_window` draws.
    Restrict to tens digits 0–6.
    """
    # Keep only valid 5-digit draws in 0–6
    recents = [s for s in [seed] + prevs if _valid_draw(s)]

    # HOT/COLD
    seq_hotcold = "".join(recents[:hot_window])
    hot, cold = [], []
    if seq_hotcold:
        cnt = Counter(int(ch) for ch in seq_hotcold if ch.isdigit() and 0 <= int(ch) <= 6)
        if cnt:
            freqs_desc = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
            cutoff = freqs_desc[min(hot_n - 1, len(freqs_desc) - 1)][1] if freqs_desc else 0
            hot = sorted([d for d, c in cnt.items() if c >= cutoff])

            freqs_asc = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]))
            cutoff_c = freqs_asc[min(cold_n - 1, len(freqs_asc) - 1)][1] if freqs_asc else 0
            cold = sorted([d for d, c in cnt.items() if c <= cutoff_c])

    # DUE: digits not seen in last two valid draws
    seq_due = "".join(recents[:due_window])
    due = []
    if seq_due:
        seen = {int(ch) for ch in seq_due if ch.isdigit() and 0 <= int(ch) <= 6}
        due = [d for d in range(7) if d not in seen]

    return hot, cold, due

def make_ctx(seed: str, prevs: list[str], combo: str, hot: list[int], cold: list[int], due: list[int]):
    combo_digits = [int(c) for c in combo] if combo else []
    seed_digits = [int(c) for c in seed]
    prev_digits = [int(c) for c in (prevs[0] or "")] if prevs else []
    prev_prev_digits = [int(c) for c in (prevs[1] or "")] if len(prevs) > 1 else []

    # Mirrors & V-Trac
    combo_mirrors = mirror_digits(combo_digits)
    seed_mirrors = mirror_digits(seed_digits)
    combo_vtracs = [VTRAC_MAP[d] for d in combo_digits]
    seed_vtracs = [VTRAC_MAP[d] for d in seed_digits]

    # Variant scalar values:
    winner_val = sum(combo_digits)
    seed_val = sum(seed_digits)

    return {
        "variant_name": "tens",

        # Core digit context
        "combo_digits": combo_digits,
        "seed_digits": seed_digits,
        "prev_seed_digits": prev_digits,
        "prev_prev_seed_digits": prev_prev_digits,
        "combo_digits_set": set(combo_digits),
        "seed_digits_set": set(seed_digits),

        # Scalars (aliases)
        "winner_value": winner_val,
        "seed_value": seed_val,
        "winner": winner_val,   # alias for expressions using 'winner'
        "seed": seed_val,       # alias for expressions using 'seed'

        # Precomputed stats
        "winner_even_count": sum(1 for d in combo_digits if d % 2 == 0),
        "winner_odd_count": sum(1 for d in combo_digits if d % 2 == 1),
        "winner_unique_count": len(set(combo_digits)),
        "winner_range": max(combo_digits) - min(combo_digits) if combo_digits else 0,
        "winner_low_count": sum(1 for d in combo_digits if d in LOW_SET),
        "winner_high_count": sum(1 for d in combo_digits if d in HIGH_SET),

        # Mirrors & V-Trac
        "combo_mirrors": combo_mirrors,
        "seed_mirrors": seed_mirrors,
        "combo_vtracs": combo_vtracs,
        "seed_vtracs": seed_vtracs,
        "mirror_map": MIRROR_MAP,
        "mirror": mirror,

        # Hot/Cold/Due + aliases + sets
        "hot_digits": hot, "cold_digits": cold, "due_digits": due,
        "hot": hot, "cold": cold, "due": due,
        "hot_set": set(hot), "cold_set": set(cold), "due_set": set(due),

        # Safe builtins for eval()
        "Counter": Counter,
        "sum_category": sum_category,
        "abs": abs, "len": len, "set": set,
        "sorted": sorted, "max": max, "min": min, "range": range,
        "any": any, "all": all,
    }

def apply_zone_filters(raw_combos, zone_filters, seed, prevs, hot, cold, due):
    if not zone_filters:
        return list(raw_combos)
    survivors = []
    killed = 0
    for c in raw_combos:
        ctx = make_ctx(seed, prevs, c, hot, cold, due)
        eliminate = False
        for f in zone_filters:
            if f.get("code"):
                try:
                    if eval(f["code"], {}, ctx):
                        eliminate = True
                        break
                except Exception:
                    # ignore broken zone filters
                    pass
        if eliminate:
            killed += 1
        else:
            survivors.append(c)
    # Safety: don't wipe the pool silently
    if not survivors and raw_combos:
        st.warning("Zone filters eliminated all candidates. Skipping zone filters for this run.")
        return list(raw_combos)
    return survivors

# ------------------- UI -------------------
st.set_page_config(page_title="Tens Filter App", layout="wide")

st.sidebar.header("Inputs")
seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):").strip()
prevs = [st.sidebar.text_input(f"Draw {i}-back:", value="").strip() for i in range(2, 11)]
method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

hot_override = st.sidebar.text_input("Hot digits override (e.g. 0 2 8):", value="")
cold_override = st.sidebar.text_input("Cold digits override (e.g. 1,3,7):", value="")
due_override = st.sidebar.text_input("Due digits override (e.g. 4 5):", value="")

track = st.sidebar.text_input("Track/Test combo (e.g., 01234):").strip()
track_norm = "".join(sorted("".join(ch for ch in track if ch.isdigit()))) if track else None

preserve_tracked = st.sidebar.checkbox("Preserve tracked combos", value=False)
inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)
select_all_toggle = st.sidebar.checkbox("Select/Deselect all filters (shown)", value=False)
hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)

# Validate inputs early
if not _valid_draw(seed):
    st.info("Enter a valid 5-digit seed first (digits 0–6).")
    st.stop()
bad_prevs = [p for p in prevs if p and not _valid_draw(p)]
if bad_prevs:
    st.warning(f"Invalid previous draws: {', '.join(bad_prevs)}. Must be 5 digits, digits 0–6.")
    st.stop()

# Load filters
manual_path = _first_existing(MANUAL_FILTER_CSV_CANDIDATES)
zone_path = _first_existing(ZONE_FILTER_CSV_CANDIDATES)
manual_filters = load_filters(manual_path, is_zone=False)
zone_filters = load_filters(zone_path, is_zone=True)

# Hot/Cold/Due (auto with optional overrides)
auto_hot, auto_cold, auto_due = auto_hot_cold_due(seed, prevs)
parse_list = lambda txt: [int(t) for t in txt.replace(",", " ").split() if t.isdigit() and 0 <= int(t) <= 6]
hot = parse_list(hot_override) or auto_hot
cold = parse_list(cold_override) or auto_cold
due = parse_list(due_override) or auto_due

st.sidebar.markdown(f"**Auto ➜** Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}")
st.sidebar.markdown(f"**Using ➜** Hot {hot} | Cold {cold} | Due {due}")

# Generate, zone-filter, unique
raw = gen_raw_combos(seed, method)
raw_count = len(raw)
raw_after_ztens = apply_zone_filters(raw, zone_filters, seed, prevs, hot, cold, due)
after_ztens_count = len(raw_after_ztens)
unique_baseline = sorted(set(raw_after_ztens))
unique_count = len(unique_baseline)

# Optional injection of tracked combo
if track_norm and inject_tracked and track_norm not in set(unique_baseline):
    unique_baseline = sorted(set(unique_baseline + [track_norm]))
    unique_count = len(unique_baseline)

# Pipeline sidebar
st.sidebar.markdown("### Pipeline")
st.sidebar.write(f"Raw generated: **{raw_count}**")
st.sidebar.write(f"Survive percentile pre-dedup: **{after_ztens_count}**")
st.sidebar.write(f"Unique enumeration: **{unique_count}**")
remaining_placeholder = st.sidebar.empty()

# Manual filter initial cut counts
init_counts = {}
for f in manual_filters:
    cuts = 0
    if not f.get("code"):
        init_counts[f["id"]] = 0
        continue
    for c in unique_baseline:
        ctx = make_ctx(seed, prevs, c, hot, cold, due)
        try:
            if eval(f["code"], {}, ctx):
                cuts += 1
        except Exception:
            # ignore broken expressions, keep app running
            pass
    init_counts[f["id"]] = cuts

display_filters = sorted(
    (f for f in manual_filters if not f["is_zone"]),
    key=lambda f: -init_counts.get(f["id"], 0)
)
if hide_zero:
    display_filters = [f for f in display_filters if init_counts.get(f["id"], 0) > 0]

st.markdown("## 🛠 Manual Filters")
st.caption(f"Applicable filters: **{len(display_filters)}**")

selection_state = {}
for f in display_filters:
    cid = f["id"]
    cuts = init_counts.get(cid, 0)
    label_bits = [f"{cid}: {f['layman']}".strip()]
    if f.get("hist"):
        label_bits.append(f"hist {f['hist']}")
    label_bits.append(f"init cut {cuts}")
    label = " | ".join(label_bits)
    checked = st.checkbox(label, key=f"chk_{cid}", value=select_all_toggle)
    selection_state[cid] = bool(checked)

# Apply sequential filters
pool = list(unique_baseline)
track_status = None
if track_norm and track_norm not in pool:
    track_status = "not_generated"

for f in display_filters:
    if selection_state.get(f["id"]):
        survivors = []
        for c in pool:
            ctx = make_ctx(seed, prevs, c, hot, cold, due)
            try:
                if eval(f["code"], {}, ctx):
                    if track_norm and c == track_norm and track_status != "not_generated":
                        track_status = f"eliminated_by:{f['id']}"
                    continue
            except Exception:
                pass
            survivors.append(c)
        pool = survivors

if track_norm and track_status is None:
    track_status = "survived" if track_norm in pool else "not_generated"

remaining_placeholder.write(f"**Remaining after filters:** {len(pool)}")

st.markdown(f"### ✅ Final Survivors: **{len(pool)}**")
with st.expander("Show survivors"):
    for c in pool:
        st.write(c)

if track_norm:
    if track_status == "not_generated":
        st.sidebar.warning("Tracked combo was **not generated** (or removed by zone filters).")
    elif track_status == "survived":
        st.sidebar.success("Tracked combo **survived** all selected filters.")
    elif track_status and track_status.startswith("eliminated_by:"):
        st.sidebar.error(f"Tracked combo **eliminated** by filter {track_status.split(':',1)[1]}.")

if pool:
    df_out = pd.DataFrame({"tens_combo": pool})
    st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), "pb_tens_survivors.csv", "text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(pool), "pb_tens_survivors.txt", "text/plain")
