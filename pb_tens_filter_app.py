# pb_tens_filter_app.py  —  Powerball Tens-Only Manual Filter Runner
# Drop-in replacement. See notes in the sidebar.

from __future__ import annotations

import os, csv, io, hashlib
from itertools import product
from collections import Counter
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st


# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = "0123456"                  # Powerball main balls tens digits
LOW_SET     = {0,1,2,3,4}
HIGH_SET    = {5,6}

def sum_category(total: int) -> str:
    if 0 <= total <= 10:  return "Very Low"
    if 11 <= total <= 13: return "Low"
    if 14 <= total <= 17: return "Mid"
    return "High"


# ----------------------------------
# Filter file loader (CSV, compiled)
# ----------------------------------
CSV_FILENAME = "pb_tens_filters_adapted.csv"

def _coerce_str(s):
    return (s if isinstance(s,str) else "").strip().strip('"').strip("'")

def load_filters_from_csv(path: str) -> List[Dict[str,Any]]:
    if not os.path.exists(path):
        st.error(f"Filter file not found: {path}")
        st.stop()

    # We allow either comma CSV or TSV
    with open(path, "r", encoding="utf-8", newline="") as f:
        sniff = csv.Sniffer().sniff(f.read(4096))
        f.seek(0)
        reader = csv.DictReader(f, dialect=sniff)

        rows: List[Dict[str,Any]] = []
        seen_ids = set()
        auto_id = 1

        for raw in reader:
            row = { (k or "").lower().strip(): (v if v is not None else "") for k,v in raw.items() }

            fid   = _coerce_str(row.get("id") or row.get("fid"))
            name  = _coerce_str(row.get("name"))
            appl  = _coerce_str(row.get("applicable_if") or row.get("applicable"))
            expr  = _coerce_str(row.get("expression") or row.get("expr"))
            stat  = _coerce_str(row.get("stat"))              # historical stat, optional
            lay   = _coerce_str(row.get("layman") or row.get("layman_explanation"))

            if not fid:
                fid = f"F{auto_id:04d}"
                auto_id += 1
            if fid in seen_ids:
                # keep it unique by appending a short hash of content
                h = hashlib.sha1((fid+expr+appl).encode("utf-8")).hexdigest()[:6]
                fid = f"{fid}_{h}"
            seen_ids.add(fid)

            # compile safely
            try:
                ac = compile(appl if appl else "True", "<applicable>", "eval")
            except SyntaxError as e:
                st.warning(f"Skipping filter {fid} — bad applicable_if: {e}")
                continue
            try:
                ec = compile(expr if expr else "False", "<expr>", "eval")
            except SyntaxError as e:
                st.warning(f"Skipping filter {fid} — bad expression: {e}")
                continue

            rows.append({
                "id": fid,
                "name": name or fid,
                "applicable_if": appl or "True",
                "expression": expr or "False",
                "expr_code": ec,
                "applicable_code": ac,
                "stat": stat,               # historical x/y label, if present in CSV
                "layman": lay,
            })

        return rows


# ---------------------------
# Tens combos generation
# ---------------------------
def generate_tens_combos(seed_tens: str, method: str) -> List[str]:
    """Return sorted unique keys of length 5, digits 0..6."""
    s = "".join(sorted(seed_tens))
    out = set()
    if method == "1-digit":
        for d in s:
            for p in product(TENS_DOMAIN, repeat=4):
                out.add("".join(sorted(d + "".join(p))))
    else:  # "2-digit pair"
        pairs = { "".join(sorted(s[i] + s[j])) for i in range(len(s)) for j in range(i+1, len(s)) }
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                out.add("".join(sorted(pair + "".join(p))))
    return sorted(out)


# ---------------------------
# Context builder
# ---------------------------
def multiset_shared(a_digits: List[int], b_digits: List[int]) -> int:
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def digits_from_str(s: str) -> List[int]:
    return [int(x) for x in s.strip() if x.isdigit()]

def build_ctx(
    combo_key: str,
    seed_tens: str,
    prevs: List[str],               # up to 6 strings
    hot_digits_input: str,
    cold_digits_input: str,
    due_digits_current: List[int]
) -> Dict[str,Any]:

    combo_tens = [int(c) for c in combo_key]
    seed       = digits_from_str(seed_tens)
    prevs_int  = [digits_from_str(p) for p in prevs]
    prev1 = prevs_int[0] if len(prevs_int) >= 1 else []
    prev2 = prevs_int[1] if len(prevs_int) >= 2 else []
    prev3 = prevs_int[2] if len(prevs_int) >= 3 else []
    prev4 = prevs_int[3] if len(prevs_int) >= 4 else []
    prev5 = prevs_int[4] if len(prevs_int) >= 5 else []
    prev6 = prevs_int[5] if len(prevs_int) >= 6 else []

    # summary values
    tens_sum        = sum(combo_tens)
    tens_even       = sum(1 for d in combo_tens if d % 2 == 0)
    tens_odd        = 5 - tens_even
    tens_unique     = len(set(combo_tens))
    tens_range      = (max(combo_tens) - min(combo_tens)) if combo_tens else 0
    tens_low        = sum(1 for d in combo_tens if d in LOW_SET)
    tens_high       = sum(1 for d in combo_tens if d in HIGH_SET)
    seed_tens_sum   = sum(seed) if seed else 0

    hot_digits = [int(x) for x in hot_digits_input.split(",") if x.strip().isdigit() and 0 <= int(x) <= 6]
    cold_digits = [int(x) for x in cold_digits_input.split(",") if x.strip().isdigit() and 0 <= int(x) <= 6]
    due_digits = list(due_digits_current or [])

    # some handy sets used in filters in your projects
    last2  = set(seed) | set(prev1)
    common = set(seed) & set(prev1)

    ctx = {
        "combo_tens": combo_tens,
        "seed_tens": seed,
        "prev_seed_tens": prev1,
        "prev_prev_seed_tens": prev2,
        "prev3_seed_tens": prev3,
        "prev4_seed_tens": prev4,
        "prev5_seed_tens": prev5,
        "prev6_seed_tens": prev6,

        "tens_sum": tens_sum,
        "seed_tens_sum": seed_tens_sum,
        "tens_even_count": tens_even,
        "tens_odd_count": tens_odd,
        "tens_unique_count": tens_unique,
        "tens_range": tens_range,
        "tens_low_count": tens_low,
        "tens_high_count": tens_high,

        "last2": last2,
        "common_to_both": common,

        "hot_digits": hot_digits,
        "cold_digits": cold_digits,
        "due_digits": due_digits,

        "sum_category": sum_category,
        "Counter": Counter,
        "shared_tens": multiset_shared,
    }
    return ctx


# ---------------------------
# Helpers
# ---------------------------
def normalize_combo_text(text: str) -> Tuple[List[str], List[str]]:
    raw_tokens = []
    for line in text.splitlines():
        for token in line.replace(",", " ").split():
            raw_tokens.append(token.strip())
    normalized, invalid = [], []
    for tok in raw_tokens:
        digits = [c for c in tok if c.isdigit()]
        if len(digits) != 5 or any(c not in TENS_DOMAIN for c in digits):
            invalid.append(tok); continue
        normalized.append("".join(sorted(digits)))
    seen, out = set(), []
    for n in normalized:
        if n not in seen:
            out.append(n); seen.add(n)
    return out, invalid


# ============================================================================
# Streamlit UI
# ============================================================================
st.set_page_config(page_title="Powerball Tens-Only — Manual Filter Runner", layout="wide")
st.sidebar.header("🎯 Powerball Tens-Only — Manual Filter Runner")
st.sidebar.caption("Loads **pb_tens_filters_adapted.csv** and applies them to tens-digit combos.")


# ---- Run/refresh control & run tag (namespacing widget keys safely)
if "_run_tag" not in st.session_state:
    st.session_state["_run_tag"] = 0

col_run1, col_run2 = st.sidebar.columns([1,1])
with col_run1:
    if st.button("▶️ Run / Refresh"):
        st.session_state["_run_tag"] += 1
with col_run2:
    if st.button("♻️ Clear selections"):
        # wipe filter checkboxes from session
        to_del = [k for k in st.session_state.keys() if k.startswith("fltkey|")]
        for k in to_del: del st.session_state[k]

RUN_TAG = st.session_state["_run_tag"]


# ---- Optional extra filter CSV upload
up = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])
extra_filters_path = None
if up is not None:
    extra_filters_path = f"_user_filters_{RUN_TAG}.csv"
    with open(extra_filters_path, "wb") as f:
        f.write(up.getbuffer())


# ---- Load filters
filter_paths = [CSV_FILENAME]
if extra_filters_path:
    filter_paths.append(extra_filters_path)

filters: List[Dict[str,Any]] = []
for p in filter_paths:
    filters.extend(load_filters_from_csv(p))


# ---- Seeds (current + up to 6 prior)
seed      = st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):", value="", placeholder="e.g., 23345").strip()
prev1     = st.sidebar.text_input("Prev tens (Draw 2-back, optional):", value="").strip()
prev2     = st.sidebar.text_input("Prev-prev tens (Draw 3-back, optional):", value="").strip()
prev3     = st.sidebar.text_input("Draw 4-back tens (optional):", value="").strip()
prev4     = st.sidebar.text_input("Draw 5-back tens (optional):", value="").strip()
prev5     = st.sidebar.text_input("Draw 6-back tens (optional):", value="").strip()

method    = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

hot_manual  = st.sidebar.text_input("Hot tens digits (manual 0–6, comma-separated):", value="")
cold_manual = st.sidebar.text_input("Cold tens digits (manual 0–6, comma-separated):", value="")
due_manual  = st.sidebar.text_input("Due digits (manual 0–6, comma-separated):", value="")

disable_due_if_empty = st.sidebar.checkbox("Disable due-based filters when due set is empty", value=True)

# Track/Test
st.sidebar.markdown("---")
track_text = st.sidebar.text_area("Track/Test combos (e.g., 00123, 23345; newline or comma-separated):", height=90)
preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
inject_tracked   = st.sidebar.checkbox("Inject tracked combos if not generated", value=False)

# Master buttons
st.sidebar.markdown("---")
col_mb1, col_mb2 = st.sidebar.columns(2)
with col_mb1:
    select_all_btn = st.button("Select all (shown)")
with col_mb2:
    clear_all_btn  = st.button("Clear all (shown)")

hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)


# ---- Input validation (only now we proceed if Run pressed at least once)
if RUN_TAG == 0:
    st.info("Set inputs then click **Run / Refresh**.")
    st.stop()

if len(seed) != 5 or (not seed.isdigit()) or any(c not in TENS_DOMAIN for c in seed):
    st.sidebar.error("Seed tens must be exactly 5 digits in 0–6 (e.g., 23345).")
    st.stop()

for label, val in [("Prev 2-back", prev1), ("Prev 3-back", prev2), ("Prev 4-back", prev3),
                   ("Prev 5-back", prev4), ("Prev 6-back", prev5)]:
    if val and (len(val)!=5 or (not val.isdigit()) or any(c not in TENS_DOMAIN for c in val)):
        st.sidebar.error(f"{label} tens must be 5 digits in 0–6 or left blank.")
        st.stop()


# ---- Auto Hot/Cold/Due if all 6 prev are present; otherwise manual
prevs_all = [prev1, prev2, prev3, prev4, prev5]
have_all_six = all(len(x)==5 for x in prevs_all)

auto_hot: List[int] = []
auto_cold: List[int] = []
auto_due: List[int] = []

if have_all_six:
    freq = Counter()
    for pv in prevs_all:
        for ch in pv:
            freq[int(ch)] += 1
    # Hot = top 3 by frequency (ties kept)
    if freq:
        maxf = max(freq.values())
        auto_hot = sorted([d for d,c in freq.items() if c == maxf])[:3]
        minf = min(freq.values())
        auto_cold = sorted([d for d,c in freq.items() if c == minf])[:3]
    # Due = digits not seen in the 6 prior draws
    seen = set()
    for pv in prevs_all:
        for ch in pv: seen.add(int(ch))
    auto_due = [d for d in range(7) if d not in seen]

# Final due set
if due_manual.strip():
    due_set = [int(x) for x in due_manual.split(",") if x.strip().isdigit() and 0 <= int(x) <= 6]
elif have_all_six:
    due_set = auto_due
else:
    due_set = []

# ---- Build generated pool
combos = generate_tens_combos(seed, method)

# Tracked combos normalization / injection
tracked_norm, invalid_tokens = normalize_combo_text(track_text)
if invalid_tokens:
    st.sidebar.warning("Ignored invalid combo tokens: " + ", ".join(invalid_tokens[:5]) + (" ..." if len(invalid_tokens)>5 else ""))

gen_set = set(combos)
audit: Dict[str,Dict[str,Any]] = {
    c: {
        "combo": c, "generated": (c in gen_set), "survived": None,
        "eliminated": False, "eliminated_by": None, "eliminated_name": None, "eliminated_order": None,
        "would_eliminate_by": None, "would_eliminate_name": None, "would_eliminate_order": None,
        "injected": False, "preserved": bool(preserve_tracked),
    } for c in tracked_norm
}
if inject_tracked:
    for c in tracked_norm:
        if c not in gen_set:
            combos.append(c)
            gen_set.add(c)
            audit[c]["injected"] = True


# ---- Compute initial cuts per filter
def initial_cuts(flt) -> int:
    cnt = 0
    for ck in combos:
        ctx = build_ctx(ck, seed, [prev1,prev2,prev3,prev4,prev5], hot_manual or ",".join(map(str,auto_hot)),
                        cold_manual or ",".join(map(str,auto_cold)),
                        due_set)
        try:
            if eval(flt["applicable_code"], ctx, ctx):
                # disable due filters if requested & due empty
                if disable_due_if_empty and (not due_set) and "due_digits" in (flt.get("expression","")):
                    continue
                if eval(flt["expr_code"], ctx, ctx):
                    cnt += 1
        except Exception:
            # treat as no-cut if it errors
            pass
    return cnt

init_counts = {flt["id"]: initial_cuts(flt) for flt in filters}
sorted_filters = sorted(filters, key=lambda f: (init_counts[f["id"]] == 0, -init_counts[f["id"]]))

display_filters = [f for f in sorted_filters if init_counts[f["id"]] > 0] if hide_zero else sorted_filters

# Master select/clear buttons affect only currently displayed filters
if select_all_btn:
    for idx, flt in enumerate(display_filters):
        st.session_state[f"fltkey|{RUN_TAG}|{flt['id']}|{idx}"] = True
if clear_all_btn:
    for idx, flt in enumerate(display_filters):
        st.session_state[f"fltkey|{RUN_TAG}|{flt['id']}|{idx}"] = False


# ---- Sequential application with per-filter checkboxes
st.header("🛠️ Manual Filters")

pool = list(combos)
dynamic_counts: Dict[str,int] = {}
order_index = 0

for idx, flt in enumerate(display_filters):
    fid   = flt["id"]
    name  = flt.get("name") or fid
    stat  = flt.get("stat") or ""      # historical stat from CSV (e.g., 311/311)
    lay   = flt.get("layman") or ""

    ic = init_counts[fid]
    label_left = f"{fid}: {name}"
    label_right = f"init cuts {ic}" + (f" • hist {stat}" if stat else "")
    label = f"{label_left} — {label_right}"
    ui_key = f"fltkey|{RUN_TAG}|{fid}|{idx}"
    checked = st.checkbox(label, key=ui_key, value=st.session_state.get(ui_key, False), help=(lay or None))

    if checked:
        survivors_pool = []
        dc = 0
        for ck in pool:
            ctx = build_ctx(ck, seed, [prev1,prev2,prev3,prev4,prev5], hot_manual or ",".join(map(str,auto_hot)),
                            cold_manual or ",".join(map(str,auto_cold)),
                            due_set)
            eliminate = False
            try:
                if eval(flt["applicable_code"], ctx, ctx):
                    if not (disable_due_if_empty and (not due_set) and "due_digits" in (flt.get("expression",""))):
                        eliminate = bool(eval(flt["expr_code"], ctx, ctx))
            except Exception:
                eliminate = False

            is_tracked = ck in audit
            if eliminate:
                if is_tracked and preserve_tracked:
                    # record would-eliminate info but keep it
                    if audit[ck]["would_eliminate_by"] is None:
                        audit[ck]["would_eliminate_by"]   = fid
                        audit[ck]["would_eliminate_name"] = name
                        audit[ck]["would_eliminate_order"]= order_index + 1
                    survivors_pool.append(ck)
                else:
                    dc += 1
                    if is_tracked and not audit[ck]["eliminated"]:
                        audit[ck]["eliminated"]       = True
                        audit[ck]["eliminated_by"]    = fid
                        audit[ck]["eliminated_name"]  = name
                        audit[ck]["eliminated_order"] = order_index + 1
            else:
                survivors_pool.append(ck)

        pool = survivors_pool
        dynamic_counts[fid] = dc
        order_index += 1
    else:
        dynamic_counts[fid] = 0


st.subheader(f"Remaining after filters: {len(pool)}")

# ---- Audit for tracked combos
if tracked_norm:
    st.markdown("### 🔎 Tracked/Preserved Combos — Audit")
    survivors_set = set(pool)
    for c in tracked_norm:
        if c in audit:
            audit[c]["survived"] = (c in survivors_set)
    df_audit = pd.DataFrame([
        {
            "combo": c,
            "generated": audit[c]["generated"],
            "survived": audit[c]["survived"],
            "eliminated": audit[c]["eliminated"],
            "eliminated_by": audit[c]["eliminated_by"],
            "eliminated_order": audit[c]["eliminated_order"],
            "eliminated_name": audit[c]["eliminated_name"],
            "would_eliminate_by": audit[c]["would_eliminate_by"],
            "would_eliminate_order": audit[c]["would_eliminate_order"],
            "would_eliminate_name": audit[c]["would_eliminate_name"],
            "injected": audit[c]["injected"],
            "preserved": audit[c]["preserved"],
        } for c in tracked_norm
    ])
    st.dataframe(df_audit, use_container_width=True, height=220)
    st.download_button("Download audit (CSV)", df_audit.to_csv(index=False).encode("utf-8"),
                       file_name="pb_tens_audit_tracked.csv", mime="text/csv")

# ---- Survivors
st.markdown("### ✅ Survivors")
with st.expander("Show remaining combinations"):
    for c in pool:
        st.write(c)

# ---- Downloads
df_out = pd.DataFrame({"tens_combo": pool})
st.download_button("Download survivors (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                   file_name="pb_tens_survivors.csv", mime="text/csv")
st.download_button("Download survivors (TXT)", ("\n".join(pool)).encode("utf-8"),
                   file_name="pb_tens_survivors.txt", mime="text/plain")
