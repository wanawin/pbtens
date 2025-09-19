import streamlit as st
from itertools import product
from collections import Counter
import csv, os
import pandas as pd

# =========================================
# Tens-only domain and simple helpers
# =========================================
TENS_DOMAIN = "0123456"
LOW_SET  = set([0,1,2,3,4])
HIGH_SET = set([5,6])

def sum_category(total: int) -> str:
    if 0 <= total <= 10:
        return "Very Low"
    elif 11 <= total <= 13:
        return "Low"
    elif 14 <= total <= 17:
        return "Mid"
    else:
        return "High"

def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

# =========================================
# Load filters from CSV(s)
# Required columns: id, name, expression
# Optional: applicable_if, enabled, stat
# =========================================
def load_filters(paths):
    filters = []
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = { (k or "").strip().lower(): (v if isinstance(v,str) else v) for k,v in raw.items() }
                row["id"]   = (row.get("id") or row.get("fid") or "").strip()
                row["name"] = (row.get("name") or "").strip()
                expr_raw    = (row.get("expression") or "").strip().strip('"').strip("'")
                app_raw     = (row.get("applicable_if") or "True").strip().strip('"').strip("'")
                # Keep original text for label/debug
                row["expr_str"] = expr_raw
                row["applicable_str"] = app_raw
                # compile expression/applicable
                try:
                    row["applicable_code"] = compile(app_raw or "True", "<applicable>", "eval")
                    row["expr_code"]       = compile(expr_raw or "False", "<expr>", "eval")
                except SyntaxError as e:
                    # Skip bad rows but show a soft warning
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                row["enabled_default"] = (str(row.get("enabled","")).lower() == "true")
                row["stat"] = (row.get("stat") or "").strip()   # historical x/155, etc.
                filters.append(row)
    return filters

# =========================================
# Tens-only combo generation (unique keys)
# Keys are sorted 5-char strings, e.g. '00123'
# =========================================
def generate_tens_combinations(seed_tens: str, method: str):
    seed_tens = "".join(sorted(seed_tens))
    combos_set = set()
    if method == "1-digit":
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                combos_set.add("".join(sorted(d + "".join(p))))
    else:
        # 2-digit pair method
        pairs = {
            "".join(sorted((seed_tens[i], seed_tens[j])))
            for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens))
        }
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                combos_set.add("".join(sorted(pair + "".join(p))))
    return sorted(combos_set)

# =========================================
# Context construction per combo
# =========================================
def build_ctx(seed_tens_str: str,
              p2_str: str, p3_str: str, p4_str: str, p5_str: str, p6_str: str,
              combo_str: str,
              hot_digits_list, cold_digits_list, due_digits_list):

    def to_list(s): return [int(x) for x in s] if s else []

    seed_tens = to_list(seed_tens_str)
    p2 = to_list(p2_str)
    p3 = to_list(p3_str)
    p4 = to_list(p4_str)
    p5 = to_list(p5_str)
    p6 = to_list(p6_str)

    combo_tens = [int(c) for c in combo_str]

    tens_sum   = sum(combo_tens)
    tens_even  = sum(1 for d in combo_tens if d % 2 == 0)
    tens_odd   = 5 - tens_even
    tens_uniq  = len(set(combo_tens))
    tens_range = (max(combo_tens) - min(combo_tens)) if combo_tens else 0
    tens_low   = sum(1 for d in combo_tens if d in LOW_SET)
    tens_high  = sum(1 for d in combo_tens if d in HIGH_SET)

    # unions commonly used in filters
    last2  = set(seed_tens) | set(p2)
    common = set(seed_tens) & set(p2)

    ctx = {
        # core
        "combo_tens": combo_tens,
        "seed_tens":  seed_tens,
        "prev_seed_tens": p2,
        "prev_prev_seed_tens": p3,
        # also expose up to 6-back explicitly
        "seed2_tens": p2, "seed3_tens": p3, "seed4_tens": p4, "seed5_tens": p5, "seed6_tens": p6,
        "variant_name": "tens",  # so filters referencing it never crash

        # scalar features
        "tens_sum": tens_sum,
        "tens_even_count": tens_even,
        "tens_odd_count":  tens_odd,
        "tens_unique_count": tens_uniq,
        "tens_range": tens_range,
        "tens_low_count": tens_low,
        "tens_high_count": tens_high,

        # sets & helpers
        "last2": last2,
        "common_to_both": common,
        "hot_digits":  list(hot_digits_list or []),
        "cold_digits": list(cold_digits_list or []),
        "due_digits":  list(due_digits_list or []),

        # utilities
        "Counter": Counter,
        "sum_category": sum_category,
        "shared_tens": multiset_shared,
    }
    return ctx

# =========================================
# Hot / Cold auto-calc from 6 previous draws
# Only runs if ALL six prev strings are present (5 digits each in 0..6)
# =========================================
def auto_hot_cold_from_six(prevs):
    """prevs: list[str] of length 6; each is 5 digits 0..6; return (hot_list, cold_list)"""
    all_ok = all(isinstance(s,str) and len(s)==5 and all(c in TENS_DOMAIN for c in s) for s in prevs)
    if not all_ok:
        return [], []
    cnt = Counter()
    for s in prevs:
        for ch in s:
            cnt[int(ch)] += 1
    if not cnt:
        return [], []
    maxf = max(cnt.values()); minf = min(cnt.values())
    hot  = sorted([d for d,c in cnt.items() if c == maxf])
    cold = sorted([d for d,c in cnt.items() if c == minf])
    return hot, cold

# =========================================
# Track/Test helper
# =========================================
def normalize_combo_text(text: str):
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

# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Powerball Tens-Only — Manual Filter Runner", layout="wide")
st.sidebar.header("🎯 Powerball Tens-Only — Manual Filter Runner")

# Filters
use_default = st.sidebar.checkbox("Use default adapted filters (pb_tens_filters_adapted.csv)", value=True)
uploaded_filters = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])

# Build filter path list
filter_paths = []
default_filters_path = "pb_tens_filters_adapted.csv"
if use_default and os.path.exists(default_filters_path):
    filter_paths.append(default_filters_path)

if uploaded_filters is not None:
    upath = "user_filters.csv"
    with open(upath, "wb") as f:
        f.write(uploaded_filters.getbuffer())
    filter_paths.append(upath)

filters = load_filters(filter_paths)

# Seeds
seed   = st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):",  help="Required").strip()
p2     = st.sidebar.text_input("Prev tens (Draw 2-back, optional):",      help="Optional").strip()
p3     = st.sidebar.text_input("Prev-prev tens (Draw 3-back, optional):", help="Optional").strip()
p4     = st.sidebar.text_input("Draw 4-back tens (optional):",            help="Optional").strip()
p5     = st.sidebar.text_input("Draw 5-back tens (optional):",            help="Optional").strip()
p6     = st.sidebar.text_input("Draw 6-back tens (optional):",            help="Optional").strip()

method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

# Manual Hot/Cold (override)
hot_manual  = st.sidebar.text_input("Hot tens digits (comma-separated 0–6, optional):").strip()
cold_manual = st.sidebar.text_input("Cold tens digits (comma-separated 0–6, optional):").strip()

# Due digits (manual)
due_text = st.sidebar.text_input("Due digits (manual 0–6, comma-separated):", "").strip()
disable_due_empty = st.sidebar.checkbox("Disable due-based filters when due set empty", value=True)

# Tracking / options
track_text = st.sidebar.text_area("Track/Test combos (newline/comma-separated):", height=80)
preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=False)
inject_tracked   = st.sidebar.checkbox("Inject tracked combos if not generated", value=False)
select_all       = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
hide_zero        = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=False)

# Validate seed
def valid_draw(s): return len(s)==5 and s.isdigit() and all(c in TENS_DOMAIN for c in s)
if not valid_draw(seed):
    st.info("Enter a valid **Seed tens** (exactly 5 digits in 0–6) to run.")
    st.stop()

# Auto Hot/Cold from six draws if all present
hot_auto, cold_auto = auto_hot_cold_from_six([seed, p2, p3, p4, p5, p6])  # includes 1-back too
def parse_csv_digits(s):
    out=[]
    for t in s.split(","):
        t=t.strip()
        if t.isdigit():
            v=int(t)
            if 0<=v<=6: out.append(v)
    return out

hot_digits  = parse_csv_digits(hot_manual)  if hot_manual  else (hot_auto  or [])
cold_digits = parse_csv_digits(cold_manual) if cold_manual else (cold_auto or [])
due_digits  = parse_csv_digits(due_text)

# Show what we're using
st.sidebar.markdown(f"**Auto Hot:** {hot_auto if hot_auto else '—'}  \n**Auto Cold:** {cold_auto if cold_auto else '—'}")
if hot_manual:  st.sidebar.caption(f"Using manual Hot: {hot_digits}")
if cold_manual: st.sidebar.caption(f"Using manual Cold: {cold_digits}")
if not due_digits:
    st.sidebar.caption("Due set: []")
else:
    st.sidebar.caption(f"Due set: {sorted(set(due_digits))}")

# Generate pool
combos = generate_tens_combinations(seed, method)
generated_set = set(combos)

# Normalize tracked
tracked_norm, invalid_tokens = normalize_combo_text(track_text)
if invalid_tokens:
    st.sidebar.warning("Ignored invalid entries: " + ", ".join(invalid_tokens[:6]) + (" ..." if len(invalid_tokens)>6 else ""))

# Inject tracked if requested
if inject_tracked:
    for c in tracked_norm:
        if c not in generated_set:
            combos.append(c); generated_set.add(c)

# Initial elimination counts per filter (before user clicks anything)
init_counts = {flt["id"]: 0 for flt in filters}
init_errors = {}
for flt in filters:
    ic = 0
    err = ""
    # Skip due-based filters entirely if asked and due empty
    if disable_due_empty and not due_digits and "due" in (flt.get("expr_str","") or "").lower():
        init_counts[flt["id"]] = 0
        continue
    for combo in combos:
        try:
            ctx = build_ctx(seed, p2, p3, p4, p5, p6, combo, hot_digits, cold_digits, due_digits)
            if eval(flt["applicable_code"], ctx, ctx) and eval(flt["expr_code"], ctx, ctx):
                ic += 1
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
    init_counts[flt["id"]] = ic
    if err:
        init_errors[flt["id"]] = err

# Filters sorted by initial aggressiveness (desc), with 0-cuts optionally hidden
sorted_filters = sorted(filters, key=lambda flt: (init_counts[flt["id"]]==0, -init_counts[flt["id"]], flt["id"]))
display_filters = [f for f in sorted_filters if init_counts[f["id"]]>0] if hide_zero else sorted_filters

# =========================
# Apply filters (manual)
# =========================
pool = list(combos)
tracked_set = set(tracked_norm)
audit = {
    c: {
        "combo": c, "generated": (c in generated_set),
        "survived": None,
        "eliminated": False, "eliminated_by": None, "eliminated_name": None, "eliminated_order": None,
        "would_eliminate_by": None, "would_eliminate_name": None, "would_eliminate_order": None,
        "injected": (inject_tracked and c not in generated_set), "preserved": bool(preserve_tracked),
    } for c in tracked_norm
}

st.header("🧰 Manual Filters")
st.caption("Filters are sorted by **initial aggressiveness** (cuts before any are applied). Default is **deselected**.")

dynamic_counts = {}
order_index = 0
for flt in display_filters:
    fid   = flt.get("id","?").strip()
    fname = flt.get("name","").strip()
    hist  = flt.get("stat","").strip()  # historical stat from CSV, e.g. 18/311
    ic    = init_counts.get(fid, 0)
    err   = init_errors.get(fid)
    key   = f"chk_flt_{fid}"   # unique key avoids duplicate-element errors

    label = f"{fid}: {fname} — hist {hist or '—'} | init cuts {ic}"
    if err:
        label += f" ⚠ {err}"

    checked = st.checkbox(label, key=key, value=(select_all and flt.get("enabled_default", False)))
    if not checked:
        dynamic_counts[fid] = 0
        continue

    order_index += 1
    survivors_pool = []
    dc = 0
    for combo in pool:
        try:
            ctx = build_ctx(seed, p2, p3, p4, p5, p6, combo, hot_digits, cold_digits, due_digits)
            # Optionally skip due-based filters when due is empty
            if disable_due_empty and not due_digits and "due" in (flt.get("expr_str","") or "").lower():
                eliminate = False
            else:
                eliminate = eval(flt["applicable_code"], ctx, ctx) and eval(flt["expr_code"], ctx, ctx)
        except Exception:
            eliminate = False

        is_tracked = combo in tracked_set
        if eliminate:
            if is_tracked and preserve_tracked:
                info = audit.get(combo)
                if info and info.get("would_eliminate_by") is None:
                    info["would_eliminate_by"]   = fid
                    info["would_eliminate_name"] = fname
                    info["would_eliminate_order"]= order_index
                survivors_pool.append(combo)
                continue
            dc += 1
            if is_tracked and not audit[combo]["eliminated"]:
                audit[combo]["eliminated"]       = True
                audit[combo]["eliminated_by"]    = fid
                audit[combo]["eliminated_name"]  = fname
                audit[combo]["eliminated_order"] = order_index
        else:
            survivors_pool.append(combo)
    dynamic_counts[fid] = dc
    pool = survivors_pool

st.subheader(f"Remaining after filters: {len(pool)}")

# =========================
# Audit for tracked/test combos
# =========================
survivors_set = set(pool)
if tracked_norm:
    st.markdown("### 🔎 Tracked/Preserved Combos — Audit")
    audit_rows = []
    for c in tracked_norm:
        info = audit.get(c, {})
        audit_rows.append({
            "combo": c,
            "generated": info.get("generated", False),
            "survived": (c in survivors_set),
            "eliminated": info.get("eliminated", False),
            "eliminated_by": info.get("eliminated_by"),
            "eliminated_order": info.get("eliminated_order"),
            "eliminated_name": info.get("eliminated_name"),
            "would_eliminate_by": info.get("would_eliminate_by"),
            "would_eliminate_order": info.get("would_eliminate_order"),
            "would_eliminate_name": info.get("would_eliminate_name"),
            "injected": info.get("injected", False),
            "preserved": info.get("preserved", False),
        })
    df_audit = pd.DataFrame(audit_rows, columns=[
        "combo","generated","survived",
        "eliminated","eliminated_by","eliminated_order","eliminated_name",
        "would_eliminate_by","would_eliminate_order","would_eliminate_name",
        "injected","preserved"
    ])
    st.dataframe(df_audit, use_container_width=True)
    st.download_button("Download audit (CSV)", df_audit.to_csv(index=False), file_name="pb_tens_audit_tracked.csv", mime="text/csv")

# =========================
# Survivors (collapsible) + downloads
# =========================
st.markdown("### ✅ Survivors")
with st.expander("Show remaining combinations"):
    if pool:
        tracked_survivors = [c for c in pool if c in tracked_set]
        if tracked_survivors:
            st.write("**Tracked survivors:**")
            for c in tracked_survivors:
                info = audit.get(c, {})
                if info and info.get("would_eliminate_by"):
                    st.write(f"{c} — ⚠ would be eliminated by {info['would_eliminate_by']} at step {info.get('would_eliminate_order')} ({info.get('would_eliminate_name')}) — preserved")
                else:
                    st.write(c)
            st.write("---")
        for c in pool:
            if c not in tracked_set:
                st.write(c)
    else:
        st.write("No survivors.")

# Downloads
df_out = pd.DataFrame({"tens_combo": pool})
st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), file_name="pb_tens_survivors.csv",  mime="text/csv")
st.download_button("Download survivors (TXT)", "\n".join(pool),         file_name="pb_tens_survivors.txt", mime="text/plain")
