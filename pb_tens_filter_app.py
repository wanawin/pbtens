# pb_tens_filter_app.py — Powerball Tens-Only Manual Filter Runner
# - Always show applicable manual filters (ordered by aggressiveness)
# - Master select/deselect (default OFF)
# - Run/Refresh button (prevents auto-run while typing)
# - Percentile filters are pre-dedup (unchanged)
# - Up to 6 optional previous draws (Prev1..Prev6)
# - Auto Hot/Cold computed from last K draws; fed into filter context
# - Debug pipeline panel; tracked audit; downloads

import os, csv, re
from itertools import product
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = '0123456'   # allowed tens digits
LOW_SET  = {0,1,2,3,4}
HIGH_SET = {5,6}

def sum_category(total: int) -> str:
    if 0 <= total <= 10:   return 'Very Low'
    if 11 <= total <= 13:  return 'Low'
    if 14 <= total <= 17:  return 'Mid'
    return 'High'

# ========================
# Filter loading (15-col)
# ========================
def load_filters(paths) -> List[Dict]:
    """Load filters from one or more CSVs (adapted 15-col format).
       Compiles 'applicable_if' and 'expression'. Marks percentile file by name."""
    filters = []
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = {k.lower(): v for k, v in raw.items()}
                row['id'] = (row.get('id') or row.get('fid') or '').strip()
                for key in ('name', 'applicable_if', 'expression'):
                    if key in row and isinstance(row[key], str):
                        row[key] = row[key].strip().strip('"').strip("'")
                # normalize a couple glitches
                row['expression'] = (row.get('expression') or 'False').replace('!==', '!=')
                row['expr_str']   = row['expression']
                applicable = row.get('applicable_if') or 'True'
                expr       = row.get('expression')    or 'False'
                try:
                    row['applicable_code'] = compile(applicable, '<applicable>', 'eval')
                    row['expr_code']        = compile(expr,       '<expr>',       'eval')
                except SyntaxError as e:
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')} from {os.path.basename(path)}: {e}")
                    continue
                row['enabled_default'] = (str(row.get('enabled','')).lower() == 'true')
                row['source_path']     = os.path.basename(path)
                row['is_percentile']   = (row['source_path'] == 'pb_tens_percentile_filters.csv')
                # strong unique key for UI widgets
                row['_ui_key']         = f"{row['id']}__{row['source_path']}"
                filters.append(row)
    return filters

# ========================
# Tens combo generation
# ========================
def generate_tens_combinations_both(seed_tens: str, method: str) -> Tuple[List[str], List[str]]:
    """
    Return (RAW_with_duplicates, UNIQUE_sorted) keys of 5 digits (0..6),
    where each key is the sorted multiset, e.g. '00123'.
    1-digit: choose 1 from seed, + any 4 from domain
    2-digit pair: choose any pair from seed, + any 3 from domain
    """
    seed_tens = ''.join(sorted(seed_tens))
    raw, uniq = [], set()
    if method == '1-digit':
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(p)))
                raw.append(key); uniq.add(key)
    else:
        pairs = {''.join(sorted((seed_tens[i], seed_tens[j])))
                 for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(p)))
                raw.append(key); uniq.add(key)
    return raw, sorted(uniq)

# ========================
# Helpers (tens)
# ========================
def parse_tens_5(s: str) -> str:
    """Accept '11344' OR '1,1,3,4,4' etc.; return exactly 5 digits 0–6 or '' if invalid."""
    if not s: return ""
    digs = re.findall(r"[0-6]", s)
    return ''.join(digs) if len(digs) == 5 else ""

def parse_digit_list(s: str) -> List[int]:
    out = []
    if not s: return out
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            v = int(tok)
            if 0 <= v <= 6:
                out.append(v)
    return out

def compute_hot_cold_from_history(history: List[List[int]], k: int) -> Tuple[List[int], List[int], pd.DataFrame]:
    """Hot = most frequent; Cold = least frequent across last k draws in 'history' (each draw is list of 5 tens digits)."""
    if not history or k <= 0:
        df = pd.DataFrame({"digit": list(range(7)), "count": [0]*7})
        return [], [], df
    window = history[:k]  # history built newest→older
    cnt = Counter()
    for draw in window:
        cnt.update(draw)
    freq = {d: cnt.get(d, 0) for d in range(7)}
    df = pd.DataFrame({"digit": list(freq.keys()), "count": list(freq.values())}).sort_values("count", ascending=False).reset_index(drop=True)
    if not cnt:
        return [], [], df
    maxf, minf = max(cnt.values()), min(cnt.values())
    hot  = sorted([d for d,c in cnt.items() if c == maxf])
    cold = sorted([d for d,c in cnt.items() if c == minf])
    return hot, cold, df

def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def build_ctx(seed_tens_str: str,
              prev_chain_strs: List[str],
              combo_str: str,
              hot_digits: List[int],
              cold_digits: List[int],
              due_digits_param: List[int]):
    seed_tens = [int(x) for x in seed_tens_str] if seed_tens_str else []
    prev_chain = [[int(x) for x in p] for p in prev_chain_strs if p]
    combo_tens = [int(c) for c in combo_str] if combo_str else []

    tens_sum    = sum(combo_tens)
    tens_even   = sum(1 for d in combo_tens if d % 2 == 0)
    tens_odd    = 5 - tens_even
    tens_unique = len(set(combo_tens))
    tens_range  = (max(combo_tens) - min(combo_tens)) if combo_tens else 0
    tens_low    = sum(1 for d in combo_tens if d in LOW_SET)
    tens_high   = sum(1 for d in combo_tens if d in HIGH_SET)
    seed_tens_sum = sum(seed_tens) if seed_tens else 0

    prev1 = prev_chain[0] if len(prev_chain) >= 1 else []
    prev2 = prev_chain[1] if len(prev_chain) >= 2 else []

    ctx = {
        # combo + seed
        'combo_tens': combo_tens,
        'seed_tens': seed_tens,
        'tens_sum': tens_sum,
        'seed_tens_sum': seed_tens_sum,

        # counts / ranges
        'tens_even_count': tens_even,
        'tens_odd_count': tens_odd,
        'tens_unique_count': tens_unique,
        'tens_range': tens_range,
        'tens_low_count': tens_low,
        'tens_high_count': tens_high,

        # legacy last2/common
        'prev_seed_tens': prev1,
        'prev_prev_seed_tens': prev2,
        'last2': set(seed_tens) | set(prev1),
        'common_to_both': set(seed_tens) & set(prev1),

        # utilities
        'Counter': Counter,
        'sum_category': sum_category,
        'shared_tens': multiset_shared,

        # hot/cold/due
        'hot_digits': list(hot_digits),
        'cold_digits': list(cold_digits),
        'due_digits': list(due_digits_param) if due_digits_param else [],
    }
    return ctx

def normalize_combo_text(text: str) -> Tuple[List[str], List[str]]:
    raw_tokens = []
    for line in text.splitlines():
        for token in line.replace(',', ' ').split():
            raw_tokens.append(token.strip())
    normalized, invalid = [], []
    for tok in raw_tokens:
        digits = [c for c in tok if c.isdigit()]
        if len(digits) != 5 or any(c not in TENS_DOMAIN for c in digits):
            invalid.append(tok); continue
        normalized.append(''.join(sorted(digits)))
    seen, out = set(), []
    for n in normalized:
        if n not in seen:
            out.append(n); seen.add(n)
    return out, invalid

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Powerball Tens-Only — Manual Filter Runner", layout="wide")

def main():
    st.sidebar.header("🎯 Powerball Tens-Only — Manual Filter Runner")

    # ---------- Run control (prevents auto-run while typing) ----------
    with st.sidebar.expander("⚙️ Run control", expanded=True):
        run_clicked = st.button("▶️ Run / Refresh")
    debug_pipeline = st.sidebar.checkbox("Show debug pipeline (counts & samples)", value=False)

    # ---------- Filter sources ----------
    default_filters_path = "pb_tens_filters_adapted.csv"
    default_extra_path   = "pb_tens_percentile_filters.csv"
    st.sidebar.caption("Filters default to adapted tens-only & optional percentile bands.")
    use_default = st.sidebar.checkbox("Use default adapted filters", value=True)
    uploaded_filters = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])

    filter_paths = []
    if use_default and os.path.exists(default_filters_path): filter_paths.append(default_filters_path)
    if os.path.exists(default_extra_path):                   filter_paths.append(default_extra_path)
    if uploaded_filters is not None:
        upath = "user_filters.csv"
        with open(upath, "wb") as f:
            f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)

    filters = load_filters(filter_paths)

    # ---------- Inputs (seed + 6 optional prev draws; flexible commas) ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Seeds (tens only, 0–6; accepts 11344 or 1,1,3,4,4)")
    seed_raw  = st.sidebar.text_input("Draw 1-back (seed):", placeholder="e.g., 2,3,3,4,5").strip()
    prev1_raw = st.sidebar.text_input("Draw 2-back (optional):", "").strip()
    prev2_raw = st.sidebar.text_input("Draw 3-back (optional):", "").strip()
    prev3_raw = st.sidebar.text_input("Draw 4-back (optional):", "").strip()
    prev4_raw = st.sidebar.text_input("Draw 5-back (optional):", "").strip()
    prev5_raw = st.sidebar.text_input("Draw 6-back (optional):", "").strip()

    seed  = parse_tens_5(seed_raw)
    prevs = [parse_tens_5(x) for x in [prev1_raw, prev2_raw, prev3_raw, prev4_raw, prev5_raw]]

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

    # ---------- Auto Hot/Cold from last K ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hot/Cold (auto)")
    hotcold_k = st.sidebar.slider("Compute from last K draws", min_value=1, max_value=6, value=2, step=1)

    # Build history newest→older (seed first)
    hist_draws = [[int(x) for x in seed]] if seed else []
    for p in prevs:
        if p:
            hist_draws.append([int(x) for x in p])

    hot_auto, cold_auto, df_freq = compute_hot_cold_from_history(hist_draws, min(hotcold_k, len(hist_draws)))

    st.subheader("🔥 Hot / ❄ Cold digits from seed + previous draws")
    st.dataframe(df_freq, use_container_width=True, height=220)
    st.write(f"**Hot (auto):** {hot_auto}   |   **Cold (auto):** {cold_auto}")

    # ---------- Due digits (from the same K-window) ----------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Due digits (auto from last K) with optional manual union")
    manual_due_text = st.sidebar.text_input("Manual due digits (0–6, comma-separated)", value="")
    # Auto due = digits not seen in last K draws
    seen_k = set()
    for draw in hist_draws[:hotcold_k]:
        seen_k.update(draw)
    auto_due = [d for d in range(7) if d not in seen_k]
    manual_due = parse_digit_list(manual_due_text)
    due_digits_current = sorted(set(auto_due) | set(manual_due))
    st.sidebar.write(f"**Current due set:** {{ {', '.join(map(str, due_digits_current))} }}")

    # ---------- Track/Test combos ----------
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area(
        "Track/Test combos (tens as 5 digits; one per line or comma-separated):",
        height=120
    )
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked   = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    # ---------- Guardrails: only proceed on Run ----------
    if not run_clicked:
        st.info("Set sidebar inputs, then click **Run / Refresh** to generate, see filters, and apply them.")
        return

    # ---------- Validate inputs ----------
    if len(seed) != 5:
        st.sidebar.error("Seed tens must include exactly five digits 0–6 (e.g., 23345 or 2,3,3,4,5).")
        st.stop()
    for idx, p in enumerate(prevs, start=1):
        if p and len(p) != 5:
            st.sidebar.error(f"Prev {idx} must be exactly five digits 0–6 or left blank.")
            st.stop()

    # ---------- Generate baseline ----------
    raw_combos, unique_baseline = generate_tens_combinations_both(seed, method)
    zone_filters = [f for f in filters if f.get('is_percentile')]

    def applies(flt, combo_key) -> bool:
        ctx = build_ctx(seed, prevs, combo_key, hot_auto, cold_auto, due_digits_current)
        try:
            return bool(eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx))
        except Exception:
            return False

    def apply_filter_list_raw(pool_list, flist):
        survivors = []
        for combo in pool_list:
            eliminated = False
            for flt in flist:
                if applies(flt, combo):
                    eliminated = True
                    break
            if not eliminated:
                survivors.append(combo)
        return survivors

    # Phase A: percentile filters on RAW (pre-dedup)
    if zone_filters:
        zone_survivors_raw = apply_filter_list_raw(raw_combos, zone_filters)
    else:
        zone_survivors_raw = list(raw_combos)

    # Phase B: Deduplicate
    combos = sorted(set(zone_survivors_raw))

    # ---------- Debug pipeline ----------
    if debug_pipeline:
        st.sidebar.markdown("### 🔍 Debug pipeline")
        st.sidebar.write(f"Raw generated (pre-dedup): **{len(raw_combos)}**")
        st.sidebar.write(f"In-zone survivors pre-dedup: **{len(zone_survivors_raw)}**")
        st.sidebar.write(f"Unique baseline after dedup: **{len(unique_baseline)}**")
        st.sidebar.write(f"In-zone unique after dedup: **{len(combos)}**")

    # ---------- Track/Test normalize ----------
    tracked_norm, invalid_tokens = normalize_combo_text(track_text)
    if invalid_tokens:
        st.sidebar.warning(f"Ignored invalid entries: {', '.join(invalid_tokens[:5])}" + (" ..." if len(invalid_tokens)>5 else ""))
    tracked_set = set(tracked_norm)

    generated_set = set(combos)
    audit = {
        c: {
            "combo": c,
            "generated": (c in generated_set),
            "in_zone": (c in generated_set),
            "preserved": bool(preserve_tracked),
            "injected": False,
            "eliminated": False,
            "eliminated_by": None,
            "eliminated_name": None,
            "eliminated_order": None,
            "would_eliminate_by": None,
            "would_eliminate_name": None,
            "would_eliminate_order": None,
            "stage": ("post-dedup" if c in generated_set else "pre-dedup/percentile-or-not-generated")
        } for c in tracked_norm
    }

    # Optional: inject tracked combos
    if inject_tracked:
        for c in tracked_norm:
            if c not in generated_set:
                combos.append(c); generated_set.add(c)
                if c in audit:
                    audit[c]["injected"] = True
                    audit[c]["generated"] = True
                    audit[c]["in_zone"] = True
                    audit[c]["stage"] = "injected"

    # ---------- Manual filters (non-percentile) ----------
    ui_filters = [f for f in filters if not f.get('is_percentile')]
    # Applicability check (only seed/due/hot/cold context, no per-combo gating)
    base_ctx = build_ctx(seed, prevs, combo_str="", hot_digits=hot_auto, cold_digits=cold_auto, due_digits_param=due_digits_current)
    applicable_filters = []
    for flt in ui_filters:
        try:
            show = bool(eval(flt['applicable_code'], base_ctx, base_ctx))
        except Exception:
            show = True
        if show:
            applicable_filters.append(flt)

    # Initial elimination counts (per filter vs baseline 'combos')
    init_counts = {flt['_ui_key']: 0 for flt in applicable_filters}
    for flt in applicable_filters:
        ic = 0
        for combo in combos:
            if applies(flt, combo):
                ic += 1
        init_counts[flt['_ui_key']] = ic

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Generated (pre-filter):** {len(combos)} combos")

    # Master toggle (default OFF as requested)
    master_toggle = st.sidebar.checkbox("Select/Deselect ALL manual filters", value=False, key="select_all_master")
    hide_zero     = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=False)

    # Sort by aggressiveness (desc), with 0-cuts at bottom
    sorted_filters = sorted(applicable_filters, key=lambda flt: (init_counts[flt['_ui_key']] == 0, -init_counts[flt['_ui_key']]))
    display_filters = [f for f in sorted_filters if init_counts[f['_ui_key']] > 0] if hide_zero else sorted_filters

    st.header("🔧 Manual Filters (tens-only) — ordered by initial cuts (descending)")
    if not display_filters:
        st.info("⚠ No applicable manual filters to display. Check your CSVs, disable 'Hide 0', or confirm inputs.")
    # Apply selected filters
    pool = list(combos)
    order_index = 0
    for flt in display_filters:
        order_index += 1
        cuts = init_counts[flt['_ui_key']]
        label = f"**{flt['id']}** — {flt.get('name','(no name)')}  |  init cuts: **{cuts}**"
        checked = st.checkbox(label, key=f"chk_{flt['_ui_key']}", value=master_toggle)
        if checked:
            survivors_pool = []
            dc = 0
            for combo in pool:
                if applies(flt, combo):
                    # preserve tracked if requested
                    if combo in tracked_set and preserve_tracked:
                        info = audit.get(combo)
                        if info and info.get("would_eliminate_by") is None:
                            info["would_eliminate_by"]   = flt['id']
                            info["would_eliminate_name"] = flt.get('name','')
                            info["would_eliminate_order"]= order_index
                        survivors_pool.append(combo)
                    else:
                        dc += 1
                        if combo in tracked_set:
                            info = audit.get(combo)
                            if info and not info.get("eliminated"):
                                info["eliminated"]        = True
                                info["eliminated_by"]     = flt['id']
                                info["eliminated_name"]   = flt.get('name','')
                                info["eliminated_order"]  = order_index
                else:
                    survivors_pool.append(combo)
            st.caption(f"Applied cuts: **{dc}**")
            pool = survivors_pool

    st.subheader(f"Remaining after manual filters: {len(pool)}")
    survivors_set = set(pool)

    # ---------- Audit table for tracked combos ----------
    if tracked_norm:
        st.markdown("### 🔎 Tracked/Preserved Combos — Audit")
        audit_rows = []
        for c in tracked_norm:
            info = audit.get(c, {})
            audit_rows.append({
                "combo": c,
                "stage": info.get("stage"),
                "generated": info.get("generated", False),
                "in_zone": info.get("in_zone", False),
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
            "combo","stage","generated","in_zone","survived",
            "eliminated","eliminated_by","eliminated_order","eliminated_name",
            "would_eliminate_by","would_eliminate_order","would_eliminate_name",
            "injected","preserved"
        ])
        st.dataframe(df_audit, use_container_width=True)
        st.download_button(
            "Download audit (CSV)",
            df_audit.to_csv(index=False).encode("utf-8"),
            file_name="pb_tens_audit_tracked.csv",
            mime="text/csv"
        )

    # ---------- Survivors ----------
    st.markdown("### ✅ Survivors")
    with st.expander("Show remaining combinations"):
        tracked_survivors = [c for c in pool if c in tracked_set]
        if tracked_survivors:
            st.write("**Tracked survivors:**")
            for c in tracked_survivors:
                info = audit.get(c, {})
                if info and info.get("would_eliminate_by"):
                    st.write(f"{c}  —  ⚠ would be eliminated by {info['would_eliminate_by']} at step {info.get('would_eliminate_order')} ({info.get('would_eliminate_name')}) — preserved")
                else:
                    st.write(c)
            st.write("---")
        # non-tracked
        for c in pool:
            if c not in tracked_set:
                st.write(c)

    # ---------- Downloads ----------
    df_out = pd.DataFrame({"tens_combo": pool})
    st.download_button(
        "Download survivors (CSV)",
        df_out.to_csv(index=False).encode("utf-8"),
        file_name="pb_tens_survivors.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download survivors (TXT)",
        "\n".join(pool).encode("utf-8"),
        file_name="pb_tens_survivors.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
