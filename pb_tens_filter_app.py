# pb_tens_filter_app.py — Powerball Tens-Only Manual Filter Runner (with debug, master toggle, and auto hot/cold)

import os, csv
from itertools import product
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = '0123456'  # Powerball main balls have tens digits 0..6 only
LOW_SET = set([0, 1, 2, 3, 4])
HIGH_SET = set([5, 6])

def sum_category(total: int) -> str:
    if 0 <= total <= 10:
        return 'Very Low'
    elif 11 <= total <= 13:
        return 'Low'
    elif 14 <= total <= 17:
        return 'Mid'
    else:
        return 'High'

# ========================
# Filter loading (15-col)
# ========================
def load_filters(paths) -> List[Dict]:
    """Load filters from one or more CSVs (adapted 15-col format).
    - Uses 'expression' and 'applicable_if' (compiled).
    - Marks percentile filters by filename so we can run them pre-dedup."""
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
                row['id'] = row.get('id', row.get('fid', '')).strip()
                for key in ('name', 'applicable_if', 'expression'):
                    if key in row and isinstance(row[key], str):
                        row[key] = row[key].strip().strip('"').strip("'")
                # normalize a few glitches
                row['expression'] = (row.get('expression') or 'False').replace('!==', '!=')
                row['expr_str'] = row['expression']
                applicable = row.get('applicable_if') or 'True'
                expr = row.get('expression') or 'False'
                try:
                    row['applicable_code'] = compile(applicable, '<applicable>', 'eval')
                    row['expr_code'] = compile(expr, '<expr>', 'eval')
                except SyntaxError as e:
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                row['enabled_default'] = (row.get('enabled','').lower() == 'true')
                row['source_path'] = os.path.basename(path)
                row['is_percentile'] = (row['source_path'] == 'pb_tens_percentile_filters.csv')
                filters.append(row)
    return filters

# ========================
# Tens combo generation
# ========================
def generate_tens_combinations_both(seed_tens: str, method: str) -> Tuple[List[str], List[str]]:
    """
    Return (RAW_with_duplicates, UNIQUE_sorted) combo keys as strings of 5 digits (0..6),
    where each key is the sorted multiset of tens digits, e.g. '00123'.
    1-digit: choose 1 from seed, + any 4 from domain
    2-digit pair: choose any pair from seed, + any 3 from domain
    """
    seed_tens = ''.join(sorted(seed_tens))
    raw = []
    uniq = set()

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
# Context helpers (tens)
# ========================
def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def parse_tens_digits(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    if not s.isdigit() or len(s) != 5 or any(c not in TENS_DOMAIN for c in s):
        return []
    return [int(x) for x in s]

def compute_hot_cold_from_history(history: List[List[int]], k: int) -> Tuple[List[int], List[int]]:
    """Hot = most frequent; Cold = least frequent across last k draws in 'history' (each a list of 5 tens digits)."""
    if not history or k <= 0:
        return [], []
    window = history[:k]  # history is already ordered newest→older, we’ll pass reversed if needed
    cnt = Counter()
    for draw in window:
        cnt.update(draw)
    if not cnt:
        return [], []
    maxf, minf = max(cnt.values()), min(cnt.values())
    hot = sorted([d for d, c in cnt.items() if c == maxf])
    cold = sorted([d for d, c in cnt.items() if c == minf])
    return hot, cold

def build_ctx(seed_tens_str: str,
              prev_tens_strs: List[str],
              combo_str: str,
              hot_digits: List[int],
              cold_digits: List[int],
              due_digits_param: List[int]):
    seed_tens = parse_tens_digits(seed_tens_str)
    prev_chain = [parse_tens_digits(s) for s in prev_tens_strs if s]
    combo_tens = [int(c) for c in combo_str]

    tens_sum = sum(combo_tens)
    tens_even = sum(1 for d in combo_tens if d % 2 == 0)
    tens_odd = 5 - tens_even
    tens_unique = len(set(combo_tens))
    tens_range = max(combo_tens) - min(combo_tens)
    tens_low = sum(1 for d in combo_tens if d in LOW_SET)
    tens_high = sum(1 for d in combo_tens if d in HIGH_SET)

    seed_tens_sum = sum(seed_tens) if seed_tens else 0

    # sets from the last two draws (legacy vars):
    prev = prev_chain[0] if len(prev_chain) >= 1 else []
    prev_prev = prev_chain[1] if len(prev_chain) >= 2 else []

    ctx = {
        'combo_tens': combo_tens,
        'seed_tens': seed_tens,
        'prev_seed_tens': prev,
        'prev_prev_seed_tens': prev_prev,
        'tens_sum': tens_sum,
        'seed_tens_sum': seed_tens_sum,
        'tens_even_count': tens_even,
        'tens_odd_count': tens_odd,
        'tens_unique_count': tens_unique,
        'tens_range': tens_range,
        'tens_low_count': tens_low,
        'tens_high_count': tens_high,
        'last2': set(seed_tens) | set(prev),
        'common_to_both': set(seed_tens) & set(prev),
        'Counter': Counter,
        'sum_category': sum_category,
        'shared_tens': multiset_shared,
        'hot_digits': hot_digits,
        'cold_digits': cold_digits,
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

    # --- Run gate (prevents 'auto run while typing')
    with st.sidebar.expander("⚙️ Run control", expanded=True):
        run_btn = st.button("▶️ Run / Refresh")

    # --- Debug pipeline toggle
    debug_pipeline = st.sidebar.checkbox("Show debug pipeline (counts & samples)", value=False)

    # --- Filter sources
    default_filters_path = "pb_tens_filters_adapted.csv"
    default_extra_path = "pb_tens_percentile_filters.csv"
    st.sidebar.caption("Filters default to adapted tens-only & optional percentile bands.")
    use_default = st.sidebar.checkbox("Use default adapted filters", value=True)
    uploaded_filters = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])

    filter_paths = []
    if use_default and os.path.exists(default_filters_path):
        filter_paths.append(default_filters_path)
    if os.path.exists(default_extra_path):
        filter_paths.append(default_extra_path)
    if uploaded_filters is not None:
        upath = "user_filters.csv"
        with open(upath, "wb") as f:
            f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)

    filters = load_filters(filter_paths)

    # --- Seed inputs (now 6 prior draws total)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Seeds (tens only, 5 digits 0–6)")
    seed      = st.sidebar.text_input("Draw 1-back (seed):", placeholder="e.g., 23345").strip()
    prev1     = st.sidebar.text_input("Draw 2-back (optional):").strip()
    prev2     = st.sidebar.text_input("Draw 3-back (optional):").strip()
    prev3     = st.sidebar.text_input("Draw 4-back (optional):").strip()
    prev4     = st.sidebar.text_input("Draw 5-back (optional):").strip()
    prev5     = st.sidebar.text_input("Draw 6-back (optional):").strip()
    prev_chain = [prev1, prev2, prev3, prev4, prev5]

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

    # --- Hot/Cold options (auto from last K)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hot/Cold (auto)")
    hotcold_k = st.sidebar.slider("Compute from last K draws", min_value=1, max_value=6, value=2, step=1)
    # Build a history list newest→older using the inputs we have
    hist_draws = [parse_tens_digits(seed)] + [parse_tens_digits(x) for x in prev_chain if x]
    hot_auto, cold_auto = compute_hot_cold_from_history(hist_draws, min(hotcold_k, len(hist_draws)))
    st.sidebar.caption(f"Auto Hot: {hot_auto} | Auto Cold: {cold_auto}")

    # Manual add-ons for hot/cold if you want
    hot_manual_str  = st.sidebar.text_input("Manual HOT add-ons (comma digits 0–6)", value="")
    cold_manual_str = st.sidebar.text_input("Manual COLD add-ons (comma digits 0–6)", value="")

    def parse_digit_list(s: str) -> List[int]:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 0 <= v <= 6:
                    out.append(v)
        return out

    hot_digits = sorted(set(hot_auto)  | set(parse_digit_list(hot_manual_str)))
    cold_digits = sorted(set(cold_auto) | set(parse_digit_list(cold_manual_str)))

    # --- Due digits controls (reuse prior UX)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Due digits (tens)")
    due_mode = st.sidebar.radio("Due source", ["Auto (from last K)", "Manual override", "Auto + manual (union)"], index=0)
    manual_due_text = st.sidebar.text_input("Manual due digits (0–6, comma-separated)", value="")
    # Auto due: digits not seen in last K draws
    seen = set()
    for draw in hist_draws[:hotcold_k]:
        seen.update(draw)
    auto_due = [d for d in range(7) if d not in seen]
    manual_due = parse_digit_list(manual_due_text)

    if due_mode == "Auto (from last K)":
        due_digits_current = auto_due
    elif due_mode == "Manual override":
        due_digits_current = manual_due
    else:
        due_digits_current = sorted(set(auto_due) | set(manual_due))

    st.sidebar.write(f"**Current due set:** {{ {', '.join(map(str, due_digits_current))} }}")

    # --- Track/test combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area(
        "Track/Test combos (tens as 5 digits, e.g., 00123, 23345; one per line or comma-separated):",
        height=120
    )
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    # --- Early validation (but only error on click Run)
    def validate_tens(s: str) -> bool:
        return bool(s) and s.isdigit() and len(s) == 5 and all(c in TENS_DOMAIN for c in s)

    if run_btn:
        if not validate_tens(seed):
            st.sidebar.error("Seed tens must be exactly 5 digits in 0–6 (e.g., 23345).")
            st.stop()
        for label, s in [("Draw 2-back", prev1), ("Draw 3-back", prev2), ("Draw 4-back", prev3), ("Draw 5-back", prev4), ("Draw 6-back", prev5)]:
            if s and not validate_tens(s):
                st.sidebar.error(f"{label} must be 5 digits in 0–6 or left blank.")
                st.stop()

        # Generate pool — RAW (with duplicates) and UNIQUE baseline
        raw_combos, unique_baseline = generate_tens_combinations_both(seed, method)

        # Separate percentile filters (pre-dedup)
        zone_filters = [f for f in filters if f.get('is_percentile')]
        def applies(flt, combo_key) -> bool:
            ctx = build_ctx(seed, prev_chain, combo_key, hot_digits, cold_digits, due_digits_current)
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

        # Phase A: percentile filters on RAW
        if zone_filters:
            zone_survivors_raw = apply_filter_list_raw(raw_combos, zone_filters)
        else:
            zone_survivors_raw = list(raw_combos)

        # Phase B: Deduplicate after percentile
        combos = sorted(set(zone_survivors_raw))

        if debug_pipeline:
            st.sidebar.markdown("### 🔍 Debug pipeline")
            st.sidebar.write(f"Raw generated (pre-dedup): **{len(raw_combos)}**")
            st.sidebar.write(f"In-zone survivors pre-dedup: **{len(zone_survivors_raw)}**")
            st.sidebar.write(f"Unique baseline after dedup: **{len(unique_baseline)}**")
            st.sidebar.write(f"In-zone unique after dedup: **{len(combos)}**")

        # Normalize tracked combos
        tracked_norm, invalid_tokens = normalize_combo_text(track_text)
        if invalid_tokens:
            st.sidebar.warning(f"Ignored invalid entries: {', '.join(invalid_tokens[:5])}" + (" ..." if len(invalid_tokens)>5 else ""))
        tracked_set = set(tracked_norm)

        generated_set = set(combos)
        audit = {
            c: {
                "combo": c,
                "generated": (c in generated_set),
                "in_zone": (c in generated_set),  # if not in generated_set, it was likely dropped by percentile or never generated
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

        # Manual filters (non-percentile)
        ui_filters = [f for f in filters if not f.get('is_percentile')]

        # Initial elimination counts (how many each filter would cut if applied to 'combos')
        init_counts = {flt['id']: 0 for flt in ui_filters}
        for flt in ui_filters:
            ic = 0
            for combo in combos:
                if applies(flt, combo):
                    ic += 1
            init_counts[flt['id']] = ic

        # Master toggle + visibility
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Generated (pre-filter):** {len(combos)} combos")
        master_toggle = st.sidebar.checkbox("✅ Select/Deselect ALL manual filters", value=False)
        hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=False)

        # Sort: most aggressive first; keep 0-cuts at the bottom
        sorted_filters = sorted(ui_filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))
        display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

        # Apply selected filters (always visible list)
        pool = list(combos)
        dynamic_counts = {}
        st.header("🔧 Manual Filters (tens-only)")
        order_index = 0
        for flt in display_filters:
            order_index += 1
            label = f"{flt['id']}: {flt.get('name','(unnamed)')} — cuts {init_counts[flt['id']]}"
            key = f"filter_{flt['id']}"
            default_checked = master_toggle or flt['enabled_default']
            checked = st.checkbox(label, key=key, value=default_checked)
            if checked:
                survivors_pool = []
                dc = 0
                for combo in pool:
                    if applies(flt, combo):
                        # preserve tracked if requested
                        if combo in tracked_set and preserve_tracked:
                            info = audit.get(combo)
                            if info and info.get("would_eliminate_by") is None:
                                info["would_eliminate_by"] = flt['id']
                                info["would_eliminate_name"] = flt.get('name','')
                                info["would_eliminate_order"] = order_index
                            survivors_pool.append(combo)
                        else:
                            dc += 1
                            if combo in tracked_set:
                                info = audit.get(combo)
                                if info and not info.get("eliminated"):
                                    info["eliminated"] = True
                                    info["eliminated_by"] = flt['id']
                                    info["eliminated_name"] = flt.get('name','')
                                    info["eliminated_order"] = order_index
                    else:
                        survivors_pool.append(combo)
                dynamic_counts[flt['id']] = dc
                pool = survivors_pool

        st.subheader(f"Remaining after manual filters: {len(pool)}")
        survivors_set = set(pool)

        # --- Audit table
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
            st.download_button("Download audit (CSV)", df_audit.to_csv(index=False), file_name="pb_tens_audit_tracked.csv", mime="text/csv")

        # Survivors
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
            for c in pool:
                if c not in tracked_set:
                    st.write(c)

        # Downloads
        df_out = pd.DataFrame({"tens_combo": pool})
        st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), file_name="pb_tens_survivors.csv", mime="text/csv")
        st.download_button("Download survivors (TXT)", "\n".join(pool), file_name="pb_tens_survivors.txt", mime="text/plain")
    else:
        st.info("Set your inputs in the sidebar, then click **Run / Refresh** to generate, see filters, and apply them.")

if __name__ == "__main__":
    main()
