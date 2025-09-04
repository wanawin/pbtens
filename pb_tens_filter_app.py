
import streamlit as st
from itertools import product
from collections import Counter
import csv, os
import pandas as pd

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = '0123456'  # Powerball main balls have tens digits 0..6 only
LOW_SET = set([0,1,2,3,4])
HIGH_SET = set([5,6])

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
def load_filters(paths):
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
def generate_tens_combinations(seed_tens: str, method: str) -> list:
    seed_tens = ''.join(sorted(seed_tens))
    combos_set = set()

    if method == '1-digit':
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(p)))
                combos_set.add(key)
    else:
        pairs = {''.join(sorted((seed_tens[i], seed_tens[j])))
                 for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(p)))
                combos_set.add(key)

    return sorted(combos_set)


def generate_tens_combinations_both(seed_tens: str, method: str):
    """
    Return (RAW_with_duplicates, UNIQUE_sorted) combo keys as strings of 5 digits (0..6),
    where each key is the sorted multiset of tens digits, e.g. '00123'.
    """
    seed_tens = ''.join(sorted(seed_tens))
    raw = []
    uniq = set()

    if method == '1-digit':
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(p)))
                raw.append(key)
                uniq.add(key)
    else:
        pairs = {''.join(sorted((seed_tens[i], seed_tens[j])))
                 for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(p)))
                raw.append(key)
                uniq.add(key)
    return raw, sorted(uniq)

# ========================
# Context builder (tens)
# ========================
def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def build_ctx(seed_tens_str: str,
              prev_tens_str: str,
              prev_prev_tens_str: str,
              combo_str: str,
              hot_input: str,
              cold_input: str,
              due_digits_param: list):
    seed_tens = [int(x) for x in seed_tens_str]
    prev_tens = [int(x) for x in prev_tens_str] if prev_tens_str else []
    prev_prev_tens = [int(x) for x in prev_prev_tens_str] if prev_prev_tens_str else []
    combo_tens = [int(c) for c in combo_str]

    tens_sum = sum(combo_tens)
    tens_even = sum(1 for d in combo_tens if d % 2 == 0)
    tens_odd = 5 - tens_even
    tens_unique = len(set(combo_tens))
    tens_range = max(combo_tens) - min(combo_tens)
    tens_low = sum(1 for d in combo_tens if d in LOW_SET)
    tens_high = sum(1 for d in combo_tens if d in HIGH_SET)

    hot_digits = [int(x) for x in hot_input.split(',') if x.strip().isdigit() and int(x) in range(7)]
    cold_digits = [int(x) for x in cold_input.split(',') if x.strip().isdigit() and int(x) in range(7)]

    due_digits = list(due_digits_param) if due_digits_param is not None else []

    seed_tens_sum = sum(seed_tens) if seed_tens else 0

    ctx = {
        'combo_tens': combo_tens,
        'seed_tens': seed_tens,
        'prev_seed_tens': prev_tens,
        'prev_prev_seed_tens': prev_prev_tens,
        'tens_sum': tens_sum,
        'seed_tens_sum': seed_tens_sum,
        'tens_even_count': tens_even,
        'tens_odd_count': tens_odd,
        'tens_unique_count': tens_unique,
        'tens_range': tens_range,
        'tens_low_count': tens_low,
        'tens_high_count': tens_high,
        'last2': set(seed_tens) | set(prev_tens),
        'common_to_both': set(seed_tens) & set(prev_tens),
        'Counter': Counter,
        'sum_category': sum_category,
        'shared_tens': multiset_shared,
        'hot_digits': hot_digits,
        'cold_digits': cold_digits,
        'due_digits': due_digits,
    }
    return ctx

def normalize_combo_text(text: str):
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
def main():
    st.sidebar.header("🎯 Powerball Tens-Only — Manual Filter Runner")

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

    # --- Seed inputs
    seed = st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):", placeholder="e.g., 23345").strip()
    prev_seed = st.sidebar.text_input("Prev tens (Draw 2-back, 5 digits 0–6, optional):").strip()
    prev_prev = st.sidebar.text_input("Prev-prev tens (Draw 3-back, 5 digits 0–6, optional):").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot tens digits (comma-separated 0–6, optional):").strip()
    cold_input = st.sidebar.text_input("Cold tens digits (comma-separated 0–6, optional):").strip()

    # --- Due digits controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Due digits (tens)")
    m = st.sidebar.slider("Auto window m (use last m tens seeds)", min_value=1, max_value=3, value=2, step=1)
    due_mode = st.sidebar.radio("Due source", ["Auto (from last m)", "Manual override", "Auto + manual (union)"], index=0)
    manual_due_text = st.sidebar.text_input("Manual due digits (0–6, comma-separated)", value="")
    disable_due_filters_when_empty = st.sidebar.checkbox("Disable due-based filters when due set is empty", value=True)

    def digits_from_str(s): return [int(x) for x in s] if s else []
    seeds_chain = [seed, prev_seed, prev_prev]
    seen = set(); used = 0
    for s in seeds_chain:
        if s and used < m:
            seen.update(digits_from_str(s)); used += 1
    auto_due = [d for d in range(7) if d not in seen]

    manual_due = []
    if manual_due_text.strip():
        for tok in manual_due_text.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 0 <= v <= 6:
                    manual_due.append(v)

    if due_mode == "Auto (from last m)":
        due_digits_current = auto_due
    elif due_mode == "Manual override":
        due_digits_current = manual_due
    else:
        due_digits_current = sorted(set(auto_due) | set(manual_due))

    st.sidebar.write(f"**Current due set:** {{ {', '.join(map(str, due_digits_current))} }}")

    # --- Track/test combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test combos (tens as 5 digits, e.g., 00123, 23345; one per line or comma-separated):", height=120)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    # Input validation
    if len(seed) != 5 or (not seed.isdigit()) or any(c not in TENS_DOMAIN for c in seed):
        st.sidebar.error("Seed tens must be exactly 5 digits in 0–6 (e.g., 23345)."); return
    if prev_seed and (len(prev_seed) != 5 or (not prev_seed.isdigit()) or any(c not in TENS_DOMAIN for c in prev_seed)):
        st.sidebar.error("Prev tens must be 5 digits in 0–6 or left blank."); return
    if prev_prev and (len(prev_prev) != 5 or (not prev_prev.isdigit()) or any(c not in TENS_DOMAIN for c in prev_prev)):
        st.sidebar.error("Prev-prev tens must be 5 digits in 0–6 or left blank."); return



    # Generate pool — RAW (with duplicates) and UNIQUE baseline
    raw_combos, unique_baseline = generate_tens_combinations_both(seed, method)

    # Identify percentile/zone filters by source filename (auto-applied pre-dedup)
    zone_filters = [f for f in filters if f.get('is_percentile')]

    def apply_filter_list_raw(pool_list, flist):
        survivors = []
        for combo in pool_list:
            ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
            eliminate = False
            for flt in flist:
                try:
                    if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                        eliminate = True; break
                except Exception:
                    pass
            if not eliminate:
                survivors.append(combo)
        return survivors

    # Phase A: Apply percentile filters BEFORE dedup (keeps only in-zone combos)
    if zone_filters:
        zone_survivors_raw = apply_filter_list_raw(raw_combos, zone_filters)
    else:
        zone_survivors_raw = list(raw_combos)

    # Phase B: Deduplicate
    combos = sorted(set(zone_survivors_raw))

    # Metrics — compare to full unique enumeration after dedup
    st.sidebar.markdown("""---
**Percentile pipeline**
- Raw generated (pre-dedup): **{}**
- In-zone survivors pre-dedup: **{}**
- Unique baseline after dedup: **{}**
- In-zone unique after dedup: **{}** ({}/{})""".format(
        len(raw_combos), len(zone_survivors_raw), len(unique_baseline), len(combos),
        len(combos), len(unique_baseline) if len(unique_baseline) else 1
    ))

    # Only show non-zone filters in the manual list
    ui_filters = [f for f in filters if not f.get('is_percentile')]

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
            "preserved": bool(preserve_tracked),
            "injected": False,
            "eliminated": False,
            "eliminated_by": None,
            "eliminated_name": None,
            "eliminated_order": None,
            "would_eliminate_by": None,
            "would_eliminate_name": None,
            "would_eliminate_order": None,
        } for c in tracked_norm
    }

    if inject_tracked:
        for c in tracked_norm:
            if c not in generated_set:
                combos.append(c); generated_set.add(c)
                if c in audit:
                    audit[c]["injected"] = True

    # Initial elimination counts
    init_counts = {flt['id']: 0 for flt in ui_filters}
    for flt in ui_filters:
        # Skip due-based filters if requested and due set empty
        if disable_due_filters_when_empty and not due_digits_current and 'due_digits' in flt.get('expr_str',''):
            init_counts[flt['id']] = 0; continue
        ic = 0
        for combo in combos:
            ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                    ic += 1
            except Exception:
                pass
        init_counts[flt['id']] = ic

    st.sidebar.markdown(f"**Generated (pre-filter):** {len(combos)} combos")
    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=True)

    sorted_filters = sorted(ui_filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))
    display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

    # Apply selected filters
    pool = list(combos)
    dynamic_counts = {}
    st.header("🔧 Manual Filters (tens-only)")
    order_index = 0
    for flt in display_filters:
        order_index += 1
        key = f"filter_{flt['id']}"
        default_checked = select_all and flt['enabled_default']
        checked = st.checkbox(f"{flt['id']}: {flt['name']} — init cuts {init_counts[flt['id']]}",
                              key=key, value=default_checked)
        if checked:
            # Skip due-based filter if requested and there are no due digits
            if disable_due_filters_when_empty and not due_digits_current and 'due_digits' in flt.get('expr_str',''):
                dynamic_counts[flt['id']] = 0
                continue
            survivors_pool = []
            dc = 0
            for combo in pool:
                ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
                eliminate = False
                try:
                    eliminate = eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx)
                except Exception:
                    eliminate = False

                is_tracked = combo in tracked_set
                if eliminate:
                    if is_tracked and preserve_tracked:
                        if audit.get(combo) and audit[combo]["would_eliminate_by"] is None:
                            audit[combo]["would_eliminate_by"] = flt['id']
                            audit[combo]["would_eliminate_name"] = flt.get('name','')
                            audit[combo]["would_eliminate_order"] = order_index
                        survivors_pool.append(combo)
                        continue
                    dc += 1
                    if is_tracked and not audit[combo]["eliminated"]:
                        audit[combo]["eliminated"] = True
                        audit[combo]["eliminated_by"] = flt['id']
                        audit[combo]["eliminated_name"] = flt.get('name','')
                        audit[combo]["eliminated_order"] = order_index
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

if __name__ == "__main__":
    main()
