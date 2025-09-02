
import streamlit as st
from itertools import product, combinations
from collections import Counter
import csv, os

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
                applicable = row.get('applicable_if') or 'True'
                expr = row.get('expression') or 'False'
                try:
                    row['applicable_code'] = compile(applicable, '<applicable>', 'eval')
                    row['expr_code'] = compile(expr, '<expr>', 'eval')
                except SyntaxError as e:
                    # Skip bad filters but show a warning
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                row['enabled_default'] = (row.get('enabled','').lower() == 'true')
                filters.append(row)
    return filters

# ========================
# Tens combo generation
# ========================
def generate_tens_combinations(seed_tens: str, method: str) -> list:
    # Normalize and validate
    seed_tens = ''.join(sorted(seed_tens))
    combos_set = set()

    if method == '1-digit':
        # For each digit from seed, combine with every 4-length product from 0..6
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(p)))
                combos_set.add(key)
    else:
        # "2-digit pair" from seed tens (unordered pairs), + every 3-length product from 0..6
        pairs = {''.join(sorted((seed_tens[i], seed_tens[j])))
                 for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(p)))
                combos_set.add(key)

    return sorted(combos_set)

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
              cold_input: str):
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

    # user-provided hot/cold or blank
    hot_digits = [int(x) for x in hot_input.split(',') if x.strip().isdigit() and int(x) in range(7)]
    cold_digits = [int(x) for x in cold_input.split(',') if x.strip().isdigit() and int(x) in range(7)]
    # due = not seen in last two tens seeds
    seen_last2 = set(prev_tens) | set(seed_tens)
    due_digits = [d for d in range(7) if d not in seen_last2]

    # seed stats
    seed_tens_sum = sum(seed_tens) if seed_tens else 0

    ctx = {
        # tens vectors
        'combo_tens': combo_tens,
        'seed_tens': seed_tens,
        'prev_seed_tens': prev_tens,
        'prev_prev_seed_tens': prev_prev_tens,
        # counts / sums
        'tens_sum': tens_sum,
        'seed_tens_sum': seed_tens_sum,
        'tens_even_count': tens_even,
        'tens_odd_count': tens_odd,
        'tens_unique_count': tens_unique,
        'tens_range': tens_range,
        'tens_low_count': tens_low,
        'tens_high_count': tens_high,
        # sets
        'last2': set(seed_tens) | set(prev_tens),
        'common_to_both': set(seed_tens) & set(prev_tens),
        # helpers
        'Counter': Counter,
        'sum_category': sum_category,
        'shared_tens': multiset_shared,
        # H/C/D
        'hot_digits': hot_digits,
        'cold_digits': cold_digits,
        'due_digits': due_digits,
    }
    return ctx

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
        # Include percentile filters but they can be toggled off in UI
        filter_paths.append(default_extra_path)
    if uploaded_filters is not None:
        # Save uploaded to disk and include
        upath = "user_filters.csv"
        with open(upath, "wb") as f:
            f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)

    filters = load_filters(filter_paths)

    # --- Seed inputs (tens-only, 5 digits from 0..6)
    seed = st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):", placeholder="e.g., 23345").strip()
    prev_seed = st.sidebar.text_input("Prev tens (Draw 2-back, 5 digits 0–6, optional):").strip()
    prev_prev = st.sidebar.text_input("Prev-prev tens (Draw 3-back, 5 digits 0–6, optional):").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot tens digits (comma-separated 0–6, optional):").strip()
    cold_input = st.sidebar.text_input("Cold tens digits (comma-separated 0–6, optional):").strip()

    if len(seed) != 5 or (not seed.isdigit()) or any(c not in TENS_DOMAIN for c in seed):
        st.sidebar.error("Seed tens must be exactly 5 digits in 0–6 (e.g., 23345).")
        return
    if prev_seed and (len(prev_seed) != 5 or (not prev_seed.isdigit()) or any(c not in TENS_DOMAIN for c in prev_seed)):
        st.sidebar.error("Prev tens must be 5 digits in 0–6 or left blank.")
        return
    if prev_prev and (len(prev_prev) != 5 or (not prev_prev.isdigit()) or any(c not in TENS_DOMAIN for c in prev_prev)):
        st.sidebar.error("Prev-prev tens must be 5 digits in 0–6 or left blank.")
        return

    # --- Generate pool (no auto filters — manual phase starts after generation)
    combos = generate_tens_combinations(seed, method)

    # --- Precompute initial elimination counts for display ordering
    init_counts = {flt['id']: 0 for flt in filters}
    for flt in filters:
        ic = 0
        for combo in combos:
            ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input)
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                    ic += 1
            except Exception:
                pass
        init_counts[flt['id']] = ic

    # --- UI Controls
    st.sidebar.markdown(f"**Generated (pre-filter):** {len(combos)} combos")
    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=True)

    # Sort filters by initial elimination count (desc), but push zeros to bottom
    sorted_filters = sorted(filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))

    display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

    # --- Apply selected filters sequentially
    pool = list(combos)
    dynamic_counts = {}
    st.header("🔧 Manual Filters (tens-only)")
    for flt in display_filters:
        key = f"filter_{flt['id']}"
        default_checked = select_all and flt['enabled_default']
        checked = st.checkbox(f"{flt['id']}: {flt['name']} — init cuts {init_counts[flt['id']]}",
                              key=key, value=default_checked)
        if checked:
            survivors_pool = []
            dc = 0
            for combo in pool:
                ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input)
                try:
                    if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                        dc += 1
                    else:
                        survivors_pool.append(combo)
                except Exception:
                    survivors_pool.append(combo)
            dynamic_counts[flt['id']] = dc
            pool = survivors_pool

    st.subheader(f"Remaining after manual filters: {len(pool)}")

    # Show survivors
    with st.expander("Show remaining combinations"):
        for c in pool:
            st.write(c)

    # Download buttons
    import pandas as pd
    df_out = pd.DataFrame({"tens_combo": pool})
    st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), file_name="pb_tens_survivors.csv", mime="text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(pool), file_name="pb_tens_survivors.txt", mime="text/plain")

if __name__ == "__main__":
    main()
