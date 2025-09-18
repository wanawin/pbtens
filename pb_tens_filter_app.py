# pwrbll_manual_filter_app.py
import streamlit as st
from itertools import product
from collections import Counter
import csv, os
import pandas as pd

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = '0123456'
LOW_SET = {0, 1, 2, 3, 4}
HIGH_SET = {5, 6}

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
# Filter loading
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
                for key in ('name', 'applicable_if', 'expression', 'layman_explanation', 'stat'):
                    if key in row and isinstance(row[key], str):
                        row[key] = row[key].strip().strip('"').strip("'")
                row['expression'] = (row.get('expression') or 'False').replace('!==', '!=')
                expr = row['expression']
                applicable = row.get('applicable_if') or 'True'
                try:
                    row['applicable_code'] = compile(applicable, '<applicable>', 'eval')
                    row['expr_code'] = compile(expr, '<expr>', 'eval')
                except SyntaxError as e:
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                filters.append(row)
    return filters

# ========================
# Tens combo generation
# ========================
def generate_tens_combinations(seed_tens: str, method: str):
    seed_tens = ''.join(sorted(seed_tens))
    combos = set()
    if method == '1-digit':
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                combos.add(''.join(sorted(d + ''.join(p))))
    else:
        pairs = {''.join(sorted((seed_tens[i], seed_tens[j])))
                 for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                combos.add(''.join(sorted(pair + ''.join(p))))
    return sorted(combos)

# ========================
# Context builder
# ========================
def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def build_ctx(seed_tens_str, prev_tens_str, prev_prev_tens_str, combo_str,
              hot_input, cold_input, due_digits):
    seed_tens = [int(x) for x in seed_tens_str]
    prev_tens = [int(x) for x in prev_tens_str] if prev_tens_str else []
    prev_prev_tens = [int(x) for x in prev_prev_tens_str] if prev_prev_tens_str else []
    combo_tens = [int(c) for c in combo_str]

    return {
        'combo_tens': combo_tens,
        'seed_tens': seed_tens,
        'prev_seed_tens': prev_tens,
        'prev_prev_seed_tens': prev_prev_tens,
        'tens_sum': sum(combo_tens),
        'seed_tens_sum': sum(seed_tens),
        'tens_even_count': sum(1 for d in combo_tens if d % 2 == 0),
        'tens_odd_count': sum(1 for d in combo_tens if d % 2 != 0),
        'tens_unique_count': len(set(combo_tens)),
        'tens_range': max(combo_tens) - min(combo_tens),
        'tens_low_count': sum(1 for d in combo_tens if d in LOW_SET),
        'tens_high_count': sum(1 for d in combo_tens if d in HIGH_SET),
        'hot_digits': [int(x) for x in hot_input.split(',') if x.strip().isdigit()],
        'cold_digits': [int(x) for x in cold_input.split(',') if x.strip().isdigit()],
        'due_digits': list(due_digits) if due_digits else [],
        'sum_category': sum_category,
        'shared_tens': multiset_shared,
        'Counter': Counter,
    }

# ========================
# Streamlit UI
# ========================
def main():
    st.sidebar.header("🎯 Tens-Only Manual Filter Runner")

    # Filter sources
    uploaded = st.sidebar.file_uploader("Upload filters CSV", type=["csv"])
    if uploaded:
        tmp = "user_filters.csv"
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())
        filters = load_filters(tmp)
    elif os.path.exists("filters_tens.csv"):
        filters = load_filters("filters_tens.csv")
    else:
        filters = []

    # Seeds
    seed = st.sidebar.text_input("Seed tens (5 digits 0–6):", "").strip()
    prev_seed = st.sidebar.text_input("Prev tens (optional):", "").strip()
    prev_prev = st.sidebar.text_input("Prev-prev tens (optional):", "").strip()
    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot tens digits:", "")
    cold_input = st.sidebar.text_input("Cold tens digits:", "")
    due_input = st.sidebar.text_input("Due digits:", "")
    due_digits = [int(x) for x in due_input.split(",") if x.strip().isdigit()]

    if len(seed) != 5 or any(c not in TENS_DOMAIN for c in seed):
        st.sidebar.error("Seed must be exactly 5 digits 0–6.")
        return

    combos = generate_tens_combinations(seed, method)

    # Initial elimination counts
    init_counts = {}
    for flt in filters:
        count = 0
        for combo in combos:
            ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits)
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                    count += 1
            except Exception:
                pass
        init_counts[flt['id']] = count

    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 cuts", value=True)

    sorted_filters = sorted(filters, key=lambda f: -init_counts.get(f['id'], 0))
    display_filters = [f for f in sorted_filters if init_counts.get(f['id'], 0) > 0] if hide_zero else sorted_filters

    st.header("🔧 Manual Filters")
    pool = list(combos)
    for flt in display_filters:
        fid = flt['id']
        layman = flt.get('layman_explanation', flt.get('name', ''))
        stat = flt.get('stat', '')
        cuts = init_counts.get(fid, 0)
        checked = st.checkbox(f"{fid} — {layman} — stat: {stat} — cuts: {cuts}", value=select_all)
        if checked:
            survivors = []
            for combo in pool:
                ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits)
                try:
                    eliminate = eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx)
                except Exception:
                    eliminate = False
                if not eliminate:
                    survivors.append(combo)
            pool = survivors

    st.subheader(f"Remaining after filters: {len(pool)}")
    with st.expander("Show survivors"):
        st.write(pool)

    df_out = pd.DataFrame({"combo": pool})
    st.download_button("Download survivors CSV", df_out.to_csv(index=False), "survivors.csv", "text/csv")

if __name__ == "__main__":
    main()
