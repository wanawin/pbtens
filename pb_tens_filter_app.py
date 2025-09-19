import streamlit as st
from itertools import product
from collections import Counter
import csv, os
import pandas as pd

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = '0123456'
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
# Filter loading (with historical stat)
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
                row['historical'] = row.get('stat', '').strip()  # capture stat column
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
# Context builder
# ========================
def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def build_ctx(seed_tens_str, prev_tens_str, prev_prev_tens_str,
              combo_str, hot_input, cold_input, due_digits_param):
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
        'variant_name': "tens"
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
# Streamlit App
# ========================
def main():
    st.sidebar.header("🎯 Powerball Tens-Only — Manual Filter Runner")

    # --- Filter sources
    default_filters_path = "filters_tens.csv"
    use_default = st.sidebar.checkbox("Use default filters_tens.csv", value=True)
    uploaded_filters = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])

    filter_paths = []
    if use_default and os.path.exists(default_filters_path):
        filter_paths.append(default_filters_path)
    if uploaded_filters is not None:
        upath = "user_filters.csv"
        with open(upath, "wb") as f:
            f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)

    filters = load_filters(filter_paths)

    # --- Seed inputs
    seed = st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):", placeholder="23345").strip()
    prev_seed = st.sidebar.text_input("Prev tens (Draw 2-back, optional):").strip()
    prev_prev = st.sidebar.text_input("Prev-prev tens (Draw 3-back, optional):").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot tens digits (comma-separated 0–6, optional):").strip()
    cold_input = st.sidebar.text_input("Cold tens digits (comma-separated 0–6, optional):").strip()

    # --- Due digits
    st.sidebar.markdown("---")
    st.sidebar.subheader("Due digits (tens)")
    m = st.sidebar.slider("Auto window m (last m seeds)", min_value=1, max_value=6, value=2, step=1)
    due_mode = st.sidebar.radio("Due source", ["Auto", "Manual", "Auto+Manual"], index=0)
    manual_due_text = st.sidebar.text_input("Manual due digits (0–6, comma-separated)", value="")

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

    if due_mode == "Auto":
        due_digits_current = auto_due
    elif due_mode == "Manual":
        due_digits_current = manual_due
    else:
        due_digits_current = sorted(set(auto_due) | set(manual_due))

    st.sidebar.write(f"**Current due set:** {due_digits_current}")

    # --- Track combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test combos (5 digits, one per line or comma-separated):", height=120)
    tracked_norm, invalid_tokens = normalize_combo_text(track_text)
    tracked_set = set(tracked_norm)

    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    # --- Validate input
    if len(seed) != 5 or (not seed.isdigit()) or any(c not in TENS_DOMAIN for c in seed):
        st.sidebar.error("Seed tens must be 5 digits in 0–6."); return

    # --- Generate combos
    raw_combos, unique_baseline = generate_tens_combinations_both(seed, method)
    combos = sorted(set(raw_combos))

    st.sidebar.markdown(f"**Generated combos:** {len(combos)}")

    # --- Initial elimination counts
    init_counts = {flt['id']: 0 for flt in filters}
    for flt in filters:
        ic = 0
        for combo in combos:
            ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                    ic += 1
            except Exception:
                pass
        init_counts[flt['id']] = ic

    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 eliminations", value=True)

    sorted_filters = sorted(filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))
    display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

    # --- Apply filters interactively
    st.header("🔧 Manual Filters")
    survivors = list(combos)
    dynamic_counts = {}

    for flt in display_filters:
        key = f"filter_{flt['id']}"
        ic = init_counts[flt['id']]
        checked = st.checkbox(
            f"{flt['id']}: {flt.get('name','')} | Hist: {flt.get('historical','?')} | Cuts now: {ic}",
            key=key, value=(select_all and flt['enabled_default'])
        )
        if checked:
            survivors_pool = []
            dc = 0
            for combo in survivors:
                ctx = build_ctx(seed, prev_seed, prev_prev, combo, hot_input, cold_input, due_digits_current)
                eliminate = False
                try:
                    eliminate = eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx)
                except Exception:
                    eliminate = False
                if eliminate:
                    dc += 1
                else:
                    survivors_pool.append(combo)
            dynamic_counts[flt['id']] = dc
            survivors = survivors_pool

    st.subheader(f"Remaining after filters: {len(survivors)}")
    with st.expander("Show survivors"):
        for c in survivors:
            st.write(c)

    # --- Download
    df_out = pd.DataFrame({"tens_combo": survivors})
    st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), file_name="pb_tens_survivors.csv", mime="text/csv")

if __name__ == "__main__":
    main()
