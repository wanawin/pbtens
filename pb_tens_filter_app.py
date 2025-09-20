import streamlit as st
import pandas as pd
import csv
import os
from itertools import product
from collections import Counter

# -------------------------------
# Helpers
# -------------------------------
def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    elif 16 <= total <= 24:
        return 'Low'
    elif 25 <= total <= 33:
        return 'Mid'
    else:
        return 'High'

def load_filters(path: str = "pb_tens_filters_adapted.csv") -> list:
    """Load and compile filters from CSV (no variant logic)."""
    if not os.path.exists(path):
        st.error(f"Filter file not found: {path}")
        st.stop()
    filters = []
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k.lower(): v for k, v in raw.items()}
            row['id'] = row.get('id', '').strip()
            row['layman'] = row.get('layman', '').strip()
            row['stat'] = row.get('stat', '').strip()
            expr = row.get('expression', '').strip()
            if not expr:
                continue
            try:
                row['expr_code'] = compile(expr, '<expr>', 'eval')
            except SyntaxError as e:
                st.error(f"Syntax error in filter {row['id']}: {e}")
                continue
            row['enabled_default'] = False
            filters.append(row)
    return filters

def generate_combinations(seed: str, method: str) -> list:
    all_digits = '0123456'  # only 0–6 for tens app
    combos_set = set()
    sorted_seed = ''.join(sorted(seed))
    if method == "1-digit":
        for d in sorted_seed:
            for p in product(all_digits, repeat=4):
                combos_set.add(''.join(sorted(d + ''.join(p))))
    else:  # 2-digit pair
        pairs = {''.join(sorted((sorted_seed[i], sorted_seed[j])))
                 for i in range(len(sorted_seed)) for j in range(i+1, len(sorted_seed))}
        for pair in pairs:
            for p in product(all_digits, repeat=3):
                combos_set.add(''.join(sorted(pair + ''.join(p))))
    return sorted(combos_set)

def compute_hot_cold(prev_draws: list):
    """Auto-calc hot/cold digits if 6 prev draws are provided."""
    if len(prev_draws) < 6:
        return [], []
    digits = [int(d) for draw in prev_draws for d in draw if d.isdigit()]
    counts = Counter(digits)
    hot = [d for d, _ in counts.most_common(3)]
    cold = [d for d, _ in counts.most_common()[-3:]]
    return hot, cold

# -------------------------------
# Main app
# -------------------------------
def main():
    st.sidebar.header("🔢 Powerball Tens Filter App")

    # Inputs
    seed = st.sidebar.text_input("Draw 1-back (required):").strip()
    prev_draws = [st.sidebar.text_input(f"Draw {i+2}-back (optional):").strip() for i in range(6-1)]  # 5 more inputs
    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
    hot_input = st.sidebar.text_input("Hot digits (comma-separated, overrides auto):").strip()
    cold_input = st.sidebar.text_input("Cold digits (comma-separated, overrides auto):").strip()
    check_combo = st.sidebar.text_input("Track specific combo:").strip()
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=True)
    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)

    if len(seed) != 5 or not seed.isdigit():
        st.sidebar.error("Draw 1-back must be exactly 5 digits")
        return

    # Hot/Cold auto-calc
    hot_digits, cold_digits = compute_hot_cold([seed] + prev_draws if prev_draws else [seed])
    if hot_input:
        hot_digits = [int(x) for x in hot_input.split(',') if x.strip().isdigit()]
    if cold_input:
        cold_digits = [int(x) for x in cold_input.split(',') if x.strip().isdigit()]

    st.sidebar.markdown(f"**Hot digits:** {hot_digits}  \n**Cold digits:** {cold_digits}")

    # Load filters
    filters = load_filters()

    # Generate combos
    combos = generate_combinations(seed, method)

    # Initial elimination counts
    init_counts = {flt['id']: 0 for flt in filters}
    for flt in filters:
        for combo in combos:
            cdigits = [int(c) for c in combo]
            ctx = {
                'combo_digits': cdigits,
                'combo_sum': sum(cdigits),
                'combo_sum_cat': sum_category(sum(cdigits)),
                'hot_digits': hot_digits,
                'cold_digits': cold_digits,
                'Counter': Counter,
            }
            try:
                if eval(flt['expr_code'], ctx, ctx):
                    init_counts[flt['id']] += 1
            except:
                pass

    # Filter display list
    sorted_filters = sorted(filters, key=lambda f: (init_counts[f['id']] == 0, -init_counts[f['id']]))
    display_filters = [f for f in sorted_filters if not hide_zero or init_counts[f['id']] > 0]

    # Sequential elimination
    pool = combos.copy()
    dynamic_counts = {}
    for flt in display_filters:
        key = f"filter_{flt['id']}"
        active = st.sidebar.checkbox(
            f"{flt['id']} | {flt['layman']} | hist {flt['stat']} | cut {init_counts[flt['id']]}",
            key=key,
            value=st.session_state.get(key, select_all and flt['enabled_default'])
        )
        dc = 0
        survivors_pool = []
        if active:
            for combo in pool:
                cdigits = [int(c) for c in combo]
                ctx = {
                    'combo_digits': cdigits,
                    'combo_sum': sum(cdigits),
                    'combo_sum_cat': sum_category(sum(cdigits)),
                    'hot_digits': hot_digits,
                    'cold_digits': cold_digits,
                    'Counter': Counter,
                }
                try:
                    if eval(flt['expr_code'], ctx, ctx):
                        dc += 1
                    else:
                        survivors_pool.append(combo)
                except:
                    survivors_pool.append(combo)
        else:
            survivors_pool = pool.copy()
        dynamic_counts[flt['id']] = dc
        pool = survivors_pool

    survivors = pool

    # Sidebar totals
    st.sidebar.markdown(f"**Total:** {len(combos)}  \n**Elim: {len(combos)-len(survivors)}  \n**Remain: {len(survivors)}")

    # Combo tracker
    if check_combo:
        norm = ''.join(sorted(check_combo))
        if norm in survivors:
            st.sidebar.success(f"Combo {check_combo} survived all filters")
        else:
            st.sidebar.error(f"Combo {check_combo} was eliminated")

    # Survivors download
    survivors_df = pd.DataFrame(survivors, columns=["Combo"])
    csv_bytes = survivors_df.to_csv(index=False).encode("utf-8")
    txt_bytes = "\n".join(survivors).encode("utf-8")
    st.download_button("📥 Download Survivors (CSV)", data=csv_bytes, file_name="survivors.csv", mime="text/csv")
    st.download_button("📥 Download Survivors (TXT)", data=txt_bytes, file_name="survivors.txt", mime="text/plain")

    with st.expander("Show remaining combinations"):
        for c in survivors:
            st.write(c)


if __name__ == "__main__":
    main()
