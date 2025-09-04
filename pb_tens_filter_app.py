
# streamlit run pb_tens_filter_app_full.py
import os, csv, re
from itertools import product
from typing import List, Tuple, Dict, Any
import streamlit as st

st.set_page_config(page_title="Tens Filter App (Percentiles pre‑dedup)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

TENS_DOMAIN = list(range(0,7))  # 0..6 tens digits

def digits_from_str(s: str) -> List[int]:
    """Parse five tens digits from inputs like '0-2-2-4-5', '0,2,2,4,5', '02245'."""
    if not s:
        return []
    toks = re.findall(r'\d', s)
    vals = [int(t) for t in toks]
    return vals

def validate_seed_tens(vals: List[int]) -> Tuple[bool, str]:
    if len(vals) != 5:
        return False, "Provide exactly five tens digits."
    if any(v < 0 or v > 6 for v in vals):
        return False, "Tens digits must be between 0 and 6."
    return True, ""

def generate_tens_combinations(seed_tens: str, method: str) -> List[str]:
    """
    Return UNIQUE combos (sorted keys) like '00123' built from seed tens and domain.
    Methods:
      - '1-digit': fix 1 digit from seed + 4 free digits from domain
      - '2-digit pairs': fix any 2 digits from seed + 3 free digits
    """
    seed = ''.join(sorted(seed_tens))
    combos_set = set()

    if method == '1-digit':
        for d in seed:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(str(x) for x in p)))
                combos_set.add(key)
    else:
        pairs = {''.join(sorted((seed[i], seed[j])))
                 for i in range(len(seed)) for j in range(i+1, len(seed))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(str(x) for x in p)))
                combos_set.add(key)

    return sorted(combos_set)

def generate_tens_combinations_both(seed_tens: str, method: str):
    """Return (RAW with duplicates, UNIQUE) combo keys as strings of length 5 over 0..6, sorted per key."""
    seed = ''.join(sorted(seed_tens))
    raw = []
    uniq = set()

    if method == '1-digit':
        for d in seed:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d + ''.join(str(x) for x in p)))
                raw.append(key)
                uniq.add(key)
    else:
        pairs = {''.join(sorted((seed[i], seed[j])))
                 for i in range(len(seed)) for j in range(i+1, len(seed))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair + ''.join(str(x) for x in p)))
                raw.append(key)
                uniq.add(key)
    return raw, sorted(uniq)

def build_ctx(seed_tens_list: List[int], combo_key: str) -> Dict[str, Any]:
    """Evaluation context for filter expressions. Expressions should ELIMINATE when they evaluate True.
       Exposes:
          combo_tens: List[int] length 5
          seed_tens:  List[int] length 5
          combo, seed: aliases
    """
    combo_tens = [int(c) for c in combo_key]  # '00123' -> [0,0,1,2,3]
    ctx = {
        'combo_tens': combo_tens,
        'seed_tens': seed_tens_list,
        'combo': combo_tens,
        'seed': seed_tens_list,
        # handy names
        'len': len,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'set': set,
        'all': all,
        'any': any,
        'sorted': sorted,
        'range': range,
        'TENS_DOMAIN': TENS_DOMAIN,
    }
    return ctx

# -----------------------------
# Filters loading/compiling
# -----------------------------
def load_filters_from_csv(name: str, file_bytes) -> List[dict]:
    filters = []
    if not file_bytes:
        return filters
    try:
        text = file_bytes.getvalue().decode('utf-8')
    except Exception:
        text = file_bytes.getvalue().decode('latin-1')
    lines = [ln.strip('\ufeff') for ln in text.splitlines() if ln.strip()]
    reader = csv.DictReader(lines)
    for raw in reader:
        row = {k.lower(): (v or "") for k,v in raw.items()}
        row['id'] = (row.get('id') or row.get('fid') or '').strip()
        for key in ('name','applicable_if','expression','enabled'):
            if key in row and isinstance(row[key], str):
                row[key] = row[key].strip().strip('"').strip("'")
        row['enabled_default'] = (row.get('enabled','').lower() == 'true')
        applicable = row.get('applicable_if') or 'True'
        expr = row.get('expression') or 'False'
        # Compile
        try:
            row['applicable_code'] = compile(applicable, '<applicable>', 'eval')
            row['expr_code'] = compile(expr, '<expr>', 'eval')
            row['expr_str'] = expr
        except Exception as e:
            row['compile_error'] = str(e)
        row['source_path'] = name
        # Mark percentile/zone if filename matches
        row['is_percentile'] = (os.path.basename(name) == 'pb_tens_percentile_filters.csv')
        filters.append(row)
    return filters

def apply_filter_list(pool_keys: List[str], filters: List[dict], seed_tens_list: List[int]) -> List[str]:
    survivors = []
    for key in pool_keys:
        ctx = build_ctx(seed_tens_list, key)
        eliminate = False
        for flt in filters:
            if flt.get('compile_error'):
                continue
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                    eliminate = True
                    break
            except Exception:
                # bad filter for tens context → ignore
                continue
        if not eliminate:
            survivors.append(key)
    return survivors

def count_elims(pool_keys: List[str], flt: dict, seed_tens_list: List[int]) -> int:
    if flt.get('compile_error'): return 0
    cnt = 0
    for key in pool_keys:
        ctx = build_ctx(seed_tens_list, key)
        try:
            if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx):
                cnt += 1
        except Exception:
            # incompatible expression for tens context
            pass
    return cnt

# -----------------------------
# UI
# -----------------------------

st.title("Tens Filter App — Percentiles pre‑dedup, comparison post‑dedup")

col_left, col_right = st.columns([2,1])
with col_left:
    seed_str = st.text_input("Seed tens (e.g., 0-2-2-4-5 or 02245)", value="0-2-2-4-5")
    method = st.selectbox("Generation method", ["1-digit", "2-digit pairs"], index=1)

with col_right:
    zone_file = st.file_uploader("Percentile/Zone Filters CSV (auto‑applied pre‑dedup)", type=["csv"], key="zone")
    extra_file = st.file_uploader("Additional Filters CSV (optional)", type=["csv"], key="extra")
    hide_zero = st.checkbox("Hide filters that cut 0 combos", value=True)

# Parse & validate seed
seed_vals = digits_from_str(seed_str)
ok, msg = validate_seed_tens(seed_vals)
if not ok:
    st.error(msg)
    st.stop()
seed_sorted_str = ''.join(str(x) for x in sorted(seed_vals))

# Load filters
filters = []
filters += load_filters_from_csv(zone_file.name if zone_file else "pb_tens_percentile_filters.csv", zone_file) if zone_file else []
filters += load_filters_from_csv(extra_file.name if extra_file else "pb_tens_filters.csv", extra_file) if extra_file else []

# Separate zone vs manual
zone_filters = [f for f in filters if f.get('is_percentile')]
manual_filters = [f for f in filters if not f.get('is_percentile')]

# Phase A: generate raw + unique
raw_keys, unique_keys = generate_tens_combinations_both(seed_sorted_str, method)

# Phase A: apply zone filters on RAW (pre‑dedup)
if zone_filters:
    zone_raw_survivors = apply_filter_list(raw_keys, zone_filters, seed_vals)
else:
    zone_raw_survivors = list(raw_keys)

# Phase B: deduplicate
inzone_unique = sorted(set(zone_raw_survivors))

# Metrics block
with st.sidebar:
    st.markdown("### Percentile/Zone pipeline")
    st.write(f"Raw generated (pre‑dedup): **{len(raw_keys):,}**")
    st.write(f"In‑zone survivors pre‑dedup: **{len(zone_raw_survivors):,}**")
    st.write(f"Unique baseline after dedup: **{len(unique_keys):,}**")
    st.write(f"In‑zone unique after dedup: **{len(inzone_unique):,}** "
             f"({len(inzone_unique)}/{len(unique_keys) if unique_keys else 1} kept)")

# Manual filter init pass (over in‑zone unique)
init_counts = {}
for flt in manual_filters:
    init_counts[flt['id']] = count_elims(inzone_unique, flt, seed_vals)

# Sorted display list
sorted_filters = sorted(manual_filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))  # cuts first
display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

st.markdown("### Manual Filters (applied after dedup & zones)")
if not display_filters:
    st.info("No manual filters to apply (or none cut anything). Upload an additional CSV to see filters here.")
else:
    selected_ids = []
    for flt in display_filters:
        with st.expander(f"[{flt['id']}] cuts {init_counts[flt['id']]:,} ", expanded=False):
            st.caption(flt.get('name','').strip() or "(unnamed)")
            st.code(flt.get('expr_str',''), language="python")
            checked = st.checkbox("Enable", value=flt.get('enabled_default', False), key=f"cb_{flt['id']}")
            if checked:
                selected_ids.append(flt['id'])

    # Apply selected manual filters
    selected_filters = [f for f in manual_filters if f['id'] in selected_ids]
    final_survivors = apply_filter_list(inzone_unique, selected_filters, seed_vals)

    st.markdown("### Results")
    st.write(f"**Start (unique baseline):** {len(unique_keys):,}  |  "
             f"**After zones (unique):** {len(inzone_unique):,}  |  "
             f"**After manual:** {len(final_survivors):,}")

    # Show sample
    st.write("Sample survivors (up to first 250):")
    st.write(final_survivors[:250])

    # Download
    out_csv = "combo_tens"
    csv_lines = ["combo_tens"]
    csv_lines += final_survivors
    st.download_button("Download survivors CSV", data="\n".join(csv_lines).encode("utf-8"),
                       file_name="tens_survivors.csv", mime="text/csv")

st.markdown("---")
st.caption("Expressions should evaluate to **True to eliminate** a combo. Context: combo_tens, seed_tens. "
           "Zone/percentile filters are auto‑applied **before dedup**; manual filters run after.")
