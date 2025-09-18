import streamlit as st
from itertools import product
from collections import Counter
import csv, os
import pandas as pd

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = '0123456'  # main balls tens digits 0..6 only
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
                for key in ('name','applicable_if','expression'):
                    if key in row and isinstance(row[key], str):
                        row[key] = row[key].strip().strip('"').strip("'")
                row['expression'] = (row.get('expression') or 'False').replace('!==','!=')
                row['expr_str'] = row['expression']
                applicable = row.get('applicable_if') or 'True'
                expr = row.get('expression') or 'False'
                try:
                    row['applicable_code'] = compile(applicable,'<applicable>','eval')
                    row['expr_code'] = compile(expr,'<expr>','eval')
                except SyntaxError as e:
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                row['enabled_default'] = (row.get('enabled','').lower() == 'true')
                row['_ui_key'] = f"{row['id']}_{os.path.basename(path)}"
                filters.append(row)
    return filters

# ========================
# Tens combo generation
# ========================
def generate_tens_combinations_both(seed_tens: str, method: str):
    seed_tens = ''.join(sorted(seed_tens))
    raw, uniq = [], set()
    if method == '1-digit':
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                key = ''.join(sorted(d+''.join(p)))
                raw.append(key); uniq.add(key)
    else:
        pairs = {''.join(sorted((seed_tens[i], seed_tens[j])))
                 for i in range(len(seed_tens)) for j in range(i+1,len(seed_tens))}
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = ''.join(sorted(pair+''.join(p)))
                raw.append(key); uniq.add(key)
    return raw, sorted(uniq)

# ========================
# Context builder
# ========================
def multiset_shared(a,b):
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())

def build_ctx(seed, prev, prev2, combo, hot_input, cold_input, due_digits):
    seed_tens=[int(x) for x in seed]
    prev_tens=[int(x) for x in prev] if prev else []
    prev2_tens=[int(x) for x in prev2] if prev2 else []
    combo_tens=[int(c) for c in combo]
    tens_sum=sum(combo_tens)
    tens_even=sum(1 for d in combo_tens if d%2==0)
    tens_odd=5-tens_even
    tens_unique=len(set(combo_tens))
    tens_range=max(combo_tens)-min(combo_tens)
    tens_low=sum(1 for d in combo_tens if d in LOW_SET)
    tens_high=sum(1 for d in combo_tens if d in HIGH_SET)
    hot=[int(x) for x in hot_input.split(',') if x.strip().isdigit() and int(x) in range(7)]
    cold=[int(x) for x in cold_input.split(',') if x.strip().isdigit() and int(x) in range(7)]
    ctx={
        'combo_tens':combo_tens,
        'seed_tens':seed_tens,
        'prev_seed_tens':prev_tens,
        'prev_prev_seed_tens':prev2_tens,
        'tens_sum':tens_sum,
        'tens_even_count':tens_even,
        'tens_odd_count':tens_odd,
        'tens_unique_count':tens_unique,
        'tens_range':tens_range,
        'tens_low_count':tens_low,
        'tens_high_count':tens_high,
        'hot_digits':hot,
        'cold_digits':cold,
        'due_digits':due_digits,
        'shared_tens':multiset_shared,
        'sum_category':sum_category,
    }
    return ctx

# ========================
# Streamlit UI
# ========================
def main():
    st.sidebar.header("🎯 Tens-Only — Manual Filter Runner")

    # --- Filter load
    default_path="pb_tens_filters_adapted.csv"
    filters=load_filters([default_path]) if os.path.exists(default_path) else []

    # --- Inputs
    seed=st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):","").strip()
    prev=st.sidebar.text_input("Prev tens (optional):","").strip()
    prev2=st.sidebar.text_input("Prev-prev tens (optional):","").strip()
    method=st.sidebar.selectbox("Generation Method:",["1-digit","2-digit pair"])
    hot_input=st.sidebar.text_input("Hot digits (optional):","").strip()
    cold_input=st.sidebar.text_input("Cold digits (optional):","").strip()

    # Due digits (auto from last 3 seeds)
    all_seen=set()
    for s in [seed,prev,prev2]:
        if s: all_seen.update(int(x) for x in s)
    auto_due=[d for d in range(7) if d not in all_seen]
    st.sidebar.write(f"Auto due set: {auto_due}")

    if not seed or len(seed)!=5 or any(c not in TENS_DOMAIN for c in seed):
        st.warning("Enter a valid 5-digit seed (0–6)."); return

    # --- Generate combos
    raw, uniq=generate_tens_combinations_both(seed,method)
    st.sidebar.write(f"Generated {len(uniq)} unique combos")

    # --- Initial elimination counts
    init_counts={flt['_ui_key']:0 for flt in filters}
    for flt in filters:
        ic=0
        for combo in uniq:
            ctx=build_ctx(seed,prev,prev2,combo,hot_input,cold_input,auto_due)
            try:
                if eval(flt['applicable_code'],ctx,ctx) and eval(flt['expr_code'],ctx,ctx):
                    ic+=1
            except: pass
        init_counts[flt['_ui_key']]=ic

    # --- Filter UI
    master_toggle=st.sidebar.checkbox("Select/Deselect All Filters",value=False)
    hide_zero=st.sidebar.checkbox("Hide filters with 0 eliminations",value=True)

    st.header("🔧 Manual Filters")
    sorted_filters=sorted(filters,key=lambda flt:(init_counts[flt['_ui_key']]==0,-init_counts[flt['_ui_key']]))
    display_filters=[f for f in sorted_filters if init_counts[f['_ui_key']]>0] if hide_zero else sorted_filters

    pool=list(uniq)
    order_index=0
    for idx,flt in enumerate(display_filters):
        order_index+=1
        cuts=init_counts[flt['_ui_key']]
        label=f"{flt['id']} — {flt.get('name','')} (init cuts {cuts})"
        ui_key=f"chk_{flt['_ui_key']}_{idx}"
        checked=st.checkbox(label,key=ui_key,value=master_toggle)
        if checked:
            survivors=[]
            for combo in pool:
                ctx=build_ctx(seed,prev,prev2,combo,hot_input,cold_input,auto_due)
                eliminate=False
                try:
                    eliminate=eval(flt['applicable_code'],ctx,ctx) and eval(flt['expr_code'],ctx,ctx)
                except: pass
                if not eliminate: survivors.append(combo)
            pool=survivors

    st.subheader(f"Remaining after filters: {len(pool)}")
    with st.expander("Show survivors"):
        for c in pool: st.write(c)

    df_out=pd.DataFrame({"tens_combo":pool})
    st.download_button("Download survivors (CSV)",df_out.to_csv(index=False),file_name="pb_tens_survivors.csv",mime="text/csv")

if __name__=="__main__":
    main()
