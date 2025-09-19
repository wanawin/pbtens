import streamlit as st
import csv, os
from typing import List, Dict, Any
from collections import Counter

# ------------------------
# Utilities
# ------------------------
def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return 'Very Low'
    elif 16 <= total <= 24:
        return 'Low'
    elif 25 <= total <= 33:
        return 'Mid'
    else:
        return 'High'

# ------------------------
# Load Filters
# ------------------------
def load_filters(path: str) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return filters
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader):
            row = { (k or "").strip().lower(): (v or "").strip() for k,v in raw.items() }
            row["id"]   = row.get("id", f"F{i+1}")
            row["name"] = row.get("name", row.get("layman",""))
            row["stat"] = row.get("stat","")
            app  = row.get("applicable_if","True")
            expr = row.get("expression","False").replace("!==","!=")

            try:
                row["_app_code"]  = compile(app,  "<applicable>", "eval")
                row["_expr_code"] = compile(expr, "<expr>", "eval")
                row["_error"] = None
            except SyntaxError as e:
                row["_app_code"]=None; row["_expr_code"]=None
                row["_error"]=str(e)

            row["_expr_str"]=expr
            row["_enabled_default"] = (row.get("enabled","").lower()=="true")
            filters.append(row)
    return filters

# ------------------------
# Main App
# ------------------------
def main():
    st.set_page_config(page_title="Powerball Tens-Only — Manual Filter Runner", layout="wide")
    st.sidebar.header("⚙ Controls")

    # seed + 6 back draws
    seed     = st.sidebar.text_input("Seed tens (Draw 1-back, 5 digits 0–6):").strip()
    prevs    = [st.sidebar.text_input(f"Draw {i}-back (optional):").strip() for i in range(2,7)]

    method   = st.sidebar.selectbox("Generation Method:", ["1-digit","2-digit pair"])
    hot_in   = st.sidebar.text_input("Hot digits (manual 0–6, comma-separated):").strip()
    cold_in  = st.sidebar.text_input("Cold digits (manual 0–6, comma-separated):").strip()
    due_in   = st.sidebar.text_input("Due digits (manual 0–6, comma-separated):").strip()

    disable_due = st.sidebar.checkbox("Disable due-based filters when due set empty", value=True)

    track_combos = st.sidebar.text_area("Track/Test combos (newline/comma separated):").strip()
    preserve     = st.sidebar.checkbox("Preserve tracked combos during filtering")
    inject       = st.sidebar.checkbox("Inject tracked combos if not generated")
    select_all   = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero    = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)

    # -------------------
    # Validate seed
    # -------------------
    if not seed or not seed.isdigit() or len(seed)!=5:
        st.warning("Set inputs then click **Run / Refresh**.")
        return

    # -------------------
    # Build context for filters
    # -------------------
    def digits_from_str(s: str):
        return [int(x) for x in s if x.isdigit()]

    seed_digits = digits_from_str(seed)
    prev_digits_list = [digits_from_str(p) for p in prevs]

    # Hot/Cold auto-calc
    manual_hot  = [int(x) for x in hot_in.split(",") if x.strip().isdigit()]
    manual_cold = [int(x) for x in cold_in.split(",") if x.strip().isdigit()]
    manual_due  = [int(x) for x in due_in.split(",") if x.strip().isdigit()]

    history_digits = seed_digits + sum(prev_digits_list, [])
    hot_auto, cold_auto, due_auto = [], [], []
    if history_digits:
        freq = Counter(history_digits)
        hot_auto  = [d for d,_ in freq.most_common(3)]
        cold_auto = [d for d,_ in freq.most_common()[:-4:-1]]
        all_seen = set(history_digits)
        due_auto = [d for d in range(7) if d not in all_seen]

    # merge auto + manual, dedup
    hot_digits  = sorted(set(manual_hot or hot_auto))
    cold_digits = sorted(set(manual_cold or cold_auto))
    due_digits  = sorted(set(manual_due or (due_auto if not disable_due else [])))

    st.sidebar.markdown(f"Auto Hot: {hot_auto}, Auto Cold: {cold_auto}, Auto Due: {due_auto}")

    # -------------------
    # Generate combos
    # -------------------
    all_digits = "0123456"
    combos = []
    if method=="1-digit":
        for d in seed_digits:
            for a in all_digits:
                for b in all_digits:
                    for c in all_digits:
                        for e in all_digits:
                            combos.append("".join(sorted([str(d),a,b,c,e])))
    else:
        pairs = { "".join(sorted((str(seed_digits[i]), str(seed_digits[j]))))
                 for i in range(len(seed_digits)) for j in range(i+1,len(seed_digits)) }
        for pair in pairs:
            for a in all_digits:
                for b in all_digits:
                    for c in all_digits:
                        combos.append("".join(sorted(pair+a+b+c)))

    combos = sorted(set(combos))

    # -------------------
    # Inject tracked combos if missing
    # -------------------
    tracked = []
    if track_combos:
        for part in track_combos.replace("\n",",").split(","):
            p = part.strip()
            if p:
                tracked.append("".join(sorted(p)))
    if inject:
        combos = sorted(set(combos).union(tracked))

    # -------------------
    # Load filters
    # -------------------
    filters = load_filters("pb_tens_filters_adapted.csv")
    if not filters:
        st.error("❌ No filters loaded. Check your CSV is in repo and correctly formatted.")
        return

    # -------------------
    # Evaluate filters
    # -------------------
    init_counts = {flt["id"]:0 for flt in filters}
    for flt in filters:
        if flt["_app_code"] is None or flt["_expr_code"] is None:
            continue
        for combo in combos:
            ctx = {
                "combo": combo,
                "combo_digits":[int(c) for c in combo],
                "tens_sum": sum(int(c) for c in combo),
                "hot_digits": hot_digits,
                "cold_digits": cold_digits,
                "due_digits": due_digits
            }
            try:
                if eval(flt["_app_code"], ctx, ctx) and eval(flt["_expr_code"], ctx, ctx):
                    init_counts[flt["id"]] += 1
            except Exception as e:
                flt["_error"]=str(e)

    # sort filters
    sorted_filters = sorted(filters, key=lambda f: (init_counts[f["id"]]==0, -init_counts[f["id"]]))
    if hide_zero:
        display_filters = [f for f in sorted_filters if init_counts[f["id"]]>0]
    else:
        display_filters = sorted_filters

    # -------------------
    # Apply filters sequentially
    # -------------------
    pool = combos.copy()
    dyn_counts = {}
    for idx, flt in enumerate(display_filters):
        key=f"flt_{flt['id']}_{idx}"  # unique key
        active = st.session_state.get(key, select_all and flt["_enabled_default"])
        dc=0; survivors_pool=[]
        if active and flt["_app_code"] and flt["_expr_code"]:
            for combo in pool:
                ctx={"combo":combo,"combo_digits":[int(c) for c in combo],
                     "tens_sum": sum(int(c) for c in combo),
                     "hot_digits":hot_digits,"cold_digits":cold_digits,
                     "due_digits":due_digits}
                try:
                    if eval(flt["_app_code"], ctx, ctx) and eval(flt["_expr_code"], ctx, ctx):
                        dc+=1
                        if combo in tracked and preserve: survivors_pool.append(combo)
                    else:
                        survivors_pool.append(combo)
                except: survivors_pool.append(combo)
        else:
            survivors_pool=pool.copy()
        dyn_counts[flt["id"]]=dc
        pool=survivors_pool

    survivors=pool

    # -------------------
    # UI
    # -------------------
    st.header("🛠 Manual Filters")
    for idx, flt in enumerate(display_filters):
        key=f"flt_{flt['id']}_{idx}"
        ic=init_counts[flt["id"]]; dc=dyn_counts.get(flt["id"],0)
        label=f"{flt['id']}: {flt['name']} — {dc}/{ic} eliminated (stat={flt['stat']})"
        if flt["_error"]:
            label += f" ⚠Error: {flt['_error']}"
        st.checkbox(label, key=key, value=st.session_state.get(key, select_all and flt["_enabled_default"]))

    st.subheader(f"Remaining after filters: {len(survivors)}")
    with st.expander("Show survivors"):
        st.write(survivors)

    if tracked:
        st.subheader("Tracked Combos — Audit")
        for t in tracked:
            if t in survivors: st.success(f"{t} survived")
            else: st.warning(f"{t} eliminated or not generated")

if __name__=="__main__":
    main()
