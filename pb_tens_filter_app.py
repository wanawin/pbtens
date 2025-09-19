# pb_tens_filter_app.py — Powerball Tens-Only Manual Filter Runner
# ✅ 6 draws back for hot/cold/due (seed + 5 previous)
# ✅ Filters always visible with checkboxes (default unchecked)
# ✅ Select/Deselect All + hide zero-cut filters
# ✅ Each filter row shows: ID + layman (name) + initial cuts (+ optional stat column if present)
# ✅ Tracked combo audit (generation → elimination)
# ✅ Survivors list collapsible + CSV/TXT downloads
# ✅ Optional upload of a filter CSV to override repo default

from __future__ import annotations
import os, csv
from itertools import product
from collections import Counter
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st

# ---------------------------
# Tens-only model (0..6)
# ---------------------------
TENS_DOMAIN = "0123456"      # main balls tens digits 0..6
LOW_SET  = {0,1,2,3,4}
HIGH_SET = {5,6}

def sum_category(total: int) -> str:
    if 0 <= total <= 10:   return "Very Low"
    if 11 <= total <= 13:  return "Low"
    if 14 <= total <= 17:  return "Mid"
    return "High"

# ========================
# Filter loading (unified CSV)
# Expected columns (case-insensitive):
#   id, name (layman), applicable_if, expression, enabled (optional), stat (optional)
# ========================
def load_filters(paths) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = { (k or "").lower(): (v if isinstance(v,str) else v) for k,v in raw.items() }
                row["id"]   = (row.get("id") or row.get("fid") or "").strip()
                row["name"] = (row.get("name") or "").strip().strip('"').strip("'")
                app = (row.get("applicable_if") or "True").strip().strip('"').strip("'")
                expr = (row.get("expression") or "False").strip().strip('"').strip("'")
                expr = expr.replace("!==", "!=")  # tolerate accidental !==

                # compile once
                try:
                    row["_app_code"]  = compile(app,  "<applicable>", "eval")
                    row["_expr_code"] = compile(expr, "<expr>",       "eval")
                except SyntaxError:
                    # mark as malformed; we'll list it in the sidebar
                    row["_app_code"]  = None
                    row["_expr_code"] = None

                row["_source"] = os.path.basename(path)
                row["_enabled_default"] = (str(row.get("enabled","")).lower() == "true")
                row["_expr_str"] = expr
                filters.append(row)
    return filters

# ========================
# Tens combo generation
# ========================
def generate_tens_combinations_both(seed_tens: str, method: str) -> Tuple[List[str], List[str]]:
    """
    Returns (RAW_with_duplicates, UNIQUE_sorted) tens keys (5 chars, 0..6) sorted as multiset, e.g. '00123'
    """
    seed_tens = "".join(sorted(seed_tens))
    raw, uniq = [], set()
    if method == "1-digit":
        for d in seed_tens:
            for p in product(TENS_DOMAIN, repeat=4):
                key = "".join(sorted(d + "".join(p)))
                raw.append(key); uniq.add(key)
    else:
        pairs = { "".join(sorted((seed_tens[i], seed_tens[j])))
                  for i in range(len(seed_tens)) for j in range(i+1, len(seed_tens)) }
        for pair in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                key = "".join(sorted(pair + "".join(p)))
                raw.append(key); uniq.add(key)
    return raw, sorted(uniq)

# ========================
# Context builder (tens)
# ========================
def multiset_shared(a_digits, b_digits):
    ca, cb = Counter(a_digits), Counter(b_digits)
    return sum((ca & cb).values())

def norm_digits_list_0_6(txt: str) -> List[int]:
    out=[]
    for tok in txt.replace(",", " ").split():
        if tok.isdigit():
            v=int(tok)
            if 0<=v<=6: out.append(v)
    return out

def build_ctx(
    combo_str: str,
    seed_list: List[str],
    hot_digits: List[int],
    cold_digits: List[int],
    due_digits: List[int]
) -> Dict[str, Any]:
    # seed_list = [seed, prev1, prev2, prev3, prev4, prev5]
    seed  = [int(x) for x in (seed_list[0] or "")]
    prev1 = [int(x) for x in (seed_list[1] or "")]
    prev2 = [int(x) for x in (seed_list[2] or "")]
    prev3 = [int(x) for x in (seed_list[3] or "")]
    prev4 = [int(x) for x in (seed_list[4] or "")]
    prev5 = [int(x) for x in (seed_list[5] or "")]
    combo = [int(c) for c in combo_str]

    tens_sum = sum(combo)
    tens_even = sum(1 for d in combo if d%2==0)
    tens_odd  = 5 - tens_even
    tens_unique = len(set(combo))
    tens_range  = max(combo) - min(combo)
    tens_low    = sum(1 for d in combo if d in LOW_SET)
    tens_high   = sum(1 for d in combo if d in HIGH_SET)

    seed_sum = sum(seed) if seed else 0

    ctx: Dict[str, Any] = {
        "combo_tens": combo,
        "seed_tens": seed,
        "prev_seed_tens": prev1,
        "prev_prev_seed_tens": prev2,
        "prev3_seed_tens": prev3,
        "prev4_seed_tens": prev4,
        "prev5_seed_tens": prev5,

        "tens_sum": tens_sum,
        "seed_tens_sum": seed_sum,
        "tens_even_count": tens_even,
        "tens_odd_count": tens_odd,
        "tens_unique_count": tens_unique,
        "tens_range": tens_range,
        "tens_low_count": tens_low,
        "tens_high_count": tens_high,

        "last2": set(seed) | set(prev1),
        "common_to_both": set(seed) & set(prev1),

        "Counter": Counter,
        "sum_category": sum_category,
        "shared_tens": multiset_shared,

        "hot_digits": list(hot_digits),
        "cold_digits": list(cold_digits),
        "due_digits": list(due_digits),
    }
    return ctx

def normalize_combo_text(text: str) -> Tuple[List[str], List[str]]:
    toks=[]
    for line in text.splitlines():
        toks.extend([t.strip() for t in line.replace(",", " ").split()])
    normalized, invalid = [], []
    for tok in toks:
        digs=[c for c in tok if c.isdigit()]
        if len(digs)!=5 or any(c not in TENS_DOMAIN for c in digs):
            invalid.append(tok); continue
        normalized.append("".join(sorted(digs)))
    seen=set(); out=[]
    for n in normalized:
        if n not in seen: out.append(n); seen.add(n)
    return out, invalid

# ========================
# UI + Logic
# ========================
st.set_page_config(page_title="Powerball Tens-Only — Manual Filter Runner", layout="wide")

def main():
    st.sidebar.header("🛠️ Run control")
    run_clicked = st.sidebar.button("Run / Refresh", type="primary")

    # Filter source(s)
    st.sidebar.markdown("---")
    st.sidebar.caption("Default loads from repo: pb_tens_filters_adapted.csv")
    uploaded_filters = st.sidebar.file_uploader("Upload filter CSV (optional)", type=["csv"])
    default_filters_path = "pb_tens_filters_adapted.csv"

    filter_paths=[]
    if uploaded_filters is not None:
        upath="user_filters.csv"
        with open(upath, "wb") as f: f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)
    elif os.path.exists(default_filters_path):
        filter_paths.append(default_filters_path)
    else:
        st.sidebar.error("No filter CSV found. Upload one or add pb_tens_filters_adapted.csv to the repo.")
        return

    # Seed inputs (seed + 5 previous = 6 total draws back)
    seed  = st.sidebar.text_input("Seed tens (draw 1-back, 5 digits 0–6):", value="", key="seed").strip()
    prev1 = st.sidebar.text_input("Prev (2-back, optional):", value="", key="prev1").strip()
    prev2 = st.sidebar.text_input("Prev (3-back, optional):", value="", key="prev2").strip()
    prev3 = st.sidebar.text_input("Prev (4-back, optional):", value="", key="prev3").strip()
    prev4 = st.sidebar.text_input("Prev (5-back, optional):", value="", key="prev4").strip()
    prev5 = st.sidebar.text_input("Prev (6-back, optional):", value="", key="prev5").strip()

    method = st.sidebar.selectbox("Generation Method:", ["1-digit","2-digit pair"], index=0)

    st.sidebar.markdown("### Hot / Cold / Due")
    auto_hotcold = st.sidebar.checkbox("Auto-calc Hot/Cold from last 6 draws (seed + prev1..prev5)", value=True)
    hot_manual   = st.sidebar.text_input("Hot tens digits (manual 0–6, comma-separated):", value="")
    cold_manual  = st.sidebar.text_input("Cold tens digits (manual 0–6, comma-separated):", value="")
    due_manual   = st.sidebar.text_input("Due digits (manual 0–6, comma-separated):", value="")
    disable_due_when_empty = st.sidebar.checkbox("Disable due-based filters when due set empty", value=True)

    # Tracked combos / audit
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test combos (e.g. 00123, 23345; newline or comma-separated):", height=110)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked   = st.sidebar.checkbox("Inject tracked combos if not generated", value=False)

    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False, key="master_toggle")
    hide_zero  = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=False, key="hide0")

    if not run_clicked:
        st.info("Set inputs then click **Run / Refresh**.")
        return

    # Basic validation
    if len(seed) != 5 or (not seed.isdigit()) or any(c not in TENS_DOMAIN for c in seed):
        st.error("Seed tens must be exactly 5 digits in 0–6 (e.g., 23345).")
        return
    for label, val in [("Prev",prev1),("Prev 3-back",prev2),("Prev 4-back",prev3),("Prev 5-back",prev4),("Prev 6-back",prev5)]:
        if val and (len(val)!=5 or (not val.isdigit()) or any(c not in TENS_DOMAIN for c in val)):
            st.error(f"{label} must be 5 digits in 0–6 or blank.")
            return

    # LOAD FILTERS
    filters = load_filters(filter_paths)
    malformed = [f for f in filters if f["_app_code"] is None or f["_expr_code"] is None]
    if malformed:
        with st.sidebar.expander("⚠️ Filters with syntax errors", expanded=False):
            for f in malformed[:100]:
                st.write(f'**{f.get("id","(no id)")}** — {f.get("name","(no name)")} ({f.get("_source")})')

    # Seeds chain, auto hot/cold/due
    seeds_chain = [seed, prev1, prev2, prev3, prev4, prev5]

    # Auto hot/cold from provided draws (seed + prev1..prev5)
    auto_hot, auto_cold = [], []
    if auto_hotcold:
        counts = Counter()
        for s in seeds_chain:
            for ch in (s or ""): counts[int(ch)] += 1
        if counts:
            maxf, minf = max(counts.values()), min(counts.values())
            auto_hot = sorted([k for k,c in counts.items() if c==maxf])
            auto_cold = sorted([k for k,c in counts.items() if c==minf])

    # Manual overrides (if provided, we use them; else auto)
    hot_digits  = norm_digits_list_0_6(hot_manual)  if hot_manual else auto_hot
    cold_digits = norm_digits_list_0_6(cold_manual) if cold_manual else auto_cold

    # Due: manual else auto = digits 0..6 not seen in last 2 draws (seed & prev1) — consistent with earlier logic
    if due_manual:
        due_digits = norm_digits_list_0_6(due_manual)
    else:
        seen_last2=set()
        for s in [seed, prev1]:
            for ch in (s or ""): seen_last2.add(int(ch))
        due_digits = [d for d in range(7) if d not in seen_last2]

    # GENERATE POOL (raw & unique)
    raw_combos, unique_baseline = generate_tens_combinations_both(seed, method)
    combos = sorted(set(raw_combos))  # no zone filters in this tens-only build

    # Normalize tracked list
    tracked_norm, invalid = normalize_combo_text(track_text)
    if invalid:
        st.sidebar.warning("Ignored invalid entries: " + ", ".join(invalid[:5]) + (" ..." if len(invalid)>5 else ""))
    tracked_set = set(tracked_norm)

    generated_set = set(combos)
    audit = {
        c: {"combo": c, "generated": (c in generated_set), "survived": None,
            "eliminated": False, "eliminated_by": None, "eliminated_order": None,
            "eliminated_name": None, "would_eliminate_by": None,
            "would_eliminate_order": None, "would_eliminate_name": None,
            "injected": False, "preserved": bool(preserve_tracked)}
        for c in tracked_norm
    }

    if inject_tracked:
        for c in tracked_norm:
            if c not in generated_set:
                combos.append(c); generated_set.add(c)
                if c in audit: audit[c]["injected"]=True

    # Initial cuts (per filter, before selection)
    ui_filters = [f for f in filters if f["_app_code"] and f["_expr_code"]]
    init_cuts: Dict[str, int] = {f["id"]: 0 for f in ui_filters}
    for flt in ui_filters:
        # optional skip when due empty
        if disable_due_when_empty and not due_digits and "due_digits" in (flt.get("_expr_str") or ""):
            init_cuts[flt["id"]]=0; continue
        ic=0
        for combo in combos:
            ctx = build_ctx(
                combo_str=combo,
                seed_list=seeds_chain,
                hot_digits=hot_digits,
                cold_digits=cold_digits,
                due_digits=due_digits
            )
            try:
                if eval(flt["_app_code"], ctx, ctx) and eval(flt["_expr_code"], ctx, ctx):
                    ic+=1
            except Exception:
                pass
        init_cuts[flt["id"]]=ic

    st.sidebar.markdown("---")
    st.sidebar.write(f"Generated unique combos: **{len(combos)}**")

    # Sort filters by most→least cuts; 0-cuts bottom (unless hide_zero)
    sorted_filters = sorted(
        ui_filters,
        key=lambda f: (init_cuts[f["id"]]==0, -init_cuts[f["id"]], f["id"])
    )
    display_filters = [f for f in sorted_filters if init_cuts[f["id"]]>0] if hide_zero else sorted_filters

    # =======================
    # MAIN SECTION
    # =======================
    st.title("🧰 Manual Filters (Tens Only)")

    # Filters list (collapsible)
    pool = list(combos)
    dynamic_counts: Dict[str,int] = {}
    order_index = 0

    with st.expander("Filters (most → least aggressive)", expanded=True):
        for flt in display_filters:
            order_index += 1
            fid = flt.get("id") or f"row{order_index}"
            name = flt.get("name","")
            stat = flt.get("stat","")
            cuts = init_cuts.get(fid, 0)
            label = f"{fid}: {name}"
            if stat: label += f" — stat: {stat}"
            label += f" — initial cuts: {cuts}"

            ui_key = f"chk_{fid}_{order_index}"
            default_checked = (select_all and flt["_enabled_default"])
            checked = st.checkbox(label, key=ui_key, value=default_checked)

            if not checked:
                dynamic_counts[fid]=0
                continue

            # Apply this filter
            survivors=[]; dc=0
            for combo in pool:
                ctx = build_ctx(
                    combo_str=combo,
                    seed_list=seeds_chain,
                    hot_digits=hot_digits,
                    cold_digits=cold_digits,
                    due_digits=due_digits
                )
                eliminate=False
                if disable_due_when_empty and not due_digits and "due_digits" in (flt.get("_expr_str") or ""):
                    eliminate=False
                else:
                    try:
                        eliminate = eval(flt["_app_code"], ctx, ctx) and eval(flt["_expr_code"], ctx, ctx)
                    except Exception:
                        eliminate=False

                is_tracked = combo in tracked_set
                if eliminate:
                    if is_tracked and preserve_tracked:
                        info=audit.get(combo)
                        if info and info["would_eliminate_by"] is None:
                            info["would_eliminate_by"]=fid
                            info["would_eliminate_order"]=order_index
                            info["would_eliminate_name"]=name
                        survivors.append(combo)  # keep preserved
                    else:
                        dc+=1
                        info=audit.get(combo)
                        if info and not info["eliminated"]:
                            info["eliminated"]=True
                            info["eliminated_by"]=fid
                            info["eliminated_order"]=order_index
                            info["eliminated_name"]=name
                else:
                    survivors.append(combo)
            dynamic_counts[fid]=dc
            pool=survivors

    st.subheader(f"Remaining after filters: {len(pool)}")

    # Survivors expander (collapsible) + tracked highlights
    with st.expander("Show survivors"):
        tracked_survivors=[c for c in pool if c in tracked_set]
        if tracked_survivors:
            st.write("**Tracked survivors:**")
            for c in tracked_survivors:
                info=audit.get(c,{})
                if info.get("would_eliminate_by"):
                    st.write(f"{c} — ⚠ would be eliminated by {info['would_eliminate_by']} at step {info.get('would_eliminate_order')} ({info.get('would_eliminate_name')}) — preserved")
                else:
                    st.write(c)
            st.write("---")
        for c in pool:
            if c not in tracked_set: st.write(c)

    # Track a specific combo quickly (from the sidebar input list)
    if tracked_norm:
        st.markdown("### 🔎 Tracked/Preserved Combos — Audit")
        survivors_set=set(pool)
        rows=[]
        for c in tracked_norm:
            info=audit.get(c,{})
            rows.append({
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
        df_audit=pd.DataFrame(rows)
        st.dataframe(df_audit, use_container_width=True)
        st.download_button(
            "Download audit (CSV)",
            df_audit.to_csv(index=False).encode("utf-8"),
            file_name="pb_tens_audit_tracked.csv",
            mime="text/csv"
        )

    # Downloads for survivors
    df_out = pd.DataFrame({"tens_combo": pool})
    st.download_button(
        "Download survivors CSV",
        df_out.to_csv(index=False).encode("utf-8"),
        file_name="pb_tens_survivors.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download survivors TXT",
        ("\n".join(pool)).encode("utf-8"),
        file_name="pb_tens_survivors.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
