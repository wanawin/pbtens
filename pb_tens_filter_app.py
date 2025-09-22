# pb_tens_filter_app.py
# Tens — Manual Filter Runner (Powerball main-ball tens model)
# -------------------------------------------------------------
# - Six prior draws, hot/cold/due auto-calc (overridable)
# - Percentile (zone) filters auto-applied BEFORE de-dup
# - Manual filters in main content, sorted by aggressiveness
# - Historical stat + live cut count shown per filter
# - Tracked combo audit + survivors CSV/TXT download

import os, csv
from itertools import product
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
DIGITS = "0123456"  # tens domain
MANUAL_FILTER_CSV = "pb_tens_filters_adapted.csv"
PERCENTILE_FILTER_CSV = "pb_tens_percentile_filters.csv"  # optional

# -----------------------------
# Utilities
# -----------------------------
def _safe(s: str) -> str:
    return (s or "").strip()

def _int_list_from_csv(s: str, lo: int, hi: int) -> List[int]:
    out = []
    for t in (s or "").replace(",", " ").split():
        if t.isdigit():
            v = int(t)
            if lo <= v <= hi:
                out.append(v)
    return out

def digits_from_draw(draw: str) -> List[int]:
    draw = _safe(draw)
    if len(draw) != 5 or any(ch not in DIGITS for ch in draw):
        return []
    return [int(ch) for ch in draw]

def generate_combos(seed: str, method: str) -> List[str]:
    """Return sorted, normalized tens combos (each a 5-char string in 0..6)."""
    seed = "".join(sorted(seed))
    combos = set()
    if method == "1-digit":
        for d in seed:
            for p in product(DIGITS, repeat=4):
                combos.add("".join(sorted(d + "".join(p))))
    else:  # 2-digit pair
        pairs = {
            "".join(sorted((seed[i], seed[j])))
            for i in range(len(seed)) for j in range(i+1, len(seed))
        }
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                combos.add("".join(sorted(pair + "".join(p))))
    return sorted(combos)

def hot_cold_due_from_draws(draws: List[str]) -> Tuple[List[int], List[int], List[int]]:
    """Compute from up to 6 draws; if fewer than 6 valid, still compute from what we have."""
    valid = ["".join(d for d in _safe(x) if d in DIGITS) for x in draws if _safe(x)]
    all_digits = "".join(valid)
    if not all_digits:
        return [], [], []
    cnt = Counter(all_digits)
    # top-3 hot, bottom-3 cold (ties arbitrary)
    hot = [int(x) for x, _ in cnt.most_common(3)]
    cold = [int(x) for x, _ in cnt.most_common()[-3:]]
    due = [d for d in range(7) if str(d) not in all_digits]
    return hot, cold, due

# -----------------------------
# Filter loading + expression prep
# -----------------------------
def load_filter_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for i, raw in enumerate(rdr):
            row = { (k or "").strip().lower(): (v if isinstance(v, str) else v)
                    for k, v in raw.items() }
            fid = _safe(row.get("id") or row.get("fid") or row.get("filter_id") or f"row{i+1}")
            layman = _safe(row.get("layman") or row.get("layman_explanation"))
            hist = _safe(row.get("stat") or row.get("hist") or row.get("history") or "")
            expr_txt = _safe(row.get("expression") or row.get("expr"))

            # Keep the original text we’ll show; but prepare a runtime-safe expr.
            expr_runtime = normalize_expr(expr_txt)

            code = None
            err = None
            if expr_runtime:
                try:
                    code = compile(expr_runtime, f"<expr:{fid}>", "eval")
                except SyntaxError as e:
                    err = str(e)

            rows.append({
                "id": fid,
                "layman": layman,
                "hist": hist,
                "expr_text": expr_txt,
                "expr_runtime": expr_runtime,
                "code": code,
                "error": err,
            })
    return rows

def normalize_expr(expr: str) -> str:
    """
    Map legacy/variant fields to the tens-model context.
    We DO NOT remove content; we only translate common names.
    Anything unknown just evaluates with NameError -> safely ignored by try/except.
    """
    if not expr:
        return ""
    e = expr

    # Common renames to our runtime context (tens model):
    # - combo_digits: list[int] of current candidate tens
    # - seed_digits: list[int] of Draw 1-back tens
    # - prev2_digits, prev3_digits, prev4_digits, prev5_digits, prev6_digits
    # - last2: union of seed_digits & prev2_digits
    # - common_to_both: intersection of seed_digits & prev2_digits
    # - combo_sum: sum(combo_digits), seed_sum: sum(seed_digits)
    # - combo_unique: len(set(combo_digits))
    #
    # Legacy names we translate:
    # "winner" -> "combo_sum"
    # "winner_structure" -> "combo_unique"
    # "seed_value_digits" / "seed_value" -> "seed_digits" / "seed_sum"
    # "variant_name" comparisons -> always True (we're in tens)
    repl = {
        "winner_structure": "combo_unique",
        "winner value structure": "combo_unique",   # just in case plain text appeared
        "winner value": "combo_sum",                # very old
        "winner": "combo_sum",
        "seed_value_digits": "seed_digits",
        "seed_value": "seed_sum",
    }

    for k, v in repl.items():
        e = e.replace(k, v)

    # Handle explicit checks against variant_name; force-pass them.
    e = e.replace("(variant_name equals 'tens')", "True")
    e = e.replace("(variant_name == 'tens')", "True")
    e = e.replace("variant_name == 'tens'", "True")
    e = e.replace("variant_name='tens'", "True")

    return e

# -----------------------------
# Context for expression eval
# -----------------------------
def build_ctx(combo: str,
              seed: str,
              prev2: str, prev3: str, prev4: str, prev5: str, prev6: str,
              hot: List[int], cold: List[int], due: List[int]) -> Dict:
    cdigits = [int(c) for c in combo]
    seed_digits = digits_from_draw(seed)
    p2 = digits_from_draw(prev2)
    p3 = digits_from_draw(prev3)
    p4 = digits_from_draw(prev4)
    p5 = digits_from_draw(prev5)
    p6 = digits_from_draw(prev6)

    last2 = set(seed_digits) | set(p2)
    common = set(seed_digits) & set(p2)

    ctx = {
        # canonical
        "combo": combo,
        "combo_digits": cdigits,
        "cd": cdigits,  # short alias
        "combo_sum": sum(cdigits),
        "combo_unique": len(set(cdigits)),

        "seed_digits": seed_digits,
        "seed_sum": sum(seed_digits) if seed_digits else 0,

        "prev2_digits": p2,
        "prev3_digits": p3,
        "prev4_digits": p4,
        "prev5_digits": p5,
        "prev6_digits": p6,

        "last2": last2,
        "common_to_both": common,

        # hot/cold/due
        "hot": hot,
        "cold": cold,
        "due": due,

        # helpers
        "Counter": Counter,
        "set": set,
        "max": max,
        "min": min,
        "sum": sum,
        "len": len,
        "any": any,
        "all": all,
    }
    return ctx

# -----------------------------
# Filtering passes
# -----------------------------
def apply_percentile_pre_dedup(raw: List[str],
                               zone_filters: List[Dict],
                               base_ctx_kwargs: Dict) -> List[str]:
    """Keep only combos that pass ALL zone filters (they are eliminators)."""
    if not zone_filters:
        return list(raw)

    survivors = []
    for combo in raw:
        eliminate = False
        ctx = build_ctx(combo=combo, **base_ctx_kwargs)
        for flt in zone_filters:
            if flt["code"] is None:  # compile error -> ignore
                continue
            try:
                # expression returns True => eliminate
                if eval(flt["code"], {}, ctx):
                    eliminate = True
                    break
            except Exception:
                # ignore bad filters
                pass
        if not eliminate:
            survivors.append(combo)
    return survivors

def initial_cut_counts(combos: List[str],
                       filters: List[Dict],
                       base_ctx_kwargs: Dict) -> Dict[str, int]:
    counts = {f["id"]: 0 for f in filters}
    for combo in combos:
        ctx = build_ctx(combo=combo, **base_ctx_kwargs)
        for flt in filters:
            code = flt["code"]
            if code is None:
                continue
            try:
                if eval(code, {}, ctx):
                    counts[flt["id"]] += 1
            except Exception:
                pass
    return counts

def apply_manual_filters_sequential(combos: List[str],
                                    filters: List[Dict],
                                    selected_ids: List[str],
                                    base_ctx_kwargs: Dict) -> Tuple[List[str], Dict[str, int]]:
    """Apply only the selected filters, in the shown order; return survivors and dynamic cut counts."""
    pool = list(combos)
    dyn = {fid: 0 for fid in selected_ids}
    for flt in filters:
        fid = flt["id"]
        if fid not in selected_ids:
            continue
        code = flt["code"]
        if code is None:
            continue
        new_pool = []
        for combo in pool:
            ctx = build_ctx(combo=combo, **base_ctx_kwargs)
            eliminate = False
            try:
                eliminate = bool(eval(code, {}, ctx))
            except Exception:
                eliminate = False
            if eliminate:
                dyn[fid] += 1
            else:
                new_pool.append(combo)
        pool = new_pool
    return pool, dyn

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Tens — Manual Filter Runner", layout="wide")
    st.title("🎯 Tens — Manual Filter Runner")

    # Sidebar — inputs
    st.sidebar.header("Inputs")

    seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "")
    prev2 = st.sidebar.text_input("Draw 2-back (optional):", "")
    prev3 = st.sidebar.text_input("Draw 3-back (optional):", "")
    prev4 = st.sidebar.text_input("Draw 4-back (optional):", "")
    prev5 = st.sidebar.text_input("Draw 5-back (optional):", "")
    prev6 = st.sidebar.text_input("Draw 6-back (optional):", "")

    method = st.sidebar.selectbox("Generation method:", ["1-digit", "2-digit pair"])

    hot_override = st.sidebar.text_input("Hot digits (comma-separated 0–6, overrides auto):", "")
    cold_override = st.sidebar.text_input("Cold digits (comma-separated 0–6, overrides auto):", "")
    due_override = st.sidebar.text_input("Due digits (comma-separated 0–6, overrides auto):", "")

    st.sidebar.subheader("Tracking")
    track_text = st.sidebar.text_area("Track/Test combos (newline or comma-sep):", "")
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    st.sidebar.subheader("Options")
    default_selected = st.sidebar.checkbox("Default to selected when shown", value=True)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=False)

    # Validate seed
    if len(seed) != 5 or any(ch not in DIGITS for ch in seed):
        st.warning("Enter a valid Draw 1-back: exactly 5 digits in 0–6.")
        return

    # Auto hot/cold/due (from up to 6 draws), with manual overrides
    auto_hot, auto_cold, auto_due = hot_cold_due_from_draws([seed, prev2, prev3, prev4, prev5, prev6])
    hot = _int_list_from_csv(hot_override, 0, 6) or auto_hot
    cold = _int_list_from_csv(cold_override, 0, 6) or auto_cold
    due = _int_list_from_csv(due_override, 0, 6) or auto_due

    st.sidebar.markdown(
        f"**Auto ➜** Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}<br/>"
        f"**Using ➜** Hot {hot} | Cold {cold} | Due {due}",
        unsafe_allow_html=True
    )

    # Generate full raw (w/ duplicates) then apply percentile, then dedup
    raw = generate_combos(seed, method)
    base_ctx_kwargs = dict(seed=seed, prev2=prev2, prev3=prev3, prev4=prev4, prev5=prev5, prev6=prev6,
                           hot=hot, cold=cold, due=due)

    zone_filters = load_filter_csv(PERCENTILE_FILTER_CSV)
    kept_raw = apply_percentile_pre_dedup(raw, zone_filters, base_ctx_kwargs)
    unique_baseline = sorted(set(kept_raw))

    # Track/test normalization
    tracked = []
    if _safe(track_text):
        toks = track_text.replace(",", " ").split()
        tracked = ["".join(sorted(t.strip())) for t in toks if set(t.strip()).issubset(set(DIGITS)) and len(t.strip()) == 5]

    # Inject if requested
    if inject_tracked:
        unique_baseline = sorted(set(unique_baseline) | set(tracked))

    st.caption(
        f"Generated raw: **{len(raw)}** • After zone pre-dedup keep: **{len(kept_raw)}** • "
        f"Unique baseline: **{len(unique_baseline)}**"
    )

    # Load manual filters
    manual_filters = load_filter_csv(MANUAL_FILTER_CSV)

    # Initial cut counts (vs. baseline)
    init_counts = initial_cut_counts(unique_baseline, manual_filters, base_ctx_kwargs)

    # Sort by aggressiveness (desc), then by id
    ordered = sorted(
        manual_filters,
        key=lambda f: (-init_counts.get(f["id"], 0), f["id"])
    )
    if hide_zero:
        display_filters = [f for f in ordered if init_counts.get(f["id"], 0) > 0]
    else:
        display_filters = ordered

    # ------- UI: Manual filters with live application -------
    st.subheader("🛠 Manual Filters")
    selected_ids = []
    cols = st.columns([1, 3, 2, 2])
    with cols[0]:
        st.markdown("**Use**")
    with cols[1]:
        st.markdown("**Filter**")
    with cols[2]:
        st.markdown("**hist**")
    with cols[3]:
        st.markdown("**init cuts**")

    for f in display_filters:
        fid = f["id"]
        hist = f.get("hist", "")
        init_cut = init_counts.get(fid, 0)
        label = f"{fid}: {f.get('layman','').strip() or f.get('expr_text','')}"
        c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
        with c1:
            checked = st.checkbox("", key=f"chk_{fid}", value=default_selected)
        with c2:
            st.write(label)
        with c3:
            st.write(hist or "—")
        with c4:
            st.write(init_cut)
        if checked:
            selected_ids.append(fid)

    # Apply selected filters sequentially; preserve tracked if asked
    survivors_seq, dyn_counts = apply_manual_filters_sequential(unique_baseline, manual_filters, selected_ids, base_ctx_kwargs)

    if preserve_tracked and tracked:
        # Put back tracked combos that might have been cut
        surv_set = set(survivors_seq)
        for t in tracked:
            if t not in surv_set and t in unique_baseline:
                survivors_seq.append(t)

    # ------- Remaining count + tracked audit -------
    st.subheader(f"Remaining after filters: {len(survivors_seq)}")

    if tracked:
        st.markdown("### 🔎 Tracked Combos — Quick Check")
        aud = []
        base_set = set(unique_baseline)
        surv_set = set(survivors_seq)
        for t in tracked:
            aud.append({
                "combo": t,
                "in_generated": t in base_set,
                "survived": t in surv_set,
            })
        st.dataframe(pd.DataFrame(aud), use_container_width=True, height=200)

    # ------- Survivors display + downloads -------
    with st.expander("Show survivors"):
        if survivors_seq:
            st.write("\n".join(survivors_seq))
        else:
            st.write("— none —")

    if survivors_seq:
        df_out = pd.DataFrame({"tens_combo": survivors_seq})
        st.download_button("Download survivors (CSV)", df_out.to_csv(index=False), file_name="pb_tens_survivors.csv", mime="text/csv")
        st.download_button("Download survivors (TXT)", "\n".join(survivors_seq), file_name="pb_tens_survivors.txt", mime="text/plain")


if __name__ == "__main__":
    main()
