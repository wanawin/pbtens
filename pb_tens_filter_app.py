# pb_tens_filter_app.py
import os
import csv
from itertools import product
from collections import Counter
import pandas as pd
import streamlit as st

# -----------------------------------
# Config / filenames
# -----------------------------------
MANUAL_FILTER_CSV = "pb_tens_filters_adapted.csv"
PERCENTILE_FILTER_CSV = "pb_tens_percentile_filters.csv"
TENS_DIGITS = "0123456"  # 0..6

# -----------------------------------
# Utilities
# -----------------------------------
def _safe(s):
    return (s or "").strip()

def _int_list_csv(text):
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if tok.isdigit():
            v = int(tok)
            if 0 <= v <= 6:
                out.append(v)
    return out

def _compile_row_expr(row, is_percentile):
    fid = _safe(row.get("id")) or _safe(row.get("filter_id")) or ""
    layman = _safe(row.get("layman")) or _safe(row.get("layman_explanation"))
    hist = _safe(row.get("stat")) or _safe(row.get("hist"))
    expr = _safe(row.get("expression")) or _safe(row.get("expr"))

    if not fid:
        fid = f"row{_compile_row_expr._idx}"
    _compile_row_expr._idx += 1

    if not expr:
        return {
            "id": fid, "layman": layman, "hist": hist,
            "expr_text": expr, "expr_code": None,
            "compile_error": "empty expression",
            "is_percentile": is_percentile
        }

    # compile expression
    try:
        code = compile(expr, f"<expr:{fid}>", "eval")
        err = None
    except SyntaxError as e:
        code, err = None, str(e)

    return {
        "id": fid, "layman": layman, "hist": hist,
        "expr_text": expr, "expr_code": code,
        "compile_error": err,
        "is_percentile": is_percentile
    }
_compile_row_expr._idx = 1

def load_filters():
    """Load both CSVs; keep track of which rows are percentile vs manual."""
    filters = []
    for path, flag in [
        (PERCENTILE_FILTER_CSV, True),
        (MANUAL_FILTER_CSV, False),
    ]:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                filters.append(_compile_row_expr(row, is_percentile=flag))
    return filters

def generate_tens_combos(seed: str, method: str):
    """Return (raw_with_dups, unique_sorted) of 5-digit strings in 0..6."""
    raw = []
    uniq = set()

    seed = "".join(sorted(seed))
    if method == "1-digit":
        for d in seed:
            for p in product(TENS_DIGITS, repeat=4):
                k = "".join(sorted(d + "".join(p)))
                raw.append(k); uniq.add(k)
    else:  # 2-digit pair
        pairs = { "".join(sorted((seed[i], seed[j])))
                  for i in range(len(seed)) for j in range(i+1, len(seed)) }
        for pair in pairs:
            for p in product(TENS_DIGITS, repeat=3):
                k = "".join(sorted(pair + "".join(p)))
                raw.append(k); uniq.add(k)
    return raw, sorted(uniq)

def tens_ctx(seed: str,
             hot: list[int], cold: list[int], due: list[int]) -> dict:
    """Base context shared by all evals. We expose many aliases your CSVs use."""
    seed_digits = [int(c) for c in seed]
    # Historically some filters used "seed" to mean the sum.
    seed_sum = sum(seed_digits)

    return {
        # Seed info
        "seed": seed_sum,                    # numeric seed "value" (sum)
        "seed_digits": seed_digits,         # list of ints
        "seed_value_digits": seed_digits,   # alias some rows used
        "last2": set(seed_digits),          # simple alias if ever referenced

        # Hot/Cold/Due
        "hot_digits": hot,
        "cold_digits": cold,
        "due_digits": due,

        # Tools available
        "Counter": Counter,
        "set": set,
        "sum": sum,
        "max": max,
        "min": min,
        "len": len,
        "any": any,
        "all": all,
    }

def add_combo_aliases(ctx_base: dict, combo: str) -> dict:
    """Augment context for a particular combo with lots of convenient aliases."""
    digits = [int(c) for c in combo]
    unique = set(digits)
    # An interpretable "winner" for tens-only:
    # We'll treat "winner" as the sum of tens digits (common in your CSVs),
    # and "winner_structure" as the count of unique tens digits (5=all different, etc.)
    winner_value = sum(digits)
    winner_structure = len(unique)

    # Also provide direct aliases many rows used:
    # - combo_digits
    # - combo_tens
    # - cdigits
    # - winner (numeric)
    # - winner_value (same)
    # - winner_structure (unique count)
    # - tens_sum
    ctx = dict(ctx_base)
    ctx.update({
        "combo": combo,
        "combo_digits": digits,
        "combo_tens": digits,
        "cdigits": digits,

        "tens_sum": winner_value,
        "winner": winner_value,
        "winner_value": winner_value,
        "winner_structure": winner_structure,

        # Frequently useful lambdas that appear in expressions
        "digits_from": lambda xs: [int(x) for x in xs],
    })
    return ctx

def apply_filter_to_pool(pool: list[str], flt: dict, ctx_base: dict):
    """Return (survivors, cut_count)."""
    if not flt.get("expr_code"):
        return list(pool), 0

    survivors = []
    cut = 0
    for combo in pool:
        ctx = add_combo_aliases(ctx_base, combo)
        try:
            if eval(flt["expr_code"], {}, ctx):
                cut += 1
            else:
                survivors.append(combo)
        except Exception:
            # if a row errors, keep the combo (fail-open) and don't count a cut
            survivors.append(combo)
    return survivors, cut

def auto_hot_cold_due(draws: list[str]):
    """Compute hot/cold/due from the last 6 draws if provided."""
    last6 = [d for d in draws if d][:6]
    if len(last6) < 6:
        return [], [], []
    text = "".join(last6)
    cnt = Counter(text)
    hot = [int(x) for x, _ in cnt.most_common(3)]
    cold = [int(x) for x, _ in cnt.most_common()[-3:]]
    due = [d for d in range(7) if str(d) not in text]
    return hot, cold, due

# -----------------------------------
# App
# -----------------------------------
def main():
    st.title("🎯 Powerball Tens Filter App")

    # Inputs (6 seeds)
    seed  = st.sidebar.text_input("Draw 1-back (required):", "").strip()
    s2    = st.sidebar.text_input("Draw 2-back:", "").strip()
    s3    = st.sidebar.text_input("Draw 3-back:", "").strip()
    s4    = st.sidebar.text_input("Draw 4-back:", "").strip()
    s5    = st.sidebar.text_input("Draw 5-back:", "").strip()
    s6    = st.sidebar.text_input("Draw 6-back:", "").strip()

    method = st.sidebar.selectbox("Generation method:", ["1-digit", "2-digit pair"])

    hot_override  = st.sidebar.text_input("Hot digits (override, 0–6 comma):", "")
    cold_override = st.sidebar.text_input("Cold digits (override, 0–6 comma):", "")
    due_override  = st.sidebar.text_input("Due digits (override, 0–6 comma):", "")

    # Tracked combo & status area
    tracked = st.sidebar.text_input("Track/Test combo (e.g., 00123):", "").strip()

    # Controls
    select_all = st.sidebar.checkbox("Select/Deselect all filters (shown)", value=False)
    hide_zero  = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=False)

    # Validate seed
    if len(seed) != 5 or any(c not in TENS_DIGITS for c in seed):
        st.info("Enter a 5-digit tens seed using digits 0–6 (e.g., 23345).")
        return

    # Hot/Cold/Due
    auto_hot, auto_cold, auto_due = auto_hot_cold_due([seed, s2, s3, s4, s5, s6])
    hot  = _int_list_csv(hot_override)  or auto_hot
    cold = _int_list_csv(cold_override) or auto_cold
    due  = _int_list_csv(due_override)  or auto_due
    st.sidebar.caption(f"Auto → Hot {auto_hot} | Cold {auto_cold} | Due {auto_due}")
    st.sidebar.caption(f"Using → Hot {hot} | Cold {cold} | Due {due}")

    # Generate pool
    raw, unique = generate_tens_combos(seed, method)

    # Load filters
    rows = load_filters()
    percentiles = [r for r in rows if r["is_percentile"]]
    manuals     = [r for r in rows if not r["is_percentile"]]

    # Base ctx
    base_ctx = tens_ctx(seed, hot, cold, due)

    # Phase A: percentile filters pre-dedup (keep in-zone only)
    pool = list(raw)
    for z in percentiles:
        pool, _ = apply_filter_to_pool(pool, z, base_ctx)

    # Phase B: dedup
    pool = sorted(set(pool))

    # Sidebar pipeline summary
    st.sidebar.markdown(
        f"**Pipeline**  \n"
        f"- Raw generated: **{len(raw)}**  \n"
        f"- Survive percentile pre-dedup: **{len(pool)}**  \n"
        f"- Unique enumeration: **{len(unique)}**"
    )

    # Manual filters — compute initial cuts (on the deduped pool)
    init_cuts = {}
    for flt in manuals:
        _, cut = apply_filter_to_pool(pool, flt, base_ctx)
        init_cuts[flt["id"]] = cut

    # Sort by aggressiveness
    ordered = sorted(manuals, key=lambda f: -init_cuts.get(f["id"], 0))
    shown_filters = [f for f in ordered if init_cuts.get(f["id"], 0) > 0] if hide_zero else ordered

    st.header("🛠 Manual Filters")
    st.markdown(f"Applicable filters: **{len(shown_filters)}**")

    # Now apply interactively in the order displayed
    survivors = list(pool)
    eliminated_by = None
    # “initial/eliminated/remaining” running tallies
    running_initial = len(pool)
    running_eliminated = 0

    for flt in shown_filters:
        label = f"{flt['id']}: {flt['layman']} | hist {flt['hist']} | cut {init_cuts.get(flt['id'],0)}"
        checked = st.checkbox(label, value=select_all, key=f"flt_{flt['id']}")
        if checked:
            survivors, cut = apply_filter_to_pool(survivors, flt, base_ctx)
            running_eliminated += cut
            # tracked combo live update
            if tracked and eliminated_by is None:
                tn = "".join(sorted(tracked))
                if tn not in survivors and tn in pool:
                    eliminated_by = flt['id']
        # show counts after this row (as requested: no big log table)
        st.markdown(f"Remaining: **{len(survivors)}**")

    # Tracked combo status (under its input)
    if tracked:
        tn = "".join(sorted(tracked))
        if tn not in unique:
            st.sidebar.error("Tracked combo was NOT generated.")
        elif eliminated_by:
            st.sidebar.error(f"Tracked combo eliminated by **{eliminated_by}**.")
        elif tn in survivors:
            st.sidebar.success("Tracked combo **survived** all filters.")
        else:
            st.sidebar.warning("Tracked combo was eliminated.")

    # Final survivors + downloads
    st.subheader(f"✅ Final Survivors: {len(survivors)}")
    with st.expander("Show survivors"):
        st.write(survivors)

    if survivors:
        df = pd.DataFrame({"tens_combo": survivors})
        st.download_button(
            "Download survivors (CSV)",
            df.to_csv(index=False),
            file_name="pb_tens_survivors.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download survivors (TXT)",
            "\n".join(survivors),
            file_name="pb_tens_survivors.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
