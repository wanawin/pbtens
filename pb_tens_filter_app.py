import os
import csv
from itertools import product
from collections import Counter

import pandas as pd
import streamlit as st

# =========================================================
# Config
# =========================================================
MANUAL_FILTER_CSV     = "pb_tens_filters_adapted.csv"
PERCENTILE_FILTER_CSV = "pb_tens_percentile_filters.csv"
DIGITS = "0123456"   # tens domain 0..6 only

# V-TRAC (mod 5 buckets) and mirror map (full 0..9 for safety)
V_TRAC_GROUPS = {0:1, 5:1, 1:2, 6:2, 2:3, 7:3, 3:4, 8:4, 4:5, 9:5}
MIRROR = {0:5, 5:0, 1:6, 6:1, 2:7, 7:2, 3:8, 8:3, 4:9, 9:4}

# =========================================================
# IO helpers
# =========================================================
def load_filter_csv(path: str) -> list[dict]:
    """Load a filter CSV (id, layman/name, expression, applicable_if, hist/stat, enabled)."""
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for i, raw in enumerate(rdr):
            row = { (k or "").strip().lower(): (v if isinstance(v, str) else v)
                    for k, v in raw.items() }

            fid     = (row.get("id") or row.get("fid") or f"row{i+1}").strip()
            layman  = (row.get("layman") or row.get("layman_explanation") or row.get("name") or "").strip()
            hist    = (row.get("hist") or row.get("stat") or row.get("history") or "").strip()
            expr    = (row.get("expression") or row.get("expr") or "").strip()
            app_if  = (row.get("applicable_if") or "True").strip()
            enabled = (row.get("enabled") or "").strip().lower() == "true"

            # compile (store text too for debugging)
            expr_code = None
            app_code  = None
            expr_err  = None
            app_err   = None
            try:
                app_code = compile(app_if, f"<applicable:{fid}>", "eval")
            except Exception as e:
                app_err = str(e)
            try:
                expr_code = compile(expr, f"<expression:{fid}>", "eval") if expr else None
            except Exception as e:
                expr_err = str(e)

            out.append({
                "id": fid,
                "layman": layman,
                "hist": hist,
                "expression": expr,
                "applicable_if": app_if,
                "expr_code": expr_code,
                "app_code": app_code,
                "expr_err": expr_err,
                "app_err": app_err,
                "enabled_default": enabled,
            })
    return out

# =========================================================
# Generation & context
# =========================================================
def generate_tens(seed: str, method: str) -> list[str]:
    """Generate unique sorted-5-digit combos (0..6) via 1-digit or 2-digit-pair method."""
    combos = set()
    s = "".join(sorted(seed))
    if method == "1-digit":
        for d in s:
            for p in product(DIGITS, repeat=4):
                combos.add("".join(sorted(d + "".join(p))))
    else:  # 2-digit pair
        pairs = { "".join(sorted((s[i], s[j])))
                  for i in range(len(s)) for j in range(i+1, len(s)) }
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                combos.add("".join(sorted(pair + "".join(p))))
    return sorted(combos)

def sum_category(total: int) -> str:
    # Category scheme used in your Pick-5 app (kept for compatibility)
    if 0 <= total <= 15:
        return "Very Low"
    elif 16 <= total <= 24:
        return "Low"
    elif 25 <= total <= 33:
        return "Mid"
    else:
        return "High"

def build_ctx(seed1: str, seed2: str, seed3: str, seed4: str, seed5: str, seed6: str,
              hot_digits: list[int], cold_digits: list[int], due_digits: list[int]):
    """
    Pick-5 compatible variable names (tuned for tens-only domain).
    - seed_digits               : Draw 1-back as list[int]
    - prev_seed_digits          : Draw 2-back
    - prev_prev_seed_digits     : Draw 3-back
    - (we also expose last2/common_to_both, seed_vtracs, etc.)
    """
    def digs(s): return [int(c) for c in s] if s else []

    seed_digits       = digs(seed1)
    prev_seed_digits  = digs(seed2)
    prev_prev_digits  = digs(seed3)
    prev3_digits      = digs(seed4)  # additional; not used by legacy filters, but available
    prev4_digits      = digs(seed5)
    prev5_digits      = digs(seed6)

    new_digits = set(seed_digits) - set(prev_seed_digits)

    # Build prev_pattern like your Pick-5 (sum_category + parity) for 3 most recent
    prev_pattern = []
    for arr in (prev_prev_digits, prev_seed_digits, seed_digits):
        s = sum(arr)
        prev_pattern.extend([sum_category(s), ("Even" if s % 2 == 0 else "Odd")])
    prev_pattern = tuple(prev_pattern)

    seed_counts = Counter(seed_digits)
    seed_sum    = sum(seed_digits)
    prev_sum_cat = sum_category(seed_sum)

    seed_vtracs = set(V_TRAC_GROUPS.get(d, d % 5) for d in seed_digits)

    base = {
        "seed_digits": seed_digits,
        "prev_seed_digits": prev_seed_digits,
        "prev_prev_seed_digits": prev_prev_digits,
        # extra prior draws (if any filters want them)
        "prev3_seed_digits": prev3_digits,
        "prev4_seed_digits": prev4_digits,
        "prev5_seed_digits": prev5_digits,

        "new_seed_digits": new_digits,
        "prev_pattern": prev_pattern,

        "hot_digits": hot_digits,
        "cold_digits": cold_digits,
        "due_digits": due_digits,

        "seed_counts": seed_counts,
        "seed_sum": seed_sum,
        "prev_sum_cat": prev_sum_cat,

        "seed_vtracs": seed_vtracs,
        "mirror": MIRROR,

        "common_to_both": set(seed_digits) & set(prev_seed_digits),   # common to Draw 1&2
        "last2": set(seed_digits) | set(prev_seed_digits),            # union Draw 1&2

        "Counter": Counter,
        "sum_category": sum_category,
        "V_TRAC_GROUPS": V_TRAC_GROUPS
    }
    return base

def make_combo_ctx(base_ctx: dict, combo_str: str) -> dict:
    cdigits = [int(c) for c in combo_str]
    csum = sum(cdigits)
    ctx = dict(base_ctx)  # shallow copy
    ctx.update({
        "combo_digits": cdigits,
        "combo_sum": csum,
        "combo_sum_cat": sum_category(csum),
        "combo_vtracs": set(V_TRAC_GROUPS.get(d, d % 5) for d in cdigits),
    })
    return ctx

# =========================================================
# Filtering
# =========================================================
def apply_filter_to_combo(flt: dict, ctx: dict) -> bool:
    """
    Return True if the filter ELIMINATES the combo (i.e., expression is True).
    The CSV convention is the Pick-5 one: if expr evaluates True -> eliminate.
    """
    if flt["app_err"] or flt["expr_err"]:
        return False
    try:
        if not flt["app_code"] or not bool(eval(flt["app_code"], {}, ctx)):
            return False
        return bool(eval(flt["expr_code"], {}, ctx))
    except Exception:
        return False

def initial_cut_counts(combos: list[str], filters: list[dict], base_ctx: dict) -> dict[str, int]:
    counts = {f["id"]: 0 for f in filters}
    for combo in combos:
        cctx = make_combo_ctx(base_ctx, combo)
        for flt in filters:
            if apply_filter_to_combo(flt, cctx):
                counts[flt["id"]] += 1
    return counts

def apply_filters_sequential(combos: list[str], filters: list[dict], base_ctx: dict,
                             active_map: dict[str, bool],
                             preserve_tracked: bool, tracked_set: set[str]) -> tuple[list[str], dict]:
    """
    Apply active filters in UI order, return survivors and dynamic per-filter cut counts.
    """
    pool = list(combos)
    dyn = {f["id"]: 0 for f in filters}

    for flt in filters:
        if not active_map.get(flt["id"], False):
            continue
        next_pool = []
        for combo in pool:
            cctx = make_combo_ctx(base_ctx, combo)
            eliminate = apply_filter_to_combo(flt, cctx)
            if eliminate and (combo in tracked_set) and preserve_tracked:
                # keep it, but it "would have been eliminated"
                next_pool.append(combo)
            elif eliminate:
                dyn[flt["id"]] += 1
            else:
                next_pool.append(combo)
        pool = next_pool
    return pool, dyn

def apply_percentile_filters(raw_combos: list[str], zone_filters: list[dict], base_ctx: dict]) -> list[str]:
    """
    Apply percentile/zone filters BEFORE dedup & manual list.
    Zone filters also use 'True -> eliminate' convention.
    """
    if not zone_filters:
        return list(raw_combos)

    kept = []
    for combo in raw_combos:
        cctx = make_combo_ctx(base_ctx, combo)
        eliminate = False
        for flt in zone_filters:
            if apply_filter_to_combo(flt, cctx):
                eliminate = True
                break
        if not eliminate:
            kept.append(combo)
    return kept

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="Tens — Manual Filter Runner", layout="wide")
    st.title("🎯 Tens — Manual Filter Runner")

    # -----------------------
    # Sidebar: Inputs
    # -----------------------
    st.sidebar.header("Inputs")

    seed1 = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", value="").strip()
    seed2 = st.sidebar.text_input("Draw 2-back (optional):", value="").strip()
    seed3 = st.sidebar.text_input("Draw 3-back (optional):", value="").strip()
    seed4 = st.sidebar.text_input("Draw 4-back (optional):", value="").strip()
    seed5 = st.sidebar.text_input("Draw 5-back (optional):", value="").strip()
    seed6 = st.sidebar.text_input("Draw 6-back (optional):", value="").strip()

    method = st.sidebar.selectbox("Generation method:", ["1-digit", "2-digit pair"])

    st.sidebar.markdown("---")
    auto_hc = st.sidebar.checkbox("Auto-calc Hot/Cold/Due from last 6", value=True)
    hot_manual = st.sidebar.text_input("Hot digits (comma-separated 0–6, optional):", value="")
    cold_manual = st.sidebar.text_input("Cold digits (comma-separated 0–6, optional):", value="")
    due_manual = st.sidebar.text_input("Due digits (comma-separated 0–6, optional):", value="")
    disable_due_when_empty = st.sidebar.checkbox("Disable due-based filters when due set is empty", value=True)

    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test combos (e.g., 00123, 23345; newline or comma-separated):", height=100)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos if not generated", value=False)

    st.sidebar.markdown("---")
    select_all = st.sidebar.checkbox("Select/Deselect All (shown)", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)

    # Validate seed1
    def is_tens5(s: str) -> bool:
        return len(s) == 5 and all(c in DIGITS for c in s)

    if not is_tens5(seed1):
        st.info("Enter a valid **Draw 1-back**: exactly 5 digits in 0–6.")
        return

    # Hot/Cold/Due
    def to_list(s: str) -> list[int]:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 0 <= v <= 6:
                    out.append(v)
        return out

    if auto_hc:
        all6 = [seed1, seed2, seed3, seed4, seed5, seed6]
        smushed = "".join(s for s in all6 if s)
        freq = Counter(smushed)
        # top3 hot; bottom3 cold; due = not appeared in any of the last 6 draws
        hot_auto  = [int(d) for d, _ in freq.most_common(3)]
        cold_auto = [int(d) for d, _ in freq.most_common()[-3:]] if freq else []
        due_auto  = [d for d in range(7) if str(d) not in smushed]
        hot_digits  = to_list(hot_manual)  or hot_auto
        cold_digits = to_list(cold_manual) or cold_auto
        due_digits  = to_list(due_manual)  or due_auto
    else:
        hot_digits  = to_list(hot_manual)
        cold_digits = to_list(cold_manual)
        due_digits  = to_list(due_manual)

    st.sidebar.caption(f"Auto Hot: {hot_digits} | Auto Cold: {cold_digits} | Auto Due: {due_digits}")

    # Build base ctx
    base_ctx = build_ctx(seed1, seed2, seed3, seed4, seed5, seed6, hot_digits, cold_digits, due_digits)

    # Generate raw & apply percentile first
    raw = []
    if method == "1-digit":
        for d in seed1:
            for p in product(DIGITS, repeat=4):
                raw.append("".join(sorted(d + "".join(p))))
    else:
        s = "".join(sorted(seed1))
        pairs = { "".join(sorted((s[i], s[j])))
                  for i in range(len(s)) for j in range(i+1, len(s)) }
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                raw.append("".join(sorted(pair + "".join(p))))

    # Load filters
    manual_filters     = load_filter_csv(MANUAL_FILTER_CSV)
    percentile_filters = load_filter_csv(PERCENTILE_FILTER_CSV)

    # Optionally disable due-based filters when due empty
    if disable_due_when_empty and not due_digits:
        def is_due_based(f):
            txt = (f.get("expression") or "") + " " + (f.get("applicable_if") or "")
            return "due_digits" in txt
        manual_filters     = [f for f in manual_filters if not is_due_based(f)]
        percentile_filters = [f for f in percentile_filters if not is_due_based(f)]

    pre_zone = apply_percentile_filters(raw, percentile_filters, base_ctx)
    unique_baseline = sorted(set(pre_zone))

    # Track combos
    track_norm = []
    if track_text.strip():
        tokens = track_text.replace(",", " ").split()
        for t in tokens:
            t = "".join(sorted("".join(ch for ch in t if ch.isdigit())))
            if len(t) == 5 and all(ch in DIGITS for ch in t):
                track_norm.append(t)
    tracked_set = set(track_norm)

    if inject_tracked:
        unique_baseline = sorted(set(unique_baseline) | tracked_set)

    # Initial cut counts for manual filters
    init_counts = initial_cut_counts(unique_baseline, manual_filters, base_ctx)

    # Sort (aggressive first, zero-cuts last)
    sorted_filters = sorted(
        manual_filters,
        key=lambda f: (init_counts.get(f["id"], 0) == 0, -init_counts.get(f["id"], 0), f["id"])
    )
    display_filters = [f for f in sorted_filters if init_counts.get(f["id"], 0) > 0] if hide_zero else sorted_filters

    # -----------------------------------------
    # UI: Manual Filters list (main column)
    # -----------------------------------------
    st.header("🛠 Manual Filters")
    colL, colR = st.columns([3, 2], gap="large")

    with colL:
        active_map: dict[str, bool] = {}
        for flt in display_filters:
            init_cut = init_counts.get(flt["id"], 0)
            meta = []
            if flt["hist"]:
                meta.append(f"hist {flt['hist']}")
            meta.append(f"init cut {init_cut}")
            label = f"{flt['id']}: {flt['layman']} — " + " · ".join(meta)
            active = st.checkbox(
                label,
                key=f"chk_{flt['id']}",
                value=select_all and (flt.get("enabled_default", False) or True)
            )
            active_map[flt["id"]] = active
        st.caption(f"{len(display_filters)} filters shown (of {len(manual_filters)} total).")

    # -----------------------------------------
    # Apply selected filters (sequential)
    # -----------------------------------------
    survivors, dynamic_counts = apply_filters_sequential(
        unique_baseline, display_filters, base_ctx,
        active_map, preserve_tracked, tracked_set
    )

    # -----------------------------------------
    # Audit (right)
    # -----------------------------------------
    with colR:
        st.subheader(f"Remaining after filters: {len(survivors)}")
        with st.expander("Show survivors", expanded=False):
            for c in survivors:
                st.write(c)

        # Downloads
        if survivors:
            df_surv = pd.DataFrame({"tens_combo": survivors})
            st.download_button("Download survivors (CSV)", df_surv.to_csv(index=False), file_name="pb_tens_survivors.csv", mime="text/csv")
            st.download_button("Download survivors (TXT)", "\n".join(survivors), file_name="pb_tens_survivors.txt", mime="text/plain")

        # Small table of dynamic cuts (for visible filters)
        if display_filters:
            rows = []
            for flt in display_filters:
                rows.append({
                    "id": flt["id"],
                    "layman": flt["layman"],
                    "hist": flt["hist"],
                    "init_cut": init_counts.get(flt["id"], 0),
                    "dyn_cut": dynamic_counts.get(flt["id"], 0),
                })
            st.markdown("**Cuts summary (shown filters):**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(420, 70 + 28*len(rows)))

    # -----------------------------------------
    # Combo tracker audit
    # -----------------------------------------
    if track_norm:
        st.markdown("---")
        st.subheader("🔎 Tracked Combos — Quick Check")
        surv_set = set(survivors)
        audit_rows = []
        for t in track_norm:
            audit_rows.append({
                "combo": t,
                "in_generated": (t in unique_baseline),
                "survived": (t in surv_set),
            })
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, height=220)


if __name__ == "__main__":
    main()
