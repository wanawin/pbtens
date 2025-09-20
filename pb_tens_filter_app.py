import streamlit as st
import pandas as pd
import csv, os, re
from itertools import product
from collections import Counter

APP_TITLE = "Powerball Tens Filter App"
FILTER_CSV = "pb_tens_filters_adapted.csv"   # tens-only filter bank (no variants)
TENS_DOMAIN = "0123456"                      # tens digits only

# -------------------------------
# Utility / context helpers
# -------------------------------
def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return "Very Low"
    elif 16 <= total <= 24:
        return "Low"
    elif 25 <= total <= 33:
        return "Mid"
    else:
        return "High"

def safe_id(s: str, fallback: str) -> str:
    s = (s or "").strip()
    if not s:
        return fallback
    # keep only simple key chars to avoid Streamlit key collisions
    s2 = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s2 or fallback

def compute_hot_cold_from_six(draws_1_to_6: list[str]) -> tuple[list[int], list[int]]:
    """
    Expects a list of SIX 5-digit strings (each char '0'..'6'), newest first.
    Returns (hot, cold) as lists of ints; empty lists if unavailable.
    """
    if len(draws_1_to_6) != 6:
        return [], []
    try:
        digits = [int(c) for d in draws_1_to_6 for c in d.strip() if c in TENS_DOMAIN]
    except Exception:
        return [], []
    if len(digits) != 30:  # 6 draws * 5 tens digits
        return [], []
    cnt = Counter(digits)
    if not cnt:
        return [], []
    hot = [d for d, _ in cnt.most_common(3)]
    cold = [d for d, _ in cnt.most_common()[-3:]]
    return hot, cold

def build_ctx(combo_str: str, hot_digits: list[int], cold_digits: list[int]) -> dict:
    d = [int(c) for c in combo_str]
    tens_even = sum(1 for x in d if x % 2 == 0)
    tens_odd  = 5 - tens_even
    tens_unique = len(set(d))
    tens_range = max(d) - min(d)
    tens_low = sum(1 for x in d if x in (0,1,2,3,4))
    tens_high = sum(1 for x in d if x in (5,6))
    total = sum(d)
    return {
        "combo_digits": d,
        "combo_sum": total,
        "combo_sum_cat": sum_category(total),
        "tens_even_count": tens_even,
        "tens_odd_count": tens_odd,
        "tens_unique_count": tens_unique,
        "tens_range": tens_range,
        "tens_low_count": tens_low,
        "tens_high_count": tens_high,
        "hot_digits": list(hot_digits or []),
        "cold_digits": list(cold_digits or []),
        "Counter": Counter,
    }

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data(show_spinner=False)
def load_filters(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    filters: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for idx, raw in enumerate(rdr):
            row = { (k or "").lower().strip(): (v if isinstance(v,str) else v) for k, v in raw.items() }
            fid = safe_id(row.get("id",""), f"row{idx+1}")
            layman = (row.get("layman") or row.get("layman_explanation") or row.get("explanation") or "").strip()
            stat = (row.get("stat") or row.get("hist") or "").strip()
            expr = (row.get("expression") or row.get("expr") or "").strip()
            if not expr:
                # skip empty expressions silently
                continue
            try:
                code = compile(expr, f"<expr:{fid}>", "eval")
            except SyntaxError as e:
                # carry the error text so user can see it inline
                filters.append({
                    "id": fid, "layman": layman, "stat": stat,
                    "expr_code": None, "expr_text": expr,
                    "compile_error": str(e)
                })
                continue
            filters.append({
                "id": fid, "layman": layman, "stat": stat,
                "expr_code": code, "expr_text": expr,
                "compile_error": None
            })
    return filters

# -------------------------------
# Tens pool generation (deduped)
# -------------------------------
def generate_tens_combos(seed: str, method: str) -> list[str]:
    seed_sorted = "".join(sorted(seed))
    out: set[str] = set()
    if method == "1-digit":
        for d in seed_sorted:
            for p in product(TENS_DOMAIN, repeat=4):
                out.add("".join(sorted(d + "".join(p))))
    else:  # "2-digit pair"
        pairs = {
            "".join(sorted((seed_sorted[i], seed_sorted[j])))
            for i in range(len(seed_sorted)) for j in range(i+1, len(seed_sorted))
        }
        for pr in pairs:
            for p in product(TENS_DOMAIN, repeat=3):
                out.add("".join(sorted(pr + "".join(p))))
    return sorted(out)

# -------------------------------
# App
# -------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.sidebar.title(APP_TITLE)

    # Draws
    seed = st.sidebar.text_input("Draw 1-back (required):", value="", help="5 tens digits, each 0–6")
    d2 = st.sidebar.text_input("Draw 2-back (optional):", value="")
    d3 = st.sidebar.text_input("Draw 3-back (optional):", value="")
    d4 = st.sidebar.text_input("Draw 4-back (optional):", value="")
    d5 = st.sidebar.text_input("Draw 5-back (optional):", value="")
    d6 = st.sidebar.text_input("Draw 6-back (optional):", value="")

    method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])

    hot_override = st.sidebar.text_input("Hot digits (comma-separated, overrides auto):", value="")
    cold_override = st.sidebar.text_input("Cold digits (comma-separated, overrides auto):", value="")

    track_combo = st.sidebar.text_input("Track/Test combo (e.g., 03556):", value="")

    # Toggles
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)
    st.sidebar.markdown("---")

    # Validate seed
    if not (len(seed) == 5 and all(c in TENS_DOMAIN for c in seed)):
        st.warning("Enter a valid **Draw 1-back**: exactly 5 digits in 0–6.")
        return

    # Auto Hot/Cold only if *all* draws 1..6 are present
    hot_auto: list[int] = []
    cold_auto: list[int] = []
    draws_1_to_6 = [seed, d2, d3, d4, d5, d6]
    if all(len(x.strip()) == 5 and all(c in TENS_DOMAIN for c in x.strip()) for x in draws_1_to_6):
        hot_auto, cold_auto = compute_hot_cold_from_six(draws_1_to_6)

    def parse_list(s: str) -> list[int]:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok.isdigit():
                v = int(tok)
                if 0 <= v <= 6:
                    out.append(v)
        return out

    hot_digits = parse_list(hot_override) if hot_override.strip() else hot_auto
    cold_digits = parse_list(cold_override) if cold_override.strip() else cold_auto

    st.sidebar.info(f"Hot: {hot_digits or '—'}   |   Cold: {cold_digits or '—'}")

    # Load filters
    try:
        filters = load_filters(FILTER_CSV)
    except FileNotFoundError:
        st.error(f"Filter CSV not found: **{FILTER_CSV}**")
        return

    # Generate combos
    combos = generate_tens_combos(seed, method)

    # Initial cuts per filter (independent scan)
    init_cuts: dict[str, int] = {}
    for i, flt in enumerate(filters):
        fid = flt["id"]
        if flt.get("expr_code") is None:
            init_cuts[fid] = 0
            continue
        c = 0
        for combo in combos:
            ctx = build_ctx(combo, hot_digits, cold_digits)
            try:
                eliminate = bool(eval(flt["expr_code"], ctx, ctx))
            except Exception:
                eliminate = False
            if eliminate:
                c += 1
        init_cuts[fid] = c

    # Sort and prune by zero-cut toggle
    sorted_filters = sorted(filters, key=lambda f: (init_cuts[f["id"]] == 0, -init_cuts[f["id"]]))
    display_filters = [f for f in sorted_filters if not hide_zero or init_cuts[f["id"]] > 0]

    # Bulk select controls for shown list
    cols_top = st.columns([1,1,6,1,1])
    with cols_top[0]:
        if st.button("Select all (shown)"):
            for idx, flt in enumerate(display_filters):
                key = f"flt_{idx}_{safe_id(flt['id'],'row')}"
                st.session_state[key] = True
    with cols_top[1]:
        if st.button("Clear all (shown)"):
            for idx, flt in enumerate(display_filters):
                key = f"flt_{idx}_{safe_id(flt['id'],'row')}"
                st.session_state[key] = False

    st.header("🛠️ Manual Filters")

    # Sequential elimination with unique keys
    pool = list(combos)
    dynamic_cuts: dict[str, int] = {}
    for idx, flt in enumerate(display_filters):
        fid = flt["id"]
        L   = flt.get("layman","").strip() or "(no description)"
        H   = flt.get("stat","").strip()
        ic  = init_cuts.get(fid, 0)
        err = flt.get("compile_error")

        key = f"flt_{idx}_{safe_id(fid,'row')}"   # ✅ unique key per row-instance
        label = f"{fid}: {L} — init cuts {ic}" + (f" • hist {H}" if H else "")

        active = st.checkbox(label, key=key, value=st.session_state.get(key, False), help=flt.get("expr_text",""))
        if err:
            st.caption(f"⚠️ Compile error: {err}")
            dynamic_cuts[fid] = 0
            continue

        if active:
            dc = 0
            survivors = []
            for combo in pool:
                ctx = build_ctx(combo, hot_digits, cold_digits)
                try:
                    eliminate = bool(eval(flt["expr_code"], ctx, ctx))
                except Exception:
                    eliminate = False
                if eliminate:
                    dc += 1
                else:
                    survivors.append(combo)
            dynamic_cuts[fid] = dc
            pool = survivors
        else:
            dynamic_cuts[fid] = 0

    survivors = pool

    # Summary & tracking
    st.subheader(f"Remaining after filters: {len(survivors)} of {len(combos)} generated")

    if track_combo.strip():
        norm = "".join(sorted(track_combo.strip()))
        if norm in survivors:
            st.success(f"Tracked combo {track_combo} **survived**.")
        else:
            st.error(f"Tracked combo {track_combo} **was eliminated**.")

    # Downloads
    df_out = pd.DataFrame({"tens_combo": survivors})
    st.download_button("📥 Download survivors (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                       file_name="pb_tens_survivors.csv", mime="text/csv")
    st.download_button("📥 Download survivors (TXT)", "\n".join(survivors).encode("utf-8"),
                       file_name="pb_tens_survivors.txt", mime="text/plain")

    with st.expander("Show remaining combinations"):
        for c in survivors:
            st.write(c)

    # (Optional) show a compact table of shown filters with live cuts
    if display_filters:
        small = pd.DataFrame([{
            "id": f["id"],
            "layman": f.get("layman",""),
            "hist": f.get("stat",""),
            "init_cuts": init_cuts.get(f["id"],0),
            "dynamic_cuts": dynamic_cuts.get(f["id"],0),
        } for f in display_filters])
        st.caption("Shown filters — initial vs dynamic cuts")
        st.dataframe(small, use_container_width=True, height=240)


if __name__ == "__main__":
    main()
