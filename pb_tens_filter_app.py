# pb_tens_filter_app.py — Tens-only manual filter runner (Pick-5 style UI)

import streamlit as st
import csv
import os
from collections import Counter
from itertools import product
import pandas as pd
import re

# ==============================
# Globals
# ==============================
FILTER_CSV = "pb_tens_filters_adapted.csv"
DIGITS = "0123456"   # tens-only model

# ==============================
# Helpers
# ==============================
def sum_category(total: int) -> str:
    if 0 <= total <= 15:
        return "Very Low"
    elif 16 <= total <= 24:
        return "Low"
    elif 25 <= total <= 33:
        return "Mid"
    else:
        return "High"

def safe_key(s: str, fallback: str) -> str:
    """Produce a Streamlit-safe key from an ID."""
    s = (s or "").strip()
    if not s:
        return fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or fallback

def load_filters(path: str) -> list[dict]:
    """Load filters from CSV and compile expressions (no caching)."""
    if not os.path.exists(path):
        st.error(f"Filter file not found: {path}")
        st.stop()

    filters: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for idx, raw in enumerate(rdr):
            row = { (k or "").lower().strip(): (v if isinstance(v, str) else v)
                    for k, v in raw.items() }

            fid = (row.get("id") or f"row{idx+1}").strip()
            layman = (row.get("layman") or row.get("layman_explanation") or row.get("explanation") or "").strip()
            stat = (row.get("stat") or row.get("hist") or "").strip()
            expr = (row.get("expression") or row.get("expr") or "").strip()

            if not expr:
                # skip empty formula rows silently
                continue

            try:
                code = compile(expr, f"<expr:{fid}>", "eval")
                filters.append({
                    "id": fid,
                    "key": safe_key(fid, f"row{idx+1}"),
                    "layman": layman,
                    "stat": stat,
                    "expr_text": expr,
                    "expr_code": code,
                    "compile_error": None
                })
            except SyntaxError as e:
                filters.append({
                    "id": fid,
                    "key": safe_key(fid, f"row{idx+1}"),
                    "layman": layman,
                    "stat": stat,
                    "expr_text": expr,
                    "expr_code": None,
                    "compile_error": str(e)
                })
    return filters

def generate_combos(seed: str, method: str) -> list[str]:
    """Generate deduped tens combos from seed."""
    combos = set()
    s = "".join(sorted(seed))
    if method == "1-digit":
        for d in s:
            for p in product(DIGITS, repeat=4):
                combos.add("".join(sorted(d + "".join(p))))
    else:  # 2-digit pair
        pairs = { "".join(sorted((s[i], s[j]))) for i in range(len(s)) for j in range(i+1, len(s)) }
        for pair in pairs:
            for p in product(DIGITS, repeat=3):
                combos.add("".join(sorted(pair + "".join(p))))
    return sorted(combos)

def compute_hot_cold_due(draws_1_to_6: list[str]) -> tuple[list[int], list[int], list[int]]:
    """
    Auto-calc hot, cold, due tens digits using exactly 6 draws (1-back..6-back).
    If fewer than 6 valid draws, returns empty lists (no auto).
    """
    ok = [d for d in draws_1_to_6 if len(d.strip()) == 5 and all(c in DIGITS for c in d.strip())]
    if len(ok) != 6:
        return [], [], []
    digits = [int(c) for d in ok for c in d.strip()]
    cnt = Counter(digits)
    if not cnt:
        return [], [], []
    hot = [d for d, _ in cnt.most_common(3)]
    cold = [d for d, _ in cnt.most_common()[-3:]]
    in_all = set(range(7))
    due = sorted(list(in_all - set(digits)))
    return hot, cold, due

def parse_digit_list(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            v = int(tok)
            if 0 <= v <= 6:
                out.append(v)
    return out

def build_ctx(combo_str: str, hot: list[int], cold: list[int], due: list[int]) -> dict:
    d = [int(c) for c in combo_str]
    total = sum(d)
    tens_even = sum(1 for x in d if x % 2 == 0)
    tens_odd  = 5 - tens_even
    tens_unique = len(set(d))
    tens_range = max(d) - min(d)
    tens_low = sum(1 for x in d if x in (0,1,2,3,4))
    tens_high = sum(1 for x in d if x in (5,6))
    return {
        # common names used by your CSV expressions (tens-only)
        "combo_digits": d,
        "cdigits": d,                      # alias
        "combo_sum": total,
        "combo_sum_cat": sum_category(total),
        "tens_even_count": tens_even,
        "tens_odd_count": tens_odd,
        "tens_unique_count": tens_unique,
        "tens_range": tens_range,
        "tens_low_count": tens_low,
        "tens_high_count": tens_high,
        "hot": list(hot),
        "cold": list(cold),
        "due": list(due),
        "Counter": Counter,
    }

# ==============================
# Streamlit App (Pick-5 style)
# ==============================
def main():
    st.set_page_config(page_title="Powerball Tens Manual Filter Runner", layout="wide")
    st.title("🎯 Powerball Tens — Manual Filter Runner")

    # ---- Sidebar: inputs ----
    seed = st.sidebar.text_input("Draw 1-back (required, 5 digits 0–6):", "").strip()
    d2   = st.sidebar.text_input("Draw 2-back (optional):", "").strip()
    d3   = st.sidebar.text_input("Draw 3-back (optional):", "").strip()
    d4   = st.sidebar.text_input("Draw 4-back (optional):", "").strip()
    d5   = st.sidebar.text_input("Draw 5-back (optional):", "").strip()
    d6   = st.sidebar.text_input("Draw 6-back (optional):", "").strip()

    method = st.sidebar.selectbox("Generation method:", ["1-digit", "2-digit pair"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hot/Cold/Due (tens)")
    hot_override  = st.sidebar.text_input("Hot digits override (0–6, comma-sep):", "")
    cold_override = st.sidebar.text_input("Cold digits override (0–6, comma-sep):", "")
    due_override  = st.sidebar.text_input("Due digits override (0–6, comma-sep):", "")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tracking")
    track_text = st.sidebar.text_area("Track/Test combos (newline or comma-sep):", height=90)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked   = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    st.sidebar.markdown("---")
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial cuts", value=True)
    select_all_default = st.sidebar.checkbox("Default to selected when shown", value=False)

    # ---- Validate seed ----
    if len(seed) != 5 or any(c not in DIGITS for c in seed):
        st.warning("Enter a valid **Draw 1-back**: exactly 5 digits in 0–6.")
        return

    # ---- Auto Hot/Cold/Due (only if 6 draws present) + overrides ----
    auto_hot, auto_cold, auto_due = compute_hot_cold_due([seed, d2, d3, d4, d5, d6])
    hot  = parse_digit_list(hot_override)  if hot_override.strip()  else auto_hot
    cold = parse_digit_list(cold_override) if cold_override.strip() else auto_cold
    due  = parse_digit_list(due_override)  if due_override.strip()  else auto_due

    st.sidebar.info(
        f"Auto ➜ Hot {auto_hot or '—'} | Cold {auto_cold or '—'} | Due {auto_due or '—'}\n\n"
        f"Using ➜ Hot {hot or '—'} | Cold {cold or '—'} | Due {due or '—'}"
    )

    # ---- Generate base pool ----
    base_combos = generate_combos(seed, method)

    # Track list
    tracked = []
    if track_text.strip():
        raw = track_text.replace(",", " ").split()
        tracked = ["".join(sorted(tok.strip())) for tok in raw if tok.strip()]
        if inject_tracked:
            base_combos = sorted(set(base_combos) | set(tracked))

    # ---- Load filters ----
    filters = load_filters(FILTER_CSV)

    # ---- Initial cuts (aggression ordering) ----
    init_cuts = {}
    for flt in filters:
        if flt["expr_code"] is None:
            init_cuts[flt["id"]] = 0
            continue
        c = 0
        for combo in base_combos:
            ctx = build_ctx(combo, hot, cold, due)
            try:
                if eval(flt["expr_code"], {}, ctx):
                    c += 1
            except Exception:
                pass
        init_cuts[flt["id"]] = c

    # Order by aggression (more initial cuts first; zeros last)
    ordered = sorted(filters, key=lambda f: (init_cuts.get(f["id"], 0) == 0, -init_cuts.get(f["id"], 0)))

    # Optional hide-zero
    shown_filters = [f for f in ordered if (not hide_zero) or init_cuts.get(f["id"], 0) > 0]

    # ---- Bulk toggles for shown filters ----
    c1, c2, _, _, _ = st.columns([1,1,6,1,1])
    with c1:
        if st.button("Select all (shown)"):
            for i, f in enumerate(shown_filters):
                st.session_state[f"flt_{i}_{f['key']}"] = True
    with c2:
        if st.button("Clear all (shown)"):
            for i, f in enumerate(shown_filters):
                st.session_state[f"flt_{i}_{f['key']}"] = False

    # ---- Manual filters (Pick-5 style, center) ----
    st.header("🛠 Manual Filters (sequential application)")

    # start with full pool
    pool = list(base_combos)
    dynamic_rows = []  # for compact table

    for i, flt in enumerate(shown_filters):
        fid = flt["id"]
        ic  = init_cuts.get(fid, 0)
        L   = flt.get("layman", "").strip() or "(no description)"
        H   = flt.get("stat", "").strip()
        err = flt.get("compile_error")

        # live cuts computed against current pool
        live_cuts = 0
        if flt["expr_code"] is not None:
            for combo in pool:
                try:
                    if eval(flt["expr_code"], {}, build_ctx(combo, hot, cold, due)):
                        live_cuts += 1
                except Exception:
                    pass

        # Checkbox in main area with layman text + stats
        key = f"flt_{i}_{flt['key']}"
        label = f"{fid}: {L} — init {ic}" + (f" • hist {H}" if H else "") + f" • live {live_cuts}"
        active = st.checkbox(label, key=key, value=st.session_state.get(key, select_all_default))

        # If compile error, show once and skip application
        if err:
            st.caption(f"⚠️ Compile error: {err}")
            dynamic_rows.append({"id": fid, "layman": L, "hist": H, "init_cuts": ic, "live_cuts": 0, "active": False})
            continue

        # Apply sequentially only if checked
        if active and flt["expr_code"] is not None and live_cuts > 0:
            survivors = []
            for combo in pool:
                try:
                    eliminate = bool(eval(flt["expr_code"], {}, build_ctx(combo, hot, cold, due)))
                except Exception:
                    eliminate = False
                if eliminate:
                    # preserve tracked (optional)
                    if preserve_tracked and combo in tracked:
                        survivors.append(combo)  # keep but it "would have been cut"
                    # else drop
                else:
                    survivors.append(combo)
            pool = survivors

        dynamic_rows.append({"id": fid, "layman": L, "hist": H, "init_cuts": ic, "live_cuts": live_cuts, "active": bool(active)})

    survivors = pool

    # ---- Summary & tracking ----
    st.subheader(f"Remaining after filters: {len(survivors)} of {len(base_combos)} generated")

    # Tracked report
    if tracked:
        survived = [c for c in tracked if c in survivors]
        eliminated = [c for c in tracked if c not in survivors]
        with st.expander("🔎 Tracked combos report"):
            if survived:
                st.success(f"Survived ({len(survived)}): " + ", ".join(survived))
            if eliminated:
                st.error(f"Eliminated ({len(eliminated)}): " + ", ".join(eliminated))
            if not survived and not eliminated:
                st.info("No tracked combos matched the generated pool.")

    # Small table of shown filters with live numbers
    if dynamic_rows:
        st.caption("Shown filters — initial vs live cuts (sequential view)")
        st.dataframe(pd.DataFrame(dynamic_rows), use_container_width=True, height=240)

    # ---- Survivors (download + view) ----
    st.markdown("### ✅ Survivors")
    df_out = pd.DataFrame({"tens_combo": survivors})
    st.download_button("Download survivors (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                       file_name="pb_tens_survivors.csv", mime="text/csv")
    st.download_button("Download survivors (TXT)", "\n".join(survivors).encode("utf-8"),
                       file_name="pb_tens_survivors.txt", mime="text/plain")

    with st.expander("Show remaining combinations"):
        for c in survivors:
            st.write(c)


if __name__ == "__main__":
    main()
