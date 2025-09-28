import pandas as pd
from collections import Counter
import streamlit as st

# --- Config ---
DIGITS = "0123456"  # Tens digit range

# Mirror & VTRAC (same as ones)
MIRROR_MAP = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
VTRAC_MAP = {0: "0", 5: "0", 1: "1", 6: "1", 2: "2", 7: "2", 3: "3", 8: "3", 4: "4", 9: "4"}

def mirror_digits(digs):
    return [MIRROR_MAP.get(d, d) for d in digs]

def mirror(n: int) -> int:
    return int("".join(str(MIRROR_MAP.get(int(ch), int(ch))) for ch in str(n)))

# --- Auto hot/cold/due adapted for tens ---
def auto_hot_cold_due(seed: str, prevs: list[str], hot_n=3, cold_n=3, hot_window=10, due_window=2):
    recents = [seed] + prevs[:hot_window-1]
    seq = "".join(recents)
    counts = Counter(int(ch) for ch in seq if ch.isdigit() and 0 <= int(ch) <= 6)
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    hot = [d for d, _ in ordered[:hot_n]]
    cold = [d for d, _ in sorted(counts.items(), key=lambda kv: kv[1])[:cold_n]]
    seq_due = "".join([seed] + prevs[:due_window-1])
    due = []
    if seq_due:
        seen = {int(ch) for ch in seq_due if ch.isdigit() and 0 <= int(ch) <= 6}
        due = [d for d in range(0,7) if d not in seen]
    return hot, cold, due

# --- Filter loader ---
def load_filters(path: str, is_zone: bool = False) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if is_zone:
        df = df[df["variant"].eq("zone")]
    else:
        df = df[df["variant"].ne("zone")]
    return df

def parse_list(txt):
    if not txt:
        return []
    return [int(ch) for ch in txt if ch.isdigit() and 0 <= int(ch) <= 6]

# --- Context builder ---
def make_ctx(seed: str, prevs: list[str], hot_override="", cold_override="", due_override=""):
    auto_hot, auto_cold, auto_due = auto_hot_cold_due(seed, prevs)
    hot = parse_list(hot_override) or auto_hot
    cold = parse_list(cold_override) or auto_cold
    due = parse_list(due_override) or auto_due
    return {
        "seed": seed,
        "seed_digits": [int(x) for x in seed if x.isdigit() and 0 <= int(x) <= 6],
        "hot_digits": hot,
        "cold_digits": cold,
        "due_digits": due,
        "hot": hot,
        "cold": cold,
        "due": due,
        "due_set": set(due),
    }

# --- Streamlit UI (matching ones app style) ---
st.set_page_config(page_title="PB Tens Filter App", layout="wide")
st.title("PB Tens Filter App (mirrored from Ones app)")

manual_path = st.file_uploader("Upload Tens Filters CSV", type=["csv"])
zone_path = st.file_uploader("Upload Zone Filters CSV (optional)", type=["csv"])

def load_all_filters():
    mf = load_filters(manual_path, is_zone=False)
    zf = load_filters(zone_path, is_zone=True) if zone_path else pd.DataFrame()
    return pd.concat([mf, zf], ignore_index=True)

if manual_path:
    all_filters = load_all_filters()
    st.success(f"Loaded {len(all_filters)} filters")
else:
    all_filters = pd.DataFrame()

seed = st.text_input("Seed (previous winning number)")
prevs_input = st.text_area("Prev draws (comma separated)")
prevs = [s.strip() for s in prevs_input.split(",") if s.strip()]
hot_override = st.text_input("Hot digits override")
cold_override = st.text_input("Cold digits override")
due_override = st.text_input("Due digits override")

if seed:
    ctx = make_ctx(seed, prevs, hot_override, cold_override, due_override)
    st.sidebar.write("**Auto Hot:**", ctx["hot_digits"])
    st.sidebar.write("**Auto Cold:**", ctx["cold_digits"])
    st.sidebar.write("**Auto Due:**", ctx["due_digits"])
    st.sidebar.write("**Using Hot:**", ctx["hot_digits"])
    st.sidebar.write("**Using Cold:**", ctx["cold_digits"])
    st.sidebar.write("**Using Due:**", ctx["due_digits"])

    if not all_filters.empty:
        st.subheader("Filter Results")
        results = []
        for _, row in all_filters.iterrows():
            expr = row.get("expression", "")
            try:
                eliminated = eval(expr, {}, ctx)
            except Exception:
                eliminated = False
            results.append({
                "filter_id": row.get("filter_id"),
                "expression": expr,
                "layman_explanation": row.get("layman_explanation"),
                "eliminated": eliminated,
            })
        df_out = pd.DataFrame(results)
        st.dataframe(df_out)

        # Download buttons
        csv = df_out.to_csv(index=False).encode()
        st.download_button("Download Results CSV", csv, "tens_filter_results.csv", "text/csv")
