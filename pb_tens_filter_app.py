import pandas as pd
from collections import Counter
import streamlit as st

DIGITS = "0123456"  # keep tens digit range 0-6

# --- Mirror & VTRAC helpers copied from ones ---
MIRROR_MAP = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
VTRAC_MAP = {
    0: "0", 5: "0",
    1: "1", 6: "1",
    2: "2", 7: "2",
    3: "3", 8: "3",
    4: "4", 9: "4"
}

def mirror_digits(digs):
    return [MIRROR_MAP.get(d, d) for d in digs]

def mirror(n: int) -> int:
    return int("".join(str(MIRROR_MAP.get(int(ch), int(ch))) for ch in str(n)))

# --- robust auto hot/cold/due from ones app but with digits 0-6 ---
def auto_hot_cold_due(seed: str, prevs: list[str], hot_n=3, cold_n=3, hot_window=10, due_window=2):
    recents = [seed] + prevs[:hot_window-1]
    seq = "".join(recents)
    counts = Counter(int(ch) for ch in seq if ch.isdigit())
    # restrict to 0-6
    counts = {k: v for k, v in counts.items() if 0 <= k <= 6}
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    hot = [d for d, _ in ordered[:hot_n]]
    cold = [d for d, _ in sorted(counts.items(), key=lambda kv: kv[1])[:cold_n]]

    # due digits = missing from last 2 draws
    seq_due = "".join([seed] + prevs[:due_window-1])
    due = []
    if seq_due:
        seen = {int(ch) for ch in seq_due if ch.isdigit() and 0 <= int(ch) <= 6}
        due = [d for d in range(0,7) if d not in seen]
    return hot, cold, due

# --- CSV loader ---
def load_filters(path: str, is_zone: bool = False) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if is_zone:
        df = df[df["variant"].eq("zone")]
    else:
        df = df[df["variant"].ne("zone")]
    return df

# --- Parse override digits ---
def parse_list(txt):
    if not txt:
        return []
    return [int(ch) for ch in txt if ch.isdigit() and 0 <= int(ch) <= 6]

# --- Build context for filters ---
def make_ctx(seed: str, prevs: list[str], hot_override: str = "", cold_override: str = "", due_override: str = ""):
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

# --- UI ---
st.title("PB Tens Filter App (fixed to match Ones app)")
manual_path = st.file_uploader("Upload Tens Filters CSV", type=["csv"])
if manual_path:
    df = load_filters(manual_path, is_zone=False)
    st.success(f"Loaded {len(df)} filters")

seed = st.text_input("Seed (previous winning number)")
prevs = st.text_area("Prev draws (comma separated)").split(",") if st.text_area else []
hot_override = st.text_input("Hot digits override (optional)")
cold_override = st.text_input("Cold digits override (optional)")
due_override = st.text_input("Due digits override (optional)")

if seed:
    ctx = make_ctx(seed, prevs, hot_override, cold_override, due_override)
    st.write("Auto Hot:", ctx["hot_digits"], "Cold:", ctx["cold_digits"], "Due:", ctx["due_digits"])

    # Test a filter application on dummy combos for demonstration
    if not df.empty:
        results = []
        for _, row in df.iterrows():
            expr = row["expression"]
            try:
                keep = not eval(expr, {}, ctx)
            except Exception as e:
                keep = False
            results.append((row["filter_id"], keep))
        st.write(results)
