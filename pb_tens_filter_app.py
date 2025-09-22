# pb_tens_filter_app.py — Powerball tens-only manual filter runner

from __future__ import annotations
import os, csv
from collections import Counter
from itertools import product
from typing import List, Dict, Any, Tuple

import pandas as pd
import streamlit as st

# ==============================
# Config
# ==============================
DIGITS = "0123456"
FILTER_MAIN = "pb_tens_filters_adapted.csv"
FILTER_PCT  = "pb_tens_percentile_filters.csv"

LOW_SET = {0,1,2,3,4}
HIGH_SET = {5,6}

# ==============================
# Helpers
# ==============================
def parse_int_list_csv(txt: str, lo: int, hi: int) -> List[int]:
    out=[]
    for tok in txt.split(","):
        tok=tok.strip()
        if tok.isdigit():
            v=int(tok)
            if lo<=v<=hi: out.append(v)
    return out

def multiset_shared(a: List[int], b: List[int]) -> int:
    ca,cb=Counter(a),Counter(b)
    return sum((ca&cb).values())

# ==============================
# Filters
# ==============================
def load_filters(path: str, is_pct=False) -> List[Dict[str,Any]]:
    if not os.path.exists(path): return []
    out=[]
    with open(path,newline="",encoding="utf-8") as f:
        rdr=csv.DictReader(f)
        for i,row in enumerate(rdr):
            fid=row.get("id") or f"row{i+1}"
            lay=row.get("layman") or ""
            stat=row.get("stat") or ""
            expr=row.get("expression") or "False"
            try:
                code=compile(expr,f"<expr:{fid}>","eval")
            except Exception as e:
                code=None
            out.append({"id":fid,"lay":lay,"stat":stat,"expr":expr,"code":code,"is_pct":is_pct})
    return out

def load_all_filters()->List[Dict[str,Any]]:
    flts=load_filters(FILTER_MAIN,False)
    if os.path.exists(FILTER_PCT):
        flts+=load_filters(FILTER_PCT,True)
    return flts

# ==============================
# Combo generation
# ==============================
def generate(seed:str,method:str)->Tuple[List[str],List[str]]:
    seed="".join(sorted(seed))
    raw=[]; uniq=set()
    if method=="1-digit":
        for d in seed:
            for p in product(DIGITS,repeat=4):
                k="".join(sorted(d+"".join(p)))
                raw.append(k); uniq.add(k)
    else:
        pairs={ "".join(sorted((seed[i],seed[j]))) for i in range(5) for j in range(i+1,5)}
        for pair in pairs:
            for p in product(DIGITS,repeat=3):
                k="".join(sorted(pair+"".join(p)))
                raw.append(k); uniq.add(k)
    return raw,sorted(uniq)

# ==============================
# Hot/Cold/Due
# ==============================
def compute_hotcold(draws:List[str]):
    last6=[d for d in draws if d][:6]
    if len(last6)<6: return [],[],[]
    all_digits="".join(last6)
    cnt=Counter(all_digits)
    hot=[int(x) for x,_ in cnt.most_common(3)]
    cold=[int(x) for x,_ in reversed(cnt.most_common())][:3]
    due=[d for d in range(7) if str(d) not in all_digits]
    return hot,cold,due

# ==============================
# Eval
# ==============================
def ctx(seed:str,combo:str,hot,cold,due):
    cd=[int(c) for c in combo]
    return {"combo":combo,"cdigits":cd,
            "seed":[int(c) for c in seed],
            "hot_digits":hot,"cold_digits":cold,"due_digits":due,
            "shared_tens":multiset_shared,"Counter":Counter}

def apply_once(pool:List[str],filters:List[Dict[str,Any]],ctx_base:Dict[str,Any]):
    cut_counts={f["id"]:0 for f in filters}
    surv=[]
    for combo in pool:
        c=ctx_base.copy(); c.update(ctx(ctx_base["seed_str"],combo,
                                        ctx_base["hot_digits"],ctx_base["cold_digits"],ctx_base["due_digits"]))
        elim=False
        for f in filters:
            if f["code"] is None: continue
            try:
                if eval(f["code"],{},c):
                    cut_counts[f["id"]]+=1; elim=True; break
            except Exception: pass
        if not elim: surv.append(combo)
    return surv,cut_counts

# ==============================
# UI
# ==============================
st.set_page_config(page_title="PB Tens Filter Runner",layout="wide")
st.title("🎯 PB Tens Filter Runner")

with st.sidebar:
    seed=st.text_input("Draw 1-back (5 digits 0–6):","").strip()
    prevs=[st.text_input(f"Draw {i}-back (opt):","").strip() for i in range(2,7)]
    method=st.selectbox("Generation Method:",["1-digit","2-digit pair"])
    hot_o=st.text_input("Hot override:","")
    cold_o=st.text_input("Cold override:","")
    due_o=st.text_input("Due override:","")
    tracked=st.text_input("Track combo:","").strip()
    select_all=st.checkbox("Select/Deselect All",value=False)
    hide_zero=st.checkbox("Hide 0-cut filters",value=True)

if len(seed)!=5 or any(c not in DIGITS for c in seed):
    st.warning("Enter valid 5-digit seed (0–6)."); st.stop()

auto_hot,auto_cold,auto_due=compute_hotcold([seed]+prevs)
hot=parse_int_list_csv(hot_o,0,6) or auto_hot
cold=parse_int_list_csv(cold_o,0,6) or auto_cold
due=parse_int_list_csv(due_o,0,6) or auto_due

filters=load_all_filters()
raw,uniq=generate(seed,method)

ctx_base={"seed":[int(c) for c in seed],"seed_str":seed,
          "hot_digits":hot,"cold_digits":cold,"due_digits":due}

survivors,init_cuts=apply_once(uniq,[f for f in filters if not f["is_pct"]],ctx_base)

manual=[f for f in filters if not f["is_pct"]]
sorted_manual=sorted(manual,key=lambda f:(init_cuts[f["id"]]==0,-init_cuts[f["id"]],f["id"]))

display_filters=[f for f in sorted_manual if init_cuts[f["id"]]>0] if hide_zero else sorted_manual

# Filter count header
st.header(f"🛠 Manual Filters — {len(display_filters)} available")

active_ids=[]
for f in display_filters:
    fid=f["id"]; label=f["lay"] or fid; hist=f["stat"]
    init=init_cuts.get(fid,0)
    cols=st.columns((1,6,2,2))
    cols[0].write(fid)
    cols[1].write(f"{label}{' — hist '+hist if hist else ''}")
    cols[2].write(init)
    checked=cols[3].checkbox("apply",key=f"chk_{fid}",value=select_all)
    if checked: active_ids.append(fid)

# Sequentially apply active filters
pool=list(uniq)
for f in display_filters:
    if f["id"] not in active_ids: continue
    new=[]
    for combo in pool:
        c=ctx_base.copy(); c.update(ctx(seed,combo,hot,cold,due))
        try:
            elim=eval(f["code"],{},c)
        except Exception: elim=False
        if not elim: new.append(combo)
    pool=new

st.subheader(f"Remaining after manual filters: {len(pool)}")
if tracked:
    if tracked in pool: st.success(f"Tracked {tracked} survived.")
    else: st.error(f"Tracked {tracked} eliminated.")

with st.expander("Show survivors"):
    for c in pool: st.write(c)

if pool:
    df=pd.DataFrame({"combo":pool})
    st.download_button("Download CSV",df.to_csv(index=False),"survivors.csv","text/csv")
    st.download_button("Download TXT","\n".join(pool),"survivors.txt","text/plain")
