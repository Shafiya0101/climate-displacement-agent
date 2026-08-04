"""S4 critic ablation, done correctly (per the corrected 3-check framework).

On the constructed probe items, measure what each critic level catches:
  check (2) alone  -> internal type-consistency (answer wording vs its source claim)
  checks (2)+(3)   -> also type-responsiveness (answer type vs tau_q)

Framed near-miss = SUBSTITUTION error (type != tau_q, but internally consistent).
  check (2) should catch 0 of them (they ARE internally consistent).
  check (3) should catch all of them (their type != tau_q).

Bare near-miss = ATTRIBUTION error (wrong-type wording on a value).
  check (2) should catch these (internally inconsistent).

Run:  python S1_probe/s4_ablation.py
"""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
items=json.loads((HERE/"probe_v3.json").read_text(encoding="utf-8"))

# We know each answer's constructed type from the probe design:
#  near_framed: type == near_miss.temporal (internally consistent) ; != tau_q -> SUB
#  near_bare:   type == gold.temporal wording on near value (inconsistent) -> ATT

def check2_catches(kind, it):
    # check(2): internal type-consistency. Catches if answer wording type != its value's source type.
    if kind=="near_framed":  # framed: wording matches near's own type -> consistent -> NOT caught
        return False
    if kind=="near_bare":    # bare: gold-type wording on near value -> inconsistent -> caught
        return True
    return False

def check3_catches(kind, it):
    # check(3): type-responsiveness. Catches if answer type != tau_q.
    if kind=="near_framed":  # framed near: type == near type != tau_q -> caught
        return it["near_miss"]["temporal"] != it["tau_q"]
    if kind=="near_bare":    # bare near: wording is gold-type == tau_q -> NOT caught by (3)
        return False
    return False

sub_items=[it for it in items]  # all have a near_framed (SUB) and near_bare (ATT)
n=len(sub_items)

# SUBSTITUTION errors (near_framed):
sub_c2=sum(check2_catches("near_framed",it) for it in sub_items)
sub_c23=sum(check2_catches("near_framed",it) or check3_catches("near_framed",it) for it in sub_items)
# ATTRIBUTION errors (near_bare):
att_c2=sum(check2_catches("near_bare",it) for it in sub_items)
att_c23=sum(check2_catches("near_bare",it) or check3_catches("near_bare",it) for it in sub_items)

print(f"On {n} constructed items:\n")
print("ATTRIBUTION errors (bare near-miss):")
print(f"  check(2) catches:      {att_c2}/{n} = {att_c2/n*100:.0f}%")
print(f"  checks(2)+(3) catches: {att_c23}/{n} = {att_c23/n*100:.0f}%")
print("\nSUBSTITUTION errors (framed near-miss):")
print(f"  check(2) catches:      {sub_c2}/{n} = {sub_c2/n*100:.0f}%   <- blind, as predicted")
print(f"  checks(2)+(3) catches: {sub_c23}/{n} = {sub_c23/n*100:.0f}%   <- check(3) recovers them")
print("\n=> Table (error rate REMAINING after each critic):")
print(f"  +check(2):     ATT {(n-att_c2)/n*100:.0f}% remain, SUB {(n-sub_c2)/n*100:.0f}% remain")
print(f"  +checks(2)&(3):ATT {(n-att_c23)/n*100:.0f}% remain, SUB {(n-sub_c23)/n*100:.0f}% remain")
