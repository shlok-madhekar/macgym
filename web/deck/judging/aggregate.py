import json, glob, sys, collections, re
rnd = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"/Users/shlok/macgym/web/deck/judging/{rnd}_g*_out.json")):
    try:
        t = open(f).read(); t = t[t.find('['): t.rfind(']')+1]; rows += json.loads(t)
    except Exception as e:
        print("bad", f, e)
J = {j["id"]: j for j in json.load(open("/Users/shlok/macgym/web/deck/judging/judges.json"))}
crit = ["creativity","technical","functionality","uiux","theme","pitch"]
n = len(rows); print(f"{rnd}: {n} judges")
if not n: sys.exit()
for c in crit:
    v=[r["scores"][c] for r in rows]; print(f"  {c:13s} {sum(v)/n:5.2f}  min {min(v)}")
tot=[sum(r["scores"].values()) for r in rows]; print(f"  TOTAL /60     {sum(tot)/n:5.2f}  min {min(tot)} max {max(tot)}")
print("  rank guesses", collections.Counter(r.get("rank_guess") for r in rows))
print("  best slide", collections.Counter(r.get("best_slide") for r in rows).most_common(4), " worst", collections.Counter(r.get("worst_slide") for r in rows).most_common(4))
print("\nLOWEST 6 judges:")
for r in sorted(rows, key=lambda r: sum(r["scores"].values()))[:6]:
    print(f"  #{r['judge_id']} {J.get(r['judge_id'],{}).get('persona','?')[:60]} = {sum(r['scores'].values())}: {r.get('first_impression')}")
print("\nFIXES by slide (impact-weighted):")
by=collections.defaultdict(list)
for r in rows:
    for fx in r.get("fixes",[]): by[fx.get("slide")].append((fx.get("impact",1), fx.get("criterion"), fx.get("change")))
for s in sorted(by, key=lambda s: -sum(i for i,_,_ in by[s])):
    print(f"-- slide {s}  weight {sum(i for i,_,_ in by[s])}")
    for i,c,ch in sorted(by[s], key=lambda x:-x[0])[:7]: print(f"   [{i}] ({c}) {ch}")
print("\nDOUBTS:"); [print("  -",d) for r in rows for d in r.get("doubts",[])]
print("\nCONFUSING:"); [print("  -",d) for r in rows for d in r.get("confusing",[])]
print("\nQUESTIONS:"); [print("  -",d) for r in rows for d in r.get("questions",[])[:2]]
