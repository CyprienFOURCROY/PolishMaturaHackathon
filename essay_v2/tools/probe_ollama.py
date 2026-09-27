"""Zero-shot baseline: can a 1.5B write one essay paragraph from 5 injected facts?
Bielik-1.5B-v3 vs Qwen2.5-1.5B, random kanon sections, n=3 each.
Prints raw prose + date-use stats. Nothing here is a grade."""
import json, os, random, re, sys, time, collections, urllib.request
ROOT = os.environ.get("ESSAY_OUT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
N = 3
MODELS = ["bielik-1.5b-v3:latest", "qwen2.5:1.5b"]
rows = [json.loads(l) for l in open(os.environ.get("KANON_PATH") or os.path.join(ROOT, "kanon.jsonl"), encoding="utf-8")]
rng = random.Random(SEED)

# pick 3 sections, each with an aspect that has >=5 dated facts
by = collections.defaultdict(list)
for r in rows:
    if r["data"]:
        by[(r["dzial"], r["aspekt"])].append(r)
keys = [k for k, v in by.items() if len(v) >= 5]
rng.shuffle(keys)
picked, seen = [], set()
for k in keys:
    if k[0] not in seen:
        picked.append(k); seen.add(k[0])
    if len(picked) == 3:
        break

SYSTEM = ("Jesteś maturzystą piszącym wypracowanie z historii na poziomie rozszerzonym. "
          "Piszesz poprawną polszczyzną, rzeczowo, bez punktów i nagłówków.")

def prompt(dzial, aspekt, facts):
    teza = f"Czynniki o charakterze {aspekt}m odegrały kluczową rolę w dziale: {dzial}."
    fl = "\n".join(f"- {f['fakt']} ({f['data']})" for f in facts)
    return (f"Teza wypracowania: {teza}\n"
            f"Aspekt tego akapitu: {aspekt}.\n\n"
            f"Fakty, których masz użyć (nie dodawaj innych faktów ani dat):\n{fl}\n\n"
            "Napisz jeden akapit argumentacyjny (6-8 zdań, ok. 120 słów). Każde zdanie ma wiązać fakt z tezą. "
            "Użyj wszystkich podanych faktów. Pisz tylko po polsku, ciągłym tekstem.")

def call(model, sys_msg, user):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                       "options": {"temperature": 0.7, "num_predict": 320}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return out["message"]["content"], time.time() - t

YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")
def years(s): return set(YEAR.findall(s))

results = []
for dzial, aspekt in picked:
    facts = by[(dzial, aspekt)][:5]
    inj = set()
    for f in facts: inj |= years(f["data"])
    p = prompt(dzial, aspekt, facts)
    print("=" * 100); print(f"DZIAŁ: {dzial} | ASPEKT: {aspekt} | injected years: {sorted(inj)}")
    for f in facts: print("  -", f["fakt"][:110], f"({f['data']})")
    for m in MODELS:
        for i in range(N):
            txt, dt = call(m, SYSTEM, p)
            clean = re.sub(r"\s+", " ", txt).strip()
            wc = len(clean.split())
            ys = years(clean)
            used, invented = ys & inj, ys - inj
            echo = "Fakty" in clean or "Napisz" in clean or "Teza wypracowania" in clean
            latin = bool(re.search(r"[\u4e00-\u9fff]", clean))
            stat = dict(model=m, dzial=dzial, words=wc, used=len(used), inj=len(inj), invented=sorted(invented),
                        empty=wc == 0, echo=echo, cjk=latin, sec=round(dt))
            results.append(stat)
            print(f"\n--- {m} #{i+1} | {wc} w | dates used {len(used)}/{len(inj)} | invented {sorted(invented)} | echo={echo} cjk={latin} | {dt:.0f}s")
            print(clean)
    sys.stdout.flush()

print("\n" + "=" * 100 + "\nSUMMARY")
for m in MODELS:
    rs = [r for r in results if r["model"] == m]
    print(f"{m}: n={len(rs)} | empty {sum(r['empty'] for r in rs)} | echo {sum(r['echo'] for r in rs)} | cjk {sum(r['cjk'] for r in rs)} | "
          f"mean words {sum(r['words'] for r in rs)/len(rs):.0f} | all-dates-used {sum(r['used']==r['inj'] for r in rs)}/{len(rs)} | "
          f"samples with invented dates {sum(bool(r['invented']) for r in rs)}/{len(rs)}")
json.dump(results, open(os.path.join(ROOT, f"probe_ollama_{SEED}.json"), "w"), ensure_ascii=False, indent=1)
