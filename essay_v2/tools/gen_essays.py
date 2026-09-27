"""Training-data generator v2: thesis from the bank (written before any fact was seen) + independent fact sample.
Usage: python3 essay_v2/tools/gen_essays.py <seed> <n> [--tag NAME]
Reads out/theses.jsonl and the cleaned kanon; writes out/sol/essays_<tag or seed>.jsonl (append, resumable).
Each record: temat, aspekty (CKE names), material (facts shown), essay (structured), check, dopasowanie_faktow.
Leak filter: theses with high word overlap against essay_v2/topics_real.json are skipped."""
import json, os, re, sys, time, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_sol as G                       # SYSTEM prompt (v5 house style), STANCES, FORMULA, check(), client, usage()
from gen_theses import ASPECT_MAP, norm_section, ROOT, KANON

seed, n = int(sys.argv[1]), int(sys.argv[2])
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else str(seed)
rng = random.Random(seed)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def norm2(d):
    d = norm_section(d)
    return re.sub(r"\b([IVX]+) i ([IVX]+) w\b", r"\1–\2 w", d)

rows = [json.loads(l) for l in open(KANON, encoding="utf-8")]
facts_by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    facts_by[norm2(r["dzial"])][r["aspekt"]].append(r)

bank = []
for l in open(os.path.join(ROOT, os.environ.get("THESES_FILE", "theses.jsonl")), encoding="utf-8"):
    e = json.loads(l)
    for t in e["tezy"]:
        bank.append(dict(dzial=norm2(e["dzial"]), teza=t["teza"], aspekty=t["aspekty"]))

# leak filter against real held-out topics
real = json.load(open(os.path.join(HERE, "topics_real.json"), encoding="utf-8"))["tematy"]
def words(s): return set(re.findall(r"\w+", s.lower()))
real_sets = [words(t["tekst"].split(". Zajmij")[0]) for t in real]
def leaks(teza):
    w = words(teza)
    return any(len(w & r) / max(1, len(w | r)) > 0.5 for r in real_sets)
bank = [b for b in bank if not leaks(b["teza"])]

STOP = set("oraz albo lub jako przez tego jego jej tym była było były jest przede wszystkim bardziej niż więcej okres okresie dla".split())
def stems(s):
    return {w[:5] for w in re.findall(r"\w+", s.lower()) if len(w) >= 4 and w not in STOP}

def sample_facts(dzial, teza, cke_aspect, k=6):
    """Thesis-relevant retrieval (exam-time prototype): stem overlap with the thesis, +bonus for the thesis's own
    section, small random jitter, near-duplicates removed. Pool = all facts whose kanon aspect maps to the CKE aspect."""
    q = stems(teza) | stems(dzial)
    pool = []
    for d, by_tag in facts_by.items():
        for tag in ASPECT_MAP[cke_aspect]:
            for f in by_tag.get(tag, []):
                fs = stems(f["fakt"] + " " + f.get("postac", "") + " " + f.get("termin", "") + " " + f.get("tytul", ""))
                score = len(q & fs) + (1.5 if d == dzial else 0) + (0.3 if f["data"] else 0) + rng.random() * 0.2
                pool.append((score, f, fs))
    pool.sort(key=lambda x: -x[0])
    chosen, seen = [], []
    for score, f, fs in pool:
        if any(len(fs & s) / max(1, len(fs | s)) > 0.4 for s in seen):
            continue
        chosen.append(f); seen.append(fs)
        if len(chosen) == k:
            break
    return chosen

def user_msg(b, mat):
    temat = G.FORMULA.format(teza=b["teza"], a1=b["aspekty"][0], a2=b["aspekty"][1], a3=b["aspekty"][2])
    fl = "\n".join(f'{m["id"]} [{m["aspekt"]}] {m["fakt"]}' + (f' ({m["data"]})' if m["data"] else "")
                   + (f' [termin: {m["termin"]}]' if m.get("termin") else "") for m in mat)
    return temat, f"""Temat wypracowania (dokładnie taki jak w arkuszu, NIE zmieniaj go):
"{temat}"

Aspekty (w tej kolejności): {b['aspekty'][0]}, {b['aspekty'][1]}, {b['aspekty'][2]}

Lista faktów, którymi dysponuje uczeń (i TYLKO nimi):
{fl}

Stanowisko, które MUSISZ przyjąć: {G.STANCES['zgadzam']}

Najpierw oceń, czy podane fakty pozwalają uczciwie uzasadnić tezę we wszystkich trzech aspektach
(pole "dopasowanie_faktow": "dobre" = tak; "slabe" = w co najmniej jednym aspekcie fakty ledwo dotykają tezy).
Napisz wypracowanie tak czy inaczej, wybierając najlepiej pasujące fakty i nie naciągając ich sensu.
WAŻNE: w samym wypracowaniu NIGDY nie sygnalizuj, że faktów brakuje lub że słabo pasują (żadnych zdań typu
"podane fakty nie wskazują", "trudno ocenić", "źródła nie pozwalają"). Uczeń na egzaminie pisze pewnie: trzyma
stanowisko, buduje argument z tego, co ma, a luki wypełnia ogólnym, poprawnym tłem bez nowych dat i nazw.

Zwróć JSON o polach:
{{"dopasowanie_faktow": "dobre|slabe", "teza": str (bez zmian), "temat": str (bez zmian), "stanowisko": "zgadzam",
 "wstep": str, "akapity": [{{"aspekt": str, "fakty_uzyte": [ids], "tekst": str}} x3], "zakonczenie": str}}"""

def main():
    os.makedirs(os.path.join(ROOT, "sol"), exist_ok=True)
    outp = os.path.join(ROOT, "sol", f"essays_{TAG}.jsonl")
    done = 0
    if os.path.exists(outp):
        done = sum(1 for _ in open(outp, encoding="utf-8"))
    order = list(range(len(bank))); rng.shuffle(order)
    spent0 = G.usage(); made = 0
    for idx in order[done:done + n]:
        b = bank[idx]
        mat, fid = [], 1
        for a in b["aspekty"]:
            for f in sample_facts(b["dzial"], b["teza"], a):
                mat.append(dict(id=f"F{fid}", aspekt=a, fakt=f["fakt"], data=f["data"], termin=f["termin"])); fid += 1
        if min(sum(1 for m in mat if m["aspekt"] == a) for a in b["aspekty"]) < 3:
            print(f"SKIP thin material: {b['dzial']} / {b['aspekty']}"); continue
        temat, user = user_msg(b, mat)
        t = time.time()
        rec = None
        for attempt in range(2):
            try:
                resp = G.client.chat.completions.create(model=G.MODEL, reasoning_effort=G.REASON, max_completion_tokens=3000,
                                                        response_format={"type": "json_object"},
                                                        messages=[{"role": "system", "content": G.SYSTEM}, {"role": "user", "content": user}])
                rec = json.loads(resp.choices[0].message.content); break
            except Exception as ex:  # transient API / non-JSON reply: one retry, then skip this thesis
                print(f"    retry {attempt+1}: {type(ex).__name__}: {str(ex)[:120]}"); time.sleep(3)
        if rec is None or not isinstance(rec.get("akapity"), list) or len(rec["akapity"]) != 3:
            print(f"SKIP after failures: {b['teza'][:60]}"); continue
        rec["temat"] = temat; rec["teza"] = b["teza"]
        chk = G.check(rec, mat)
        record = dict(gen="v2-bank", seed=seed, bank_idx=idx, prompt_version=G.PROMPT_VERSION, dzial=b["dzial"], aspekty=b["aspekty"],
                      stance="zgadzam", dopasowanie=rec.get("dopasowanie_faktow"), material=mat, essay=rec, check=chk,
                      usage=dict(prompt=resp.usage.prompt_tokens, completion=resp.usage.completion_tokens), model=G.MODEL, reasoning=G.REASON)
        with open(outp, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        made += 1
        print(f"[{made}/{n}] {b['dzial'][:40]:40} | fit={rec.get('dopasowanie_faktow')} | {chk['words']}w | {time.time()-t:.0f}s | {b['teza'][:70]}")
        sys.stdout.flush()
    print(f"spend this run ${G.usage()-spent0:.2f} | total ${G.usage():.2f} | -> {outp}")

if __name__ == "__main__":
    main()
