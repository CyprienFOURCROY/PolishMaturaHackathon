"""Blind CKE-rubric judge for essay records, via GPT Sol on Forgehand.
Usage: python3 out/tools/judge_sol.py <records.jsonl> [more.jsonl ...] [--limit N] [--reasoning low|medium]
The judge sees ONLY the topic and the essay text (no facts, no model, no variant). Scores are cached by
essay hash in out/judge/cache.jsonl and written back as `judge` into a sibling *_judged.jsonl per input file.
Costs money per uncached essay."""
import json, os, re, sys, time, hashlib, random
from openai import OpenAI

ROOT = os.environ.get("ESSAY_OUT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")  # gitignored data dir
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # essay_v2/
_flag_vals = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a in ("--limit", "--reasoning")}
args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in _flag_vals]
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10**9
REASON = sys.argv[sys.argv.index("--reasoning") + 1] if "--reasoning" in sys.argv else "low"
MODEL = "gpt-6-sol"
origin = os.getenv("FORGEHAND_URL", "https://app.forgehand.app").rstrip("/")
client = OpenAI(api_key=os.environ["FORGEHAND_TOKEN"],
                base_url=f"{origin}/api/v1/teams/{os.environ['FORGEHAND_TEAM_ID']}/llm/v1", max_retries=0)

RUBRIC = open(os.path.join(HERE, "rubryka.md"), encoding="utf-8").read()

SYSTEM = f"""Jesteś egzaminatorem CKE oceniającym wypracowanie z historii (poziom rozszerzony, formuła 2023).
Oceniasz WYŁĄCZNIE według poniższych kryteriów CKE. Jesteś surowy i konkretny, jak na prawdziwym egzaminie:
poziom "bogata" wymaga argumentacji pogłębionej i szczegółowej faktografii, "zadowalająca" wymaga prawidłowej
faktografii i elementów refleksji, "powierzchowna" to uogólnienia i podstawowa faktografia. Sprawdzasz też
poprawność merytoryczną na podstawie własnej wiedzy historycznej: każdą ewidentną pomyłkę (chronologia, terminologia,
związki przyczynowo-skutkowe, atrybucja) wypisujesz i odejmujesz łącznie 1-3 pkt od A (0, jeśli brak błędów).

=== KRYTERIA CKE ===
{RUBRIC}
=== KONIEC KRYTERIÓW ===

Zwróć wyłącznie JSON:
{{"stanowisko_obecne": bool,
 "aspekty": [{{"nazwa": str, "poziom": "bogata|zadowalajaca|powierzchowna|brak", "funkcjonalnosc": "pelna|czesciowa|niefunkcjonalna", "uzasadnienie": str (1-2 zdania)}} x3],
 "A_przed_odjeciem": int (0-12, z tabeli),
 "bledy_merytoryczne": [str],
 "odjecie": int (0-3),
 "A": int,
 "B": int (0-3), "B_uzasadnienie": str,
 "liczba_slow": int,
 "suma": int (A+B, maks. 15),
 "najwazniejsza_slabosc": str (jedno zdanie: co najbardziej obniżyło ocenę)}}"""

def essay_text(rec):
    e = rec.get("essay") or {}
    if "full_text" in rec:
        return rec.get("temat") or e.get("temat", ""), rec["full_text"]
    parts = [e.get("wstep", "")] + [a.get("tekst", "") for a in e.get("akapity", [])] + [e.get("zakonczenie", "")]
    return e.get("temat", ""), "\n\n".join(p for p in parts if p)

CACHE_PATH = os.path.join(ROOT, "judge", "cache.jsonl")
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
cache = {}
if os.path.exists(CACHE_PATH):
    for l in open(CACHE_PATH, encoding="utf-8"):
        c = json.loads(l); cache[c["hash"]] = c

def judge_one(temat, text):
    h = hashlib.sha256((MODEL + REASON + temat + text).encode()).hexdigest()[:16]
    if h in cache:
        return cache[h]["judge"], 0.0, True
    user = f"TEMAT WYPRACOWANIA:\n{temat}\n\nWYPRACOWANIE:\n{text}"
    j = None
    for attempt, budget in enumerate((2500, 5000)):
        resp = client.chat.completions.create(
            model=MODEL, reasoning_effort=REASON, max_completion_tokens=budget,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        content = resp.choices[0].message.content or ""
        try:
            j = json.loads(content); break
        except json.JSONDecodeError:
            print(f"    [judge] attempt {attempt+1}: non-JSON reply, finish_reason={resp.choices[0].finish_reason}, "
                  f"len={len(content)}, completion_tokens={resp.usage.completion_tokens}")
    if j is None:
        j = dict(error="non-json", aspekty=[], A=None, A_przed_odjeciem=None, odjecie=None, B=None, suma=None,
                 bledy_merytoryczne=[], najwazniejsza_slabosc="(judge failed)")
    j["_judge_model"] = MODEL; j["_reasoning"] = REASON
    j["_tokens"] = dict(prompt=resp.usage.prompt_tokens, completion=resp.usage.completion_tokens)
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(hash=h, judge=j), ensure_ascii=False) + "\n")
    cache[h] = dict(hash=h, judge=j)
    return j, resp.usage.prompt_tokens + resp.usage.completion_tokens, False

def usage():
    import httpx
    r = httpx.get(f"{origin}/api/v1/teams/{os.environ['FORGEHAND_TEAM_ID']}/llm/usage",
                  headers={"Authorization": f"Bearer {os.environ['FORGEHAND_TOKEN']}"}, timeout=30)
    return r.json()["spentMicroUsd"] / 1e6

done = 0
spent0 = usage()
for path in args:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    order = list(range(len(recs))); random.Random(0).shuffle(order)  # blind order
    for idx in order:
        if done >= LIMIT: break
        r = recs[idx]
        temat, text = essay_text(r)
        j, toks, cached = judge_one(temat, text)
        r["judge"] = j
        done += 0 if cached else 1
        lv = "/".join(a.get("poziom", "?")[:4] for a in j["aspekty"]) or "FAILED"
        print(f"{os.path.basename(path)}:{idx+1} {'(cached)' if cached else ''} | A={j['A']} (pre {j['A_przed_odjeciem']}, -{j['odjecie']}) B={j['B']} → {j['suma']}/15 | {lv} | {r.get('dzial','')[:45]}")
        if j["bledy_merytoryczne"]: print("    błędy:", "; ".join(j["bledy_merytoryczne"])[:300])
        print("    słabość:", j["najwazniejsza_slabosc"][:200])
        sys.stdout.flush()
    outp = re.sub(r"\.jsonl$", "_judged.jsonl", path)
    with open(outp, "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {outp}")
print(f"judge spend this run: ${usage()-spent0:.3f} | total team spend ${usage():.3f}")
