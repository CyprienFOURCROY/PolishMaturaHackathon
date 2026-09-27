"""Thesis bank: for each curriculum section in kanon, GPT Sol writes N contestable CKE-style theses, each with a
CKE aspect triple, from the SECTION NAME and the list of aspects that have facts — never from the facts themselves.
This mirrors the exam, where the thesis exists before any fact is retrieved.
Usage: python3 essay_v2/tools/gen_theses.py [--limit K] [--per-section N]
Writes/appends out/theses.jsonl (skips sections already present)."""
import json, os, re, sys, collections
from openai import OpenAI

ROOT = os.environ.get("ESSAY_OUT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")
KANON = os.environ.get("KANON_PATH") or os.path.join(ROOT, "kanon_clean.jsonl")
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10**9
PER = int(sys.argv[sys.argv.index("--per-section") + 1]) if "--per-section" in sys.argv else 8
MODEL = "gpt-6-sol"
origin = os.getenv("FORGEHAND_URL", "https://app.forgehand.app").rstrip("/")
client = OpenAI(api_key=os.environ["FORGEHAND_TOKEN"],
                base_url=f"{origin}/api/v1/teams/{os.environ['FORGEHAND_TEAM_ID']}/llm/v1", max_retries=0)

# CKE aspect label -> kanon aspect tags it draws facts from
ASPECT_MAP = {
    "polityczny": ["polityczny"], "społeczny": ["społeczny"], "gospodarczy": ["gospodarczy"],
    "ekonomiczny": ["gospodarczy"], "społeczno-gospodarczy": ["społeczny", "gospodarczy"],
    "kulturowy": ["kulturowy", "religijny"], "religijny": ["religijny"], "militarny": ["militarny"],
    "ustrojowy": ["ustrojowy"], "polityczno-ustrojowy": ["polityczny", "ustrojowy"],
    "dyplomatyczny": ["międzynarodowy"], "międzynarodowy": ["międzynarodowy"],
}

def norm_section(d):
    d = re.sub(r"\s+", " ", d.strip()).rstrip(".")
    return re.sub(r"\bwieku\b", "w", d)

def available_cke_aspects(counts, min_facts=4):
    out = []
    for cke, tags in ASPECT_MAP.items():
        if sum(counts.get(t, 0) for t in tags) >= min_facts:
            out.append(cke)
    return out

def load_sections():
    rows = [json.loads(l) for l in open(KANON, encoding="utf-8")]
    by = collections.defaultdict(collections.Counter)
    titles = collections.defaultdict(set)
    for r in rows:
        by[norm_section(r["dzial"])][r["aspekt"]] += 1
        for t in (r.get("tytul"), r.get("termin"), r.get("postac")):
            if t: titles[norm_section(r["dzial"])].add(t.strip())
    return by, titles

SYSTEM = """Jesteś autorem tematów wypracowań maturalnych z historii (CKE, poziom rozszerzony, formuła 2023).
Tworzysz TEZY do tematów typu: "<TEZA> Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej
argumentacji aspekty: X, Y i Z."
Wzory prawdziwych tez CKE: "Klęska Polski w 1939 roku była nieunikniona." / "Unia w Krewie okazała się bardziej
korzystna dla Litwy niż dla Polski." / "Lata 1871–1914 są niesłusznie określane jako belle époque." / "Niepodległość
Polska zawdzięczała przede wszystkim przywództwu Józefa Piłsudskiego." / "Rewolucja przemysłowa przyniosła
społeczeństwom europejskim więcej szkody niż pożytku." / "O upadku Rzeczypospolitej zadecydowały w głównej mierze
czynniki wewnętrzne." / "Blok państw komunistycznych nie był monolitem."
Zasady: jedno zdanie, 6-15 słów, bez uzasadnienia ("ponieważ"), bez wyliczania aspektów. Teza to SPORNA OCENA
(hierarchia przyczyn, superlatyw, bilans, "przełom", "nieunikniona", "niesłusznie"), z którą da się zarówno zgodzić,
jak i nie zgodzić na podstawie faktów z podręcznika. Tezy w zestawie mają dotyczyć różnych zagadnień działu, nie tego
samego w innych słowach. Do każdej tezy dobierz trójkę RÓŻNYCH, NIENAKŁADAJĄCYCH SIĘ aspektów z listy dostępnych (nie łącz np. "społeczny" ze "społeczno-gospodarczy" ani "polityczny" z "polityczno-ustrojowy"), tak jak robi CKE
(najczęściej: polityczny / społeczno-gospodarczy / kulturowy; też militarny, ustrojowy, dyplomatyczny, ekonomiczny).
Odpowiadasz wyłącznie poprawnym JSON."""

def main():
    by, titles = load_sections()
    outp = os.path.join(ROOT, os.environ.get("THESES_FILE", "theses.jsonl"))
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)["dzial"] for l in open(outp, encoding="utf-8") if l.strip()}
    n = 0
    for dzial, counts in sorted(by.items()):
        if dzial in done or n >= LIMIT:
            continue
        aspects = available_cke_aspects(counts)
        if len(aspects) < 3:
            print(f"SKIP {dzial}: only {aspects}"); continue
        tl = sorted(titles[dzial]); tl_txt = "; ".join(tl[:120])
        user = (f"Dział podstawy programowej: {dzial}\nDostępne aspekty (fakty istnieją): {', '.join(aspects)}\n"
                f"Zagadnienia, postacie i terminy, o których baza wiedzy MA fakty (tezy muszą dotyczyć TYCH zagadnień, "
                f"tak aby dało się je uzasadnić faktami z bazy we wszystkich trzech aspektach; nie wychodź poza tę listę):\n{tl_txt}\n\n"
                f"Napisz {PER} tez o różnych zagadnieniach z listy. Zwróć JSON: {{\"tezy\": [{{\"teza\": str, \"aspekty\": [3 nazwy z listy dostępnych]}} x{PER}]}}")
        resp = client.chat.completions.create(model=MODEL, reasoning_effort="low", max_completion_tokens=2500,
                                              response_format={"type": "json_object"},
                                              messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        j = json.loads(resp.choices[0].message.content)
        def disjoint(trip):
            tags = [set(ASPECT_MAP[a]) for a in trip]
            return all(not (tags[i] & tags[k]) for i in range(3) for k in range(i + 1, 3))
        good = [t for t in j.get("tezy", []) if len(t.get("aspekty", [])) == 3 and len(set(t["aspekty"])) == 3
                and all(a in aspects for a in t["aspekty"]) and disjoint(t["aspekty"])]
        with open(outp, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(dzial=dzial, dostepne=aspects, tezy=good), ensure_ascii=False) + "\n")
        n += 1
        print(f"[{n}] {dzial} | {len(good)}/{len(j.get('tezy', []))} valid | tokens {resp.usage.prompt_tokens}+{resp.usage.completion_tokens}")
        for t in good: print(f"      - {t['teza']}  [{', '.join(t['aspekty'])}]")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
