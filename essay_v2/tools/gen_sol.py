"""Generate N structured CKE-style essay records with GPT Sol from kanon.jsonl facts.
Tiny-batch tool: python3 gen_sol.py <seed> <n> [--reasoning low]
Writes out/sol/batch_<seed>_v<PROMPT_VERSION>.jsonl and prints each essay + auto checks. Costs money per call."""
import json, os, random, re, sys, time, collections
from openai import OpenAI

seed, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[1].isdigit() else (0, 0)
REASON = sys.argv[sys.argv.index("--reasoning") + 1] if "--reasoning" in sys.argv else "low"
MODEL = "gpt-6-sol"
origin = os.getenv("FORGEHAND_URL", "https://app.forgehand.app").rstrip("/")
client = OpenAI(api_key=os.environ["FORGEHAND_TOKEN"],
                base_url=f"{origin}/api/v1/teams/{os.environ['FORGEHAND_TEAM_ID']}/llm/v1", max_retries=0)

ROOT = os.environ.get("ESSAY_OUT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")  # gitignored data dir
KANON = os.environ.get("KANON_PATH") or os.path.join(ROOT, "kanon.jsonl")  # git show origin/bartek:maturaai-bartosz/knowledge_base/kanon.jsonl > out/kanon.jsonl
rows = [json.loads(l) for l in open(KANON, encoding="utf-8")]
rng = random.Random(seed)
by_dz = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by_dz[r["dzial"]][r["aspekt"]].append(r)

PROMPT_VERSION = int(sys.argv[sys.argv.index("--variant") + 1][1:]) if "--variant" in sys.argv else 5  # house style

VARIANT_ADDON = {
    3: "",
    4: """
POGŁĘBIENIE (wariant 4): każdy fakt przedstawiasz jako łańcuch trzech kroków w dwóch-trzech prostych zdaniach:
(1) co się stało (fakt z datą), (2) jaki mechanizm uruchomił (przyczyna → skutek), (3) co to oznacza dla tezy.
Używaj terminologii historycznej podanej przy faktach (pole termin), np. zamiast "szlachta zyskała" pisz
"ruch egzekucyjny umocnił pozycję szlachty". Limit słów rośnie do 400-430; akapity po 105-120 słów.""",
    5: """
ARGUMENT NAJPIERW (wariant 5): każdy akapit otwiera KONKRETNE twierdzenie o tym aspekcie, które samo w sobie jest
argumentem za tezą (złe: "W aspekcie politycznym teza się potwierdza"; dobre: "Politycznie rządy X oznaczały
podporządkowanie państwa jednej grupie"). Fakty służą jako dowody tego twierdzenia. W każdym akapicie jedno zdanie
refleksji, które ocenia znaczenie lub wskazuje granicę zjawiska ("Warto zauważyć, że...", "Istotne jest, że...").
Zakończenie wyciąga wniosek ogólny, nie powtarza wstępu. Limit słów bez zmian (380-400).""",
}
VARIANT_ADDON[6] = VARIANT_ADDON[5] + """
REJESTR ZAAWANSOWANY (wariant 6): UCHYLAM ograniczenia językowe z sekcji JĘZYK. Pisz jak bardzo dobry maturzysta
o ambicjach historycznych: dopuszczalne zdania wielokrotnie złożone (do 30 słów), precyzyjna terminologia historyczna
(pole termin i własna), bogatsze słownictwo oceniające (np. "przesądziło", "ugruntowało", "zantagonizowało",
"legitymizacja", "centralizacja"), imiesłowowe równoważniki zdań. Nadal: bez metafor, bez ogólników, każde zdanie niesie treść."""
VARIANT_ADDON[7] = VARIANT_ADDON[5] + """
KONTRARGUMENT (wariant 7): w każdym akapicie jedno zdanie typu "Choć [fakt lub okoliczność, która mogłaby osłabiać
tezę], to jednak [dlaczego mimo to teza się broni]". Zastrzeżenie musi wynikać z podanych faktów lub z ogólnej wiedzy
bez nowych dat i nazw, a zdanie MUSI kończyć się na korzyść tezy. To pokazuje wnikliwość analizy, nie zmienia stanowiska."""
VARIANT_ADDON[8] = VARIANT_ADDON[5] + """
GĘSTA FAKTOGRAFIA (wariant 8): użyj 5-6 faktów w każdym akapicie (wszystkich podanych dla aspektu, jeśli jest ich
tyle). Każdy fakt nadal ma jedno zdanie wyjaśniające, ale krótsze. Limit słów rośnie do 420-450; akapity po 115-130 słów."""
VARIANT_ADDON[9] = VARIANT_ADDON[5] + """
ŁAŃCUCH AKAPITÓW (wariant 9): drugi i trzeci akapit otwiera zdanie, które wiąże ten aspekt przyczynowo z poprzednim
(np. "Zmiany ustrojowe przełożyły się bezpośrednio na gospodarkę, ponieważ..."). Pierwszy akapit otwiera jedno zdanie
ogólnego tła historycznego (bez nowych dat i nazw). Zakończenie pokazuje, jak trzy aspekty wzajemnie się wzmacniały,
i formułuje wniosek ogólny. Przestawienie akapitów musiałoby zaburzyć tok rozumowania."""

def sample_facts(fs, k):
    fs = fs[:]; rng.shuffle(fs)
    dated = [f for f in fs if f["data"]]; undated = [f for f in fs if not f["data"]]
    return (dated[:k - 2] + undated[:2] + dated[k - 2:])[:k]

def pick_material():
    """Pick a section and three CKE-style aspects. Half the time, when the section has both
    'społeczny' and 'gospodarczy' facts, merge them into CKE's usual 'społeczno-gospodarczy'."""
    dz = [d for d, a in by_dz.items() if sum(len(v) >= 4 for v in a.values()) >= 3]
    d = rng.choice(dz)
    avail = {a: v for a, v in by_dz[d].items() if len(v) >= 4}
    groups = []  # list of (label, [facts])
    if "społeczny" in avail and "gospodarczy" in avail and len(avail) >= 4 and rng.random() < 0.5:
        groups.append(("społeczno-gospodarczy", sample_facts(avail["społeczny"], 3) + sample_facts(avail["gospodarczy"], 3)))
        rest = [a for a in avail if a not in ("społeczny", "gospodarczy")]
    else:
        rest = list(avail)
    rng.shuffle(rest)
    for a in rest[:3 - len(groups)]:
        groups.append((a, sample_facts(avail[a], 6)))
    rng.shuffle(groups)
    aspects = [g[0] for g in groups]
    mat, fid = [], 1
    for a, fs in groups:
        for f in fs:
            mat.append(dict(id=f"F{fid}", aspekt=a, fakt=f["fakt"], data=f["data"], postac=f["postac"], termin=f["termin"])); fid += 1
    return d, aspects, mat

FORMULA = ("{teza} Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji "
           "aspekty: {a1}, {a2} i {a3}.")
STANCES = {
    "zgadzam": "Zgadzam się z tezą. Cała argumentacja potwierdza tezę.",
    "kategoryczna": ("Teza jest zbyt kategoryczna: zawiera trafne jądro, ale wymaga zastrzeżeń. Argumentacja pokazuje, "
                     "co w tezie jest słuszne i gdzie jest uproszczeniem, bez zaprzeczania własnemu stanowisku."),
}

SYSTEM = """Jesteś doświadczonym nauczycielem historii i egzaminatorem CKE. Piszesz wzorcowe wypracowania maturalne
(historia, poziom rozszerzony). Posłużą one jako dane treningowe dla bardzo małego modelu językowego, więc muszą być
proste, jednolite i całkowicie przewidywalne w budowie.

JĘZYK (najważniejsze):
- polszczyzna dobrego ucznia liceum, nie historyka: słownictwo szkolne, bez metafor, bez wyszukanych sformułowań;
- jedno zdanie = jedna myśl; zdania krótkie lub średnie (8-18 słów); unikaj zdań wielokrotnie złożonych i wtrąceń;
- pisz konkretnie: kto, co, kiedy, jaki skutek. Nie pisz ogólników typu "miało to dalekosiężne konsekwencje";
- każdy fakt wiąż z tezą jednym prostym zdaniem wyjaśniającym (np. "Świadczy to o tym, że...", "Dzięki temu...",
  "Oznaczało to, że..."), a nie rozbudowaną analizą;
- używaj prostych wskaźników zespolenia: "Ponadto", "Z kolei", "W rezultacie", "Dlatego", "Podobnie", "Zatem".

BUDOWA (zawsze taka sama):
- wstęp (50-60 słów): jedno-dwa zdania osadzające temat w czasie i przestrzeni, potem wyraźne stanowisko
  "Zgadzam się z tą tezą" i jedno zdanie zapowiadające trzy aspekty;
- trzy akapity (95-110 słów każdy), jeden na aspekt, w podanej kolejności: zdanie otwierające, które wiąże aspekt z tezą;
  3-4 fakty z listy, każdy z jednym zdaniem wyjaśniającym, jak potwierdza tezę; zdanie zamykające akapit;
- zakończenie (40-50 słów): powtórzenie stanowiska i krótkie podsumowanie trzech aspektów, bez nowych faktów;
- razem 380-400 słów; bez nagłówków, list, pogrubień, cudzysłowów wokół tezy.

FAKTY:
- używasz WYŁĄCZNIE faktów, dat, nazwisk, nazw i wydarzeń z podanej listy. Nie dodajesz żadnej innej daty, osoby,
  nazwy ani wydarzenia. Ogólne sformułowania bez nowych faktów są dozwolone;
- nie zmieniasz sensu faktów. Jeśli fakt nie pasuje do tezy, po prostu go nie używasz.

Odpowiadasz wyłącznie poprawnym JSON."""
SYSTEM = SYSTEM + VARIANT_ADDON[PROMPT_VERSION]

def user_msg(dz, aspects, mat, stance):
    fl = "\n".join(f'{m["id"]} [{m["aspekt"]}] {m["fakt"]}' + (f' ({m["data"]})' if m["data"] else "")
                   + (f' [termin: {m["termin"]}]' if m.get("termin") else "") for m in mat)
    return f"""Dział podstawy programowej: {dz}
Aspekty (w tej kolejności): {aspects[0]}, {aspects[1]}, {aspects[2]}

Lista faktów:
{fl}

Krok 1. Sformułuj TEZĘ w stylu CKE: JEDNO krótkie zdanie (najwyżej 15 słów), BEZ uzasadnienia (bez "ponieważ",
"gdyż", "dzięki temu, że"). Teza MUSI być SPORNĄ OCENĄ, z którą rozsądny uczeń mógłby się nie zgodzić: hierarchia
przyczyn ("przede wszystkim", "głównie"), superlatyw ("najważniejszy", "najwybitniejszy", "największy"), bilans ("więcej
szkody niż pożytku", "bardziej korzystna dla X niż dla Y"), ocena ("przełom", "porażka", "sukces", "nieunikniona",
"niesłusznie nazywana..."). Teza NIE może być opisem ani wyliczeniem aspektów (złe: "X był okresem przemian politycznych,
społecznych i kulturowych"). Teza dotyczy tego działu i da się ją potwierdzić powyższymi faktami we wszystkich trzech
aspektach. Wzory: "X był najwybitniejszym władcą...", "Y przyniosło więcej szkody niż pożytku.", "Z zawdzięczała przede
wszystkim...", "Rok N był przełomem w...", "Klęska W była nieunikniona."
Temat wypracowania powstanie przez wstawienie tezy do formuły: "{FORMULA}"

Krok 2. Stanowisko, które MUSISZ przyjąć: {STANCES[stance]}

Krok 3. Napisz wypracowanie.

Zwróć JSON o polach:
{{"teza": str, "temat": str (pełna formuła z tezą i aspektami), "stanowisko": "{stance}",
 "wstep": str, "akapity": [{{"aspekt": str, "fakty_uzyte": [ids], "tekst": str}} x3], "zakonczenie": str}}"""

YEAR = re.compile(r"\b(\d{3,4})\b")
def years(s): return set(YEAR.findall(s))

def check(rec, mat):
    ids = {m["id"] for m in mat}
    full = " ".join([rec["wstep"]] + [a["tekst"] for a in rec["akapity"]] + [rec["zakonczenie"]])
    words = len(re.sub(r"\s+", " ", full).split())
    fact_years = set()
    for m in mat: fact_years |= years(m["data"]) | years(m["fakt"])
    extra_years = years(full) - fact_years
    bad_ids = [i for a in rec["akapity"] for i in a["fakty_uzyte"] if i not in ids]
    md = bool(re.search(r"[*#]|\n-", full))
    return dict(words=words, teza_words=len(rec["teza"].split()), extra_years=sorted(extra_years), bad_ids=bad_ids, markdown=md,
                n_par=len(rec["akapity"]), aspects_ok=[a["aspekt"] for a in rec["akapity"]] == [m for m in dict.fromkeys(x["aspekt"] for x in mat)])

def usage():
    import httpx
    r = httpx.get(f"{origin}/api/v1/teams/{os.environ['FORGEHAND_TEAM_ID']}/llm/usage",
                  headers={"Authorization": f"Bearer {os.environ['FORGEHAND_TOKEN']}"}, timeout=30)
    return r.json()["spentMicroUsd"] / 1e6

def main():
    os.makedirs(os.path.join(ROOT, "sol"), exist_ok=True)
    out = open(os.path.join(ROOT, "sol", f"batch_{seed}_v{PROMPT_VERSION}.jsonl"), "a")
    stance_cycle = ["zgadzam"]
    spent0 = usage()
    for i in range(n):
        dz, aspects, mat = pick_material()
        stance = stance_cycle[i % len(stance_cycle)]
        t = time.time()
        resp = client.chat.completions.create(
            model=MODEL, reasoning_effort=REASON, max_completion_tokens=3000,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(dz, aspects, mat, stance)}])
        txt = resp.choices[0].message.content
        rec = json.loads(txt)
        chk = check(rec, mat)
        u = resp.usage
        record = dict(seed=seed, i=i, prompt_version=PROMPT_VERSION, dzial=dz, aspekty=aspects, stance=stance, material=mat, essay=rec, check=chk,
                      usage=dict(prompt=u.prompt_tokens, completion=u.completion_tokens), model=MODEL, reasoning=REASON)
        out.write(json.dumps(record, ensure_ascii=False) + "\n"); out.flush()
        spent = usage()
        print("=" * 100)
        print(f"[{i+1}/{n}] DZIAŁ: {dz} | stance={stance} | {time.time()-t:.0f}s | tokens in {u.prompt_tokens} out {u.completion_tokens} | "
              f"cost this call ${spent-spent0:.3f} | total spent ${spent:.3f}")
        spent0 = spent
        print(f"CHECK: {chk}")
        print(f"\nTEMAT: {rec['temat']}\n")
        print(rec["wstep"], "\n")
        for a in rec["akapity"]:
            print(f"[{a['aspekt']} | {a['fakty_uzyte']}]\n{a['tekst']}\n")
        print(rec["zakonczenie"])
        print("\nFAKTY PODANE:")
        for m in mat: print(f"  {m['id']} [{m['aspekt']}] {m['fakt'][:100]} {('('+m['data']+')') if m['data'] else ''}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
