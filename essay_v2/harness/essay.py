"""Exam-time essay harness. Item 26 text in -> one essay out.

    python3 essay_v2/harness/essay.py --topic-id 2025-1 [--model bielik-1.5b-v3] [--n 6]
    python3 essay_v2/harness/essay.py --item-text "Zadanie zawiera trzy tematy. ... 1. ... 2. ... 3. ..."
    python3 essay_v2/harness/essay.py --eval-all [--samples 3]      # all held-out aspekty topics -> out/eval/

Pipeline (code owns everything except the body-paragraph prose):
  1. parse the 1-3 topics; keep the 'aspekty' ones; choose the one the KB covers best (retrieval strength)
  2. retrieve 6 thesis-relevant facts per CKE aspect from the KB (stem overlap + section bonus, deduped)
  3. intro + conclusion from templates that quote the thesis verbatim (no inflection, always grammatical)
  4. per aspect: n candidate paragraphs from the model -> hard gates (no foreign years/names, stance not negated,
     no markdown/echo, length band) -> heuristic score -> best; none pass -> n more once -> tier-3 fallback
     (the fact sentences themselves joined with fixed connectives)
  5. assemble; if < MIN_WORDS regenerate the shortest paragraph with 2n candidates
Output record has the same shape as the Sol records, so judge_sol.py and build_viewer.py work on it."""
import json, os, re, sys, random, time, collections, urllib.request, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
ROOT = os.environ.get("ESSAY_OUT") or os.path.join(os.path.dirname(V2), "out")
KANON = os.environ.get("KANON_PATH") or os.path.join(ROOT, "kanon_clean.jsonl")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

ASPECT_MAP = {
    "polityczny": ["polityczny"], "społeczny": ["społeczny"], "gospodarczy": ["gospodarczy"],
    "ekonomiczny": ["gospodarczy"], "społeczno-gospodarczy": ["społeczny", "gospodarczy"],
    "kulturowy": ["kulturowy", "religijny"], "religijny": ["religijny"], "militarny": ["militarny"],
    "ustrojowy": ["ustrojowy"], "polityczno-ustrojowy": ["polityczny", "ustrojowy"],
    "dyplomatyczny": ["międzynarodowy"], "międzynarodowy": ["międzynarodowy"],
}
DEFAULT_ASPECTS = ["polityczny", "społeczno-gospodarczy", "kulturowy"]
MIN_WORDS, PAR_MIN, PAR_MAX = 340, 70, 230   # PAR_MAX was 170: Bielik's natural paragraph at num_predict=320 is ~190-215 words

# ---------------------------------------------------------------- KB + retrieval
STOP = set("oraz albo lub jako przez tego jego jej tym była było były jest przede wszystkim bardziej niż więcej okres okresie dla "
           "wieku roku latach lata stulecia stuleciu okresu".split())   # date scaffolding words sit in nearly every fact
def stems(s): return {w[:5] for w in re.findall(r"\w+", s.lower()) if len(w) >= 4 and w not in STOP}
LOWER_VOCAB = set()   # every lowercase token in the KB; filled by KB(); a word never seen lowercase is a proper noun
def norm_section(d):
    d = re.sub(r"\s+", " ", d.strip()).rstrip("."); d = re.sub(r"\bwieku\b", "w", d)
    return re.sub(r"\b([IVX]+) i ([IVX]+) w\b", r"\1–\2 w", d)

ROMAN = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,"XI":11,"XII":12,"XIII":13,"XIV":14,
         "XV":15,"XVI":16,"XVII":17,"XVIII":18,"XIX":19,"XX":20,"XXI":21}
ERA_KEYWORDS = [  # (regex on lowercase thesis, (lo, hi))
    (r"staro[żz]ytn", (-3000, 476)), (r"[śs]redniowiecz|wiek[óo]w [śs]rednich", (476, 1492)), (r"nowo[żz]ytn", (1492, 1789)),
    (r"o[śs]wiecen", (1690, 1800)), (r"renesans|odrodzeni", (1400, 1600)), (r"reformac", (1517, 1648)),
    (r"napoleo", (1799, 1815)), (r"zabor|rozbior", (1772, 1918)), (r"mi[ęe]dzywojenn|ii rzeczpospolit|ii rp\b", (1918, 1939)),
    (r"i wojn[ay] [śs]wiatow", (1914, 1918)), (r"ii wojn[ay] [śs]wiatow", (1939, 1945)), (r"zimn\w* wojn", (1947, 1991)),
    (r"\bprl\b|polsk\w* ludow", (1944, 1989)), (r"komunist", (1917, 1991)), (r"jagiellon", (1386, 1572)),
    (r"piast", (960, 1370)), (r"rzeczpospolit\w* obojga|szlacheck", (1569, 1795)), (r"rewolucj\w* przemys", (1760, 1914)),
]
def period_of(text):
    """(lo, hi) years implied by a thesis: explicit years, 'XVIII w.', 'II połowie XIX w.', 'lata 50. XX w.'; None if none."""
    t = text
    yrs = [int(y) for y in re.findall(r"\b(1\d{3}|20\d{2}|[5-9]\d{2})\b", t)]
    lo, hi = (min(yrs), max(yrs)) if yrs else (None, None)
    for m in re.finditer(r"\b(?:(I|II)\s+po[łl]owi\w*\s+)?(X{0,2}(?:IX|IV|V?I{0,3}))\s*(?:w\.|w\b|wiek)", t):
        half, rom = m.group(1), m.group(2)
        if rom not in ROMAN: continue
        c = ROMAN[rom]; a, b = (c - 1) * 100 + 1, c * 100
        if half == "I": b = a + 49
        if half == "II": a = a + 50
        lo, hi = (a if lo is None else min(lo, a)), (b if hi is None else max(hi, b))
    for m in re.finditer(r"\b(X{0,2}(?:IX|IV|V?I{0,3}))-wieczn", t):
        c = ROMAN.get(m.group(1))
        if c: a, b = (c - 1) * 100 + 1, c * 100; lo, hi = (a if lo is None else min(lo, a)), (b if hi is None else max(hi, b))
    if lo is None:
        tl = t.lower()
        for rx, (a, b) in ERA_KEYWORDS:
            if re.search(rx, tl): lo, hi = a, b; break
    for m in re.finditer(r"lat(?:a|ach)\s+(\d)0\.\s*(X{0,2}(?:IX|IV|V?I{0,3}))\s*(?:w|wiek)", t):
        c = ROMAN.get(m.group(2)); 
        if c: a = (c - 1) * 100 + int(m.group(1)) * 10; lo, hi = (a if lo is None else min(lo, a)), (a + 9 if hi is None else max(hi, a + 9))
    if lo is not None and lo == hi and len(yrs) == 1:
        lo, hi = lo - 12, hi + 1          # a single year in a thesis: causes lead up to it, aftermath is off-topic
    return (lo, hi) if lo is not None else None

def fact_year(r):
    m = re.search(r"\b(1\d{3}|20\d{2}|[1-9]\d{2})\b", r.get("data", "") or "")
    return int(m.group(1)) if m else None

def period_score(r, win):
    """+1 inside the window (±15y), 0 if undated or no window, -3 if far outside (>60y)."""
    if not win: return 0.0
    y = fact_year(r)
    if y is None: return 0.0
    lo, hi = win
    tol = 2 if (hi - lo) < 30 else 5
    if lo - tol <= y <= hi + tol: return 1.0
    d = min(abs(y - lo), abs(y - hi))
    return -1.5 if d <= 40 else (-3.0 if d <= 80 else -9.0)

META_SECTION = "Historia jako nauka"   # historiography: never essay material, pollutes section lock

class KB:
    def __init__(self, path=KANON):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        df = collections.Counter()
        for r in self.rows:
            r["_stems"] = stems(" ".join([r["fakt"], r.get("postac", ""), r.get("termin", ""), r.get("tytul", "")]))
            r["_dz"] = norm_section(r["dzial"])
            df.update(r["_stems"])
        import math
        N = len(self.rows)
        self.idf = {st: math.log((N + 1) / (c + 1)) + 1.0 for st, c in df.items()}
        self.by_tag = collections.defaultdict(list)
        for r in self.rows: self.by_tag[r["aspekt"]].append(r)
        self._dz_cache = {}
        LOWER_VOCAB.update(w for r in self.rows for w in re.findall(r"\b[a-ząęółśżźćń]+\b", r["fakt"]))

    def wov(self, q, r):
        """IDF-weighted stem overlap: rare stems ('przem', 'jagie') dominate common ones ('społe', 'europ')."""
        return sum(self.idf.get(st, 1.0) for st in (q & r["_stems"]))

    def section(self, teza):
        """Thesis-level section lock, computed once per thesis: IDF-weighted overlap, rows far outside the thesis's
        period count little, so 'Rewolucja przemysłowa ... społeczeństwom europejskim' lands in the 19th c., not 1968."""
        if teza not in self._dz_cache:
            q = {st for st in stems(teza) if not st.isdigit()}; win = period_of(teza); w = collections.Counter()
            rare = collections.defaultdict(float)   # best idf of a matched stem per section: generic words must not lock
            for r in self.rows:
                if r["_dz"] == META_SECTION: continue
                ov = self.wov(q, r)
                if len(q & r["_stems"]) >= 1 and ov >= 3.0:
                    ps = period_score(r, win)
                    if ps <= -3: continue
                    w[r["_dz"]] += ov * ov
                    rare[r["_dz"]] = max(rare[r["_dz"]], max(self.idf.get(st, 1.0) for st in (q & r["_stems"])))
            w = collections.Counter({dz: v for dz, v in w.items() if rare[dz] >= 3.5})
            best = w.most_common(1)[0][0] if w else None
            def in_win_count(dz):
                return sum(1 for r in self.rows if r["_dz"] == dz and period_score(r, win) > 0)
            if win and (best is None or w[best] < 30 or in_win_count(best) < 5):
                # words did not discriminate (e.g. 'belle époque'): take the section with most facts inside the window
                cover = collections.Counter(r["_dz"] for r in self.rows if r["_dz"] != META_SECTION and period_score(r, win) > 0)
                if cover:
                    # tie-break by section-name overlap with the thesis ('średniowiecznej Europy' ~ 'średniowiecza')
                    best = max(cover, key=lambda dz: cover[dz] + 60 * len(q & stems(dz)))
            self._dz_cache[teza] = best
        return self._dz_cache[teza]

    def window(self, teza):
        """Period window: from the thesis text if it has one, else the 10th-90th percentile of the locked section's
        dated facts (so 'Rewolucja przemysłowa ...' gets the 19th century, not 1968)."""
        win = period_of(teza)
        if win: return win
        dz = self.section(teza)
        ys = sorted(y for r in self.rows if r["_dz"] == dz for y in [fact_year(r)] if y)
        if len(ys) < 5: return None
        return (ys[len(ys) // 10] - 10, ys[(len(ys) * 9) // 10] + 10)

    def retrieve(self, teza, cke_aspect, k=6, rng=None):
        q = stems(teza); win = self.window(teza); best_dz = self.section(teza)
        tags = ASPECT_MAP.get(cke_aspect, [cke_aspect])
        pool = []
        def add(r, in_tag):
            ov = self.wov(q, r); in_dz = r["_dz"] == best_dz; ps = period_score(r, win)
            if ps <= -9: return
            if ov == 0 and not in_dz: return
            if r["_dz"] == META_SECTION and not re.search(r"historyk|historiograf|szko[łl]\w* historyczn", teza.lower()): return
            if not in_dz and win and fact_year(r) is None: ps -= 2.0     # undated off-section rows dodge the period check
            score = ov + ps + (8.0 if in_dz else 0) + (0.3 if r["data"] else 0) + (0 if in_tag else -2.5) + (rng.random() * 0.2 if rng else 0)
            pool.append((score, in_dz or ov >= 12.0, r))
        for tag in tags:
            for r in self.by_tag.get(tag, []): add(r, True)
        # in-section facts of other aspects are better than off-section junk when the aspect pool is thin
        if sum(1 for _, _, r in pool if r["_dz"] == best_dz) < 4 and best_dz:
            for r in self.rows:
                if r["_dz"] == best_dz and r["aspekt"] not in tags: add(r, False)
        pool.sort(key=lambda x: -x[0])
        strong = [(s, r) for s, st, r in pool if st]
        weak = [(s, r) for s, st, r in pool if not st]
        pool = strong + weak[:max(0, 3 - len(strong))]
        chosen, seen = [], []
        for s, r in pool:
            if any(len(r["_stems"] & t) / max(1, len(r["_stems"] | t)) > 0.4 for t in seen): continue
            chosen.append((s, r)); seen.append(r["_stems"])
            if len(chosen) == k: break
        return chosen  # [(score, row)]

# ---------------------------------------------------------------- topic parsing / choice
FORMULA_RE = re.compile(r"uwzględniając w swojej argumentacji aspekt(?:y)?:?\s*(.+?)\.\s*$", re.S)
def parse_topics(item_text):
    """Return [{'num','teza','tekst','aspekty' or None}] from item 26 text (or a single topic string)."""
    t = re.sub(r"\s+", " ", item_text).strip()
    parts = re.split(r"(?:^|\s)(\d)\.\s+(?=[A-ZŁŚŻŹĆŃÓ„\"])", t)
    topics = []
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            topics.append((parts[i], parts[i + 1].strip()))
    else:
        topics.append(("1", t))
    out = []
    for num, txt in topics:
        m = re.search(r"^(.*?)\s*Zajmij stanowisko", txt)
        if not m: continue
        teza = m.group(1).strip().strip("„”\"")
        asp = None
        am = FORMULA_RE.search(txt)
        if am:
            raw = am.group(1)
            asp = [a.strip().strip(".").lower() for a in re.split(r",|\bi\b", raw) if a.strip()]
            asp = [a for a in asp if a in ASPECT_MAP]
            if len(asp) != 3: asp = None
        out.append(dict(num=num, teza=teza, tekst=txt, aspekty=asp))
    return out

def choose_topic(topics, kb):
    cands = [t for t in topics if t["aspekty"]] or [dict(t, aspekty=DEFAULT_ASPECTS) for t in topics[:1]]
    for t in cands:
        t["coverage"] = sum(sum(s for s, _ in kb.retrieve(t["teza"], a, k=4)) for a in t["aspekty"])
    return max(cands, key=lambda t: t["coverage"])

# ---------------------------------------------------------------- templates (thesis quoted verbatim)
def q(teza):
    return "„" + teza.strip().rstrip(".") + "”"
def intro(teza, aspekty):
    return (f"Temat skłania do oceny tezy: {q(teza)}. Zgadzam się z tą tezą. Uzasadnię swoje stanowisko, "
            f"odwołując się do trzech aspektów: {aspekty[0]}ego, {aspekty[1]}ego i {aspekty[2]}ego.")
def conclusion(teza, aspekty):
    return (f"Podsumowując, przedstawione argumenty z zakresu aspektów {aspekty[0]}ego, {aspekty[1]}ego i {aspekty[2]}ego "
            f"potwierdzają tezę: {q(teza)}. Zgadzam się z nią, ponieważ w każdym z tych obszarów fakty wskazują ten sam kierunek.")
def adj(a):
    """'polityczny' -> 'polityczny' (nominative used in intro with -ego genitive suffix appended by caller)."""
    return a

def fix_case(aspekty):
    """Genitive singular of the aspect adjectives for 'aspektu X-ego'. Compound 'społeczno-gospodarczy' -> 'społeczno-gospodarczego'."""
    return [re.sub(r"y$", "", a) for a in aspekty]

# ---------------------------------------------------------------- model call (KNOWN_TRAPS: system msg, no stop list)
SYSTEM = ("Jesteś maturzystą piszącym wypracowanie z historii na poziomie rozszerzonym. Piszesz poprawną polszczyzną, "
          "ciągłym tekstem, bez nagłówków, list i pogrubień. Używasz wyłącznie faktów podanych w poleceniu; nie dodajesz "
          "innych dat, nazw ani wydarzeń. Zgadzasz się z tezą i konsekwentnie ją uzasadniasz.")
def par_prompt(temat, aspekt, facts):
    fl = "\n".join(f"- {r['fakt']}" + (f" ({r['data']})" if r["data"] else "") for r in facts)
    return (f"Temat wypracowania:\n{temat}\n\nStanowisko: zgadzam się z tezą.\n\nAspekt tego akapitu: {aspekt}.\n"
            f"Fakty, których możesz użyć (tylko te):\n{fl}\n\n"
            f"Napisz jeden akapit argumentacyjny (ok. 100-120 słów), który uzasadnia tezę w tym aspekcie.")

def call(model, user, temperature=0.7, num_predict=320):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       "options": {"temperature": temperature, "num_predict": num_predict}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]["content"]

# ---------------------------------------------------------------- gates + scoring
YEAR = re.compile(r"\b(\d{3,4})\b")
NEG = re.compile(r"nie zgadzam|nie można się zgodzić|teza (jest )?(błędna|fałszywa|nietrafna)|nie potwierdza(ją)? tezy|obala tez|przeciw tezie", re.I)
HEDGE = re.compile(r"podane fakty|brak (danych|faktów|informacji)|trudno (ocenić|stwierdzić)|nie pozwalają|nie wiadomo", re.I)
ECHO = re.compile(r"Fakty, których|Napisz jeden akapit|Temat wypracowania|Aspekt tego akapitu|Stanowisko:|Teraz (zacznij|napisz)|"
                  r"Dodatkowe informacje|Pamiętaj|To zdanie uzasadnia|Ten akapit|Powyższy akapit|Uwaga:|Fakty:|możesz (dodać|użyć)|"
                  r"Poniżej znajduje|Tekst ten|Moje stanowisko i argumenty|Uzupełnienie akapitu|Oto (jeden )?akapit|Słowniczek|"
                  r"\b(Zaczęłaś|Zaczynasz|Zgadzasz|Powinieneś|Powinnaś|Musisz|Wybierasz|Zawierasz)\b|możesz (też |również )?(dodać|napisać|użyć)", re.I)
LABEL = re.compile(r"^[>\s\"„”']*(?:(?:Akapit[^:]{0,30}|Wypracowanie|Autor|Odpowiedź|Tekst|Argumentacja)\s*:\s*)?[>\s\"„”']*")
# inline heading of 1-4 words at a sentence start, followed by a capitalised sentence: 'Teza: Rewolucja...', 'Wpływ na społeczeństwo: W...'
HEADING = re.compile(r"(^|(?<=[.!?]\s))[A-ZŻŹĆŃÓŚŁĘĄ][\wąęółśżźćń-]*(?:\s[\wąęółśżźćń-]+){0,3}:\s+(?=[A-ZŻŹĆŃÓŚŁĘĄ„\"])")
CONNECT = ["ponadto", "z kolei", "dlatego", "zatem", "w rezultacie", "dzięki temu", "oznacza", "świadczy", "pokazuje", "podobnie", "w konsekwencji"]

def clean(txt):
    t = re.sub(r"\*\*|__|^#+\s*", "", txt, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    t = LABEL.sub("", t, count=1)                # drop 'Akapit argumentacyjny:', 'Wypracowanie:', leading '>' / quotes
    t = HEADING.sub("", t)                       # drop inline headings the model uses to structure its answer
    for _ in range(2):                           # cut where the model starts echoing scaffolding / commenting on itself
        e = ECHO.search(t)
        if not e: break
        if e.start() > 0: t = t[:e.start()]
        else: t = re.sub(r"^[^.!?]*[.!?]\s*", "", t, count=1)   # echo is the opening sentence ('Poniżej znajduje się...'): drop it
    m = re.search(r"^(.*?[.!?])[^.!?]*$", t)   # trim to last full sentence
    return (m.group(1) if m else t).strip(' "„”')

def cap_tokens(s): return set(re.findall(r"\b[A-ZŁŚŻŹĆŃÓ][a-ząęółśżźćń]{3,}", s))

def gate(par, facts, teza):
    words = len(par.split())
    if words < PAR_MIN or words > PAR_MAX: return f"len {words}"
    allowed_text = " ".join(r["fakt"] + " " + r["data"] for r in facts) + " " + teza
    bad_years = YEAR.findall(par) and (set(YEAR.findall(par)) - set(YEAR.findall(allowed_text)))
    if bad_years: return f"foreign years {sorted(bad_years)}"
    if NEG.search(par): return "negates stance"
    if ECHO.search(par): return "prompt echo"
    if re.search(r"[*#•]|\n- |\s-\s(?=[\wŻŹĆŃÓŚŁĘĄ])|(?<!\d)\d\.\s|\(\.\.\.\)", par): return "markdown/list"   # ' - x' = bullets after whitespace collapse; '(...)' = spliced quotes; '1789. ' is a year, not a list
    if re.search(r"[\u4e00-\u9fff\u0400-\u04ff]", par): return "foreign script"
    # sentences that mention nothing from the facts (pure invention) must be a minority
    fs = set().union(*(r["_stems"] for r in facts)) | stems(teza)   # restating the thesis is anchored too
    sents = [s for s in re.split(r"(?<=[.!?])\s+", par) if s.strip()]
    unanchored = sum(1 for s in sents if not (stems(s) & fs))
    if sents and unanchored / len(sents) > 0.5: return f"unanchored {unanchored}/{len(sents)}"
    # capitalised names not present anywhere in facts/thesis (allow sentence starts, also after .” ." .) )
    starts = set(re.findall(r"(?:^|[.!?…][”\"')]*\s+[„\"(]*)([A-ZŁŚŻŹĆŃÓ][a-ząęółśżźćń]{3,})", par))
    allowed_caps = cap_tokens(allowed_text) | {s.split()[0].strip(",.") for s in sents if s.split()} | starts
    foreign = {c for c in cap_tokens(par) if c not in allowed_caps and c[:5] not in {a[:5] for a in allowed_caps}}
    if len(foreign) > 1: return f"foreign names {sorted(foreign)[:3]}"
    return None

def score(par, facts):
    fs_per_fact = [r["_stems"] for r in facts]
    ps = stems(par)
    covered = sum(1 for f in fs_per_fact if len(f & ps) >= 2)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", par) if s.strip()]
    conn = sum(1 for c in CONNECT if c in par.lower())
    rep = len(sents) - len({s[:40] for s in sents})
    words = len(par.split())
    s = 3 * covered + min(conn, 4) - 3 * rep - (2 if HEDGE.search(par) else 0) - abs(words - 110) / 20
    return s

def fallback_paragraph(aspekt, facts, teza):
    """Tier 3: the KB's own sentences + fixed connectives. Shallow but grammatical and on-topic."""
    conns = ["", "Ponadto ", "Z kolei ", "Warto też zauważyć, że ", "Podobnie "]
    sents = []
    for i, r in enumerate(facts[:4]):
        f = r["fakt"].rstrip(".")
        c = conns[i % len(conns)]
        decap = bool(c) and f.split()[0].strip(",").lower() in LOWER_VOCAB   # not for proper nouns: 'Ponadto Stefan Czarniecki'
        sents.append((c + (f[0].lower() + f[1:] if decap else f)) + (f" ({r['data']})" if r["data"] and not YEAR.search(f) else "") + ".")
    head = f"W aspekcie {re.sub(r'y$', 'ym', aspekt)} teza znajduje potwierdzenie w konkretnych faktach."
    tail = "Wszystkie te fakty pokazują ten sam kierunek zmian i dlatego wspierają przyjęte stanowisko."
    return " ".join([head] + sents + [tail])

# ---------------------------------------------------------------- generation loop
def best_paragraph(model, temat, aspekt, facts, teza, n, log):
    tried = []
    for round_ in range(2):
        for i in range(n):
            raw = call(model, par_prompt(temat, aspekt, facts))
            par = clean(raw)
            g = gate(par, facts, teza)
            tried.append(dict(par=par, gate=g, score=None if g else score(par, facts)))
        ok = [t for t in tried if t["gate"] is None]
        if ok:
            best = max(ok, key=lambda t: t["score"])
            log.append(dict(aspekt=aspekt, tier=1, tried=len(tried), passed=len(ok), gates=collections.Counter(t["gate"] for t in tried if t["gate"])))
            return best["par"], 1
    log.append(dict(aspekt=aspekt, tier=3, tried=len(tried), passed=0, gates=collections.Counter(t["gate"] for t in tried if t["gate"])))
    return fallback_paragraph(aspekt, facts, teza), 3

def write_essay(item_text, kb, model, n=6, seed=0):
    rng = random.Random(seed)
    topics = parse_topics(item_text)
    topic = choose_topic(topics, kb)
    teza, aspekty = topic["teza"], topic["aspekty"]
    temat = topic["tekst"]
    material = {a: [r for _, r in kb.retrieve(teza, a, k=6, rng=rng)] for a in aspekty}
    log = []
    pars, tiers = [], []
    for a in aspekty:
        p, tier = best_paragraph(model, temat, a, material[a], teza, n, log)
        pars.append(p); tiers.append(tier)
    gen = fix_case(aspekty)
    w, z = intro(teza, gen), conclusion(teza, gen)
    total = lambda: len(" ".join([w] + pars + [z]).split())
    if total() < MIN_WORDS:
        k = min(range(3), key=lambda i: len(pars[i].split()))
        p, tier = best_paragraph(model, temat, aspekty[k], material[aspekty[k]], teza, 2 * n, log)
        if len(p.split()) > len(pars[k].split()): pars[k], tiers[k] = p, tier
    full = "\n\n".join([w] + pars + [z])
    mat = [dict(id=f"F{i+1}", aspekt=a, fakt=r["fakt"], data=r["data"], termin=r.get("termin", ""))
           for i, (a, r) in enumerate((a, r) for a in aspekty for r in material[a])]
    essay = dict(teza=teza, temat=temat, stanowisko="zgadzam", wstep=w,
                 akapity=[dict(aspekt=a, fakty_uzyte=[], tekst=p) for a, p in zip(aspekty, pars)], zakonczenie=z)
    return dict(gen="harness", model=model, n=n, seed=seed, topic_num=topic["num"], chosen_from=[t["num"] for t in topics],
                coverage=topic.get("coverage"), dzial="", aspekty=aspekty, stance="zgadzam", material=mat, essay=essay,
                tiers=tiers, log=[dict(l, gates=dict(l["gates"])) for l in log], words=total(), full_text=full,
                check=dict(words=total(), n_par=3, tiers=tiers))

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id"); ap.add_argument("--item-text"); ap.add_argument("--eval-all", action="store_true")
    ap.add_argument("--model", default="bielik-1.5b-v3:latest"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--samples", type=int, default=1); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    kb = KB()
    real = json.load(open(os.path.join(V2, "topics_real.json"), encoding="utf-8"))["tematy"]
    if a.eval_all:
        ids = [t["id"] for t in real if t["rodzaj"] == "aspekty"]
    elif a.topic_id:
        ids = [a.topic_id]
    else:
        ids = [None]
    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    tag = a.tag or re.sub(r"[^a-z0-9]+", "-", a.model.lower())
    outp = os.path.join(ROOT, "eval", f"harness_{tag}.jsonl")
    for tid in ids:
        text = a.item_text if tid is None else next(t["tekst"] for t in real if t["id"] == tid)
        for s in range(a.samples):
            t0 = time.time()
            rec = write_essay(text, kb, a.model, n=a.n, seed=s)
            rec["topic_id"] = tid; rec["dzial"] = tid or "item"
            with open(outp, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{tid} s{s}] topic {rec['topic_num']} of {rec['chosen_from']} | {rec['words']}w | tiers {rec['tiers']} | {time.time()-t0:.0f}s")
            for l in rec["log"]: print(f"    {l['aspekt']:24} tier{l['tier']} tried={l['tried']} passed={l['passed']} gates={l['gates']}")
            sys.stdout.flush()
    print("->", outp)
