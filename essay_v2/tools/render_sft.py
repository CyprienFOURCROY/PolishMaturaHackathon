"""Render judged essay records into SFT chat examples for Bielik-1.5B.
Usage: python3 essay_v2/tools/render_sft.py out/sol/essays_*_judged.jsonl [--min-score 10] [--out out/sft]
Writes out/sft/train.jsonl and out/sft/val.jsonl: {"messages":[{role,content}...], "task": ..., "src": ...}

Three task shapes from every record (so the same data serves whichever code/model split the harness ends up with):
  esej      : topic + facts (grouped by aspect)          -> full 5-paragraph essay
  akapit    : topic + stance + one aspect + its facts    -> that body paragraph
  wstep_zak : topic + stance + aspect list               -> intro + conclusion
Variety: facts shuffled within aspect, 1-2 distractor facts from another aspect/section added (never used in the
target), aspect word 'gospodarczy' sometimes shown as 'ekonomiczny'. The model must learn: use only supplied facts."""
import json, os, re, sys, random, glob

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MIN = int(sys.argv[sys.argv.index("--min-score") + 1]) if "--min-score" in sys.argv else 10
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "out/sft"
rng = random.Random(1234)

SYSTEM = ("Jesteś maturzystą piszącym wypracowanie z historii na poziomie rozszerzonym. Piszesz poprawną polszczyzną, "
          "ciągłym tekstem, bez nagłówków, list i pogrubień. Używasz wyłącznie faktów podanych w poleceniu; nie dodajesz "
          "innych dat, nazw ani wydarzeń. Zgadzasz się z tezą i konsekwentnie ją uzasadniasz.")

def fact_line(m):
    return f"- {m['fakt']}" + (f" ({m['data']})" if m.get("data") else "")

def facts_block(mat, aspects, all_mats):
    out = []
    for a in aspects:
        fs = [m for m in mat if m["aspekt"] == a]
        rng.shuffle(fs)
        # distractors: 0-2 facts from a different aspect of another record, appended, never used in target
        if rng.random() < 0.6 and all_mats:
            other = rng.choice(all_mats)
            dis = [m for m in other if m["aspekt"] != a]
            fs = fs + rng.sample(dis, min(len(dis), rng.choice([1, 1, 2])))
            rng.shuffle(fs)
        label = a
        if a == "gospodarczy" and rng.random() < 0.3:
            label = "ekonomiczny"
        out.append(f"Aspekt {label}:\n" + "\n".join(fact_line(m) for m in fs))
    return "\n\n".join(out)

def render(rec, all_mats):
    e = rec["essay"]; mat = rec["material"]; aspects = rec["aspekty"]
    temat = e["temat"]
    used_ids = {i for a in e["akapity"] for i in a["fakty_uzyte"]}
    mat_used = [m for m in mat if m["id"] in used_ids] or mat
    # keep unused-but-supplied facts too (the model must learn to select), but cap at 6 per aspect
    mat_shown = []
    for a in aspects:
        fs = [m for m in mat if m["aspekt"] == a]
        used = [m for m in fs if m["id"] in used_ids]; rest = [m for m in fs if m["id"] not in used_ids]
        rng.shuffle(rest)
        mat_shown += used + rest[: max(0, 6 - len(used))]
    body = [a["tekst"] for a in e["akapity"]]
    full = "\n\n".join([e["wstep"]] + body + [e["zakonczenie"]])
    ex = []
    # (a) full essay
    u = (f"Temat wypracowania:\n{temat}\n\nFakty, których możesz użyć (tylko te):\n\n{facts_block(mat_shown, aspects, all_mats)}\n\n"
         f"Napisz wypracowanie (wstęp ze stanowiskiem, trzy akapity — po jednym na aspekt w podanej kolejności, zakończenie; 350-420 słów).")
    ex.append(dict(task="esej", messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}, {"role": "assistant", "content": full}]))
    # (b) one paragraph per aspect
    for k, a in enumerate(aspects):
        fs = [m for m in mat_shown if m["aspekt"] == a]
        u = (f"Temat wypracowania:\n{temat}\n\nStanowisko: zgadzam się z tezą.\n\nAspekt tego akapitu: {a}.\n"
             f"Fakty, których możesz użyć (tylko te):\n{facts_block(fs, [a], all_mats).split(chr(10), 1)[1]}\n\n"
             f"Napisz jeden akapit argumentacyjny (ok. 100-120 słów), który uzasadnia tezę w tym aspekcie.")
        ex.append(dict(task="akapit", messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}, {"role": "assistant", "content": body[k]}]))
    # (c) intro + conclusion
    u = (f"Temat wypracowania:\n{temat}\n\nStanowisko: zgadzam się z tezą. Aspekty w kolejności: {', '.join(aspects)}.\n\n"
         f"Napisz wstęp (50-60 słów, ze stanowiskiem i zapowiedzią aspektów) oraz zakończenie (40-50 słów). Oddziel je pustą linią.")
    ex.append(dict(task="wstep_zak", messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}, {"role": "assistant", "content": e["wstep"] + "\n\n" + e["zakonczenie"]}]))
    return ex

recs = []
for pat in args:
    for path in glob.glob(pat):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            j = r.get("judge") or {}
            if j.get("suma") is None or j["suma"] < MIN or j.get("bledy_merytoryczne"):
                continue
            recs.append(r)
rng.shuffle(recs)
all_mats = [r["material"] for r in recs]
n_val = max(1, len(recs) // 20)
val, train = recs[:n_val], recs[n_val:]
os.makedirs(OUT, exist_ok=True)
for name, rs in (("train", train), ("val", val)):
    n = 0
    with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rs:
            for ex in render(r, all_mats):
                ex["src"] = f"{r.get('dzial','')}|{r['essay']['teza'][:60]}"
                f.write(json.dumps(ex, ensure_ascii=False) + "\n"); n += 1
    print(f"{name}: {len(rs)} essays -> {n} examples")
print(f"kept {len(recs)} records with score >= {MIN} and no flagged errors -> {OUT}/")
