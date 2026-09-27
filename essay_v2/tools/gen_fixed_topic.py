"""Write ONE essay for a FIXED CKE topic from a hand-picked fact list, with GPT Sol.
Calibration/ceiling tool only: the material is chosen by a human, so this measures judges (or the
upper bound of the writing style), never the pipeline. Never use for the held-out model evaluation.

Usage: python3 out/tools/gen_fixed_topic.py <spec.json> [--variant v5] [--weak]
spec.json: {"label": str, "temat": str, "aspekty": [3 names], "facts": [{"aspekt":..., "fakt":..., "data":..., "termin":...}]}
Appends the record to out/sol/fixed_<label>.jsonl and prints the essay."""
import json, os, sys, time
sys.argv_backup = list(sys.argv)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_sol as G  # noqa: E402  (importable thanks to the main() guard)

spec = json.load(open(sys.argv[1], encoding="utf-8"))
WEAK = "--weak" in sys.argv
variant = G.PROMPT_VERSION
mat = [dict(id=f"F{i+1}", aspekt=f["aspekt"], fakt=f["fakt"], data=f.get("data", ""), postac="", termin=f.get("termin", ""))
       for i, f in enumerate(spec["facts"])]
aspects = spec["aspekty"]

fl = "\n".join(f'{m["id"]} [{m["aspekt"]}] {m["fakt"]}' + (f' ({m["data"]})' if m["data"] else "")
               + (f' [termin: {m["termin"]}]' if m.get("termin") else "") for m in mat)
weak_rules = ""
if WEAK:
    weak_rules = """
WERSJA SŁABA (celowo): pisz jak przeciętny, słabo przygotowany uczeń. Użyj tylko 2 faktów w każdym akapicie, bez zdań
wyjaśniających związek z tezą (same fakty, potem ogólnik typu "to pokazuje, że teza jest słuszna"). Łącznie 300-320 słów.
Stanowisko wyraź, ale nie uzasadniaj we wstępie. Zakończenie jedno-dwa zdania. Bez błędów faktograficznych i językowych."""

user = f"""Temat wypracowania (dokładnie taki jak w arkuszu, NIE zmieniaj go):
"{spec['temat']}"

Aspekty (w tej kolejności): {aspects[0]}, {aspects[1]}, {aspects[2]}

Lista faktów:
{fl}

Stanowisko, które MUSISZ przyjąć: {G.STANCES['zgadzam']}
{weak_rules}
Zwróć JSON o polach:
{{"teza": str (teza wyjęta z tematu, bez zmian), "temat": str (temat dokładnie jak wyżej), "stanowisko": "zgadzam",
 "wstep": str, "akapity": [{{"aspekt": str, "fakty_uzyte": [ids], "tekst": str}} x3], "zakonczenie": str}}"""

t = time.time()
resp = G.client.chat.completions.create(
    model=G.MODEL, reasoning_effort=G.REASON, max_completion_tokens=3000,
    response_format={"type": "json_object"},
    messages=[{"role": "system", "content": G.SYSTEM}, {"role": "user", "content": user}])
rec = json.loads(resp.choices[0].message.content)
rec["temat"] = spec["temat"]
chk = G.check(rec, mat)
record = dict(fixed_topic=True, label=spec["label"] + ("_weak" if WEAK else ""), prompt_version=variant, weak=WEAK,
              dzial=spec["label"], aspekty=aspects, stance="zgadzam", material=mat, essay=rec, check=chk,
              usage=dict(prompt=resp.usage.prompt_tokens, completion=resp.usage.completion_tokens), model=G.MODEL, reasoning=G.REASON)
outp = os.path.join(G.ROOT, "sol", f"fixed_{record['label']}.jsonl")
open(outp, "a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"{time.time()-t:.0f}s | tokens in {resp.usage.prompt_tokens} out {resp.usage.completion_tokens} | CHECK {chk}\n-> {outp}\n")
print(rec["wstep"], "\n")
for a in rec["akapity"]: print(f"[{a['aspekt']} | {a['fakty_uzyte']}]\n{a['tekst']}\n")
print(rec["zakonczenie"])
