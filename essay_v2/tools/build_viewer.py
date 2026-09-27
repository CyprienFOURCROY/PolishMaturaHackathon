"""Build a single-file HTML viewer for all essay records in out/sol/*.jsonl.
Usage: python3 out/tools/build_viewer.py   -> writes out/viewer.html (open in Chrome)
Records may be Sol-generated (essay + material) or later model outputs; anything with
{"essay": {...}, "material": [...]} renders. Years in the essay are green if they appear
in the supplied facts, red otherwise."""
import json, glob, html, os, re, sys

ROOT = os.environ.get("ESSAY_OUT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out")  # gitignored data dir
recs = []
paths = sorted(glob.glob(os.path.join(ROOT, "sol", "*.jsonl")))
# prefer the *_judged.jsonl twin when it exists
paths = [p for p in paths if p.endswith("_judged.jsonl") or p.replace(".jsonl", "_judged.jsonl") not in paths]
for path in paths:
    for i, line in enumerate(open(path, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["_file"] = os.path.basename(path)
        r["_line"] = i + 1
        recs.append(r)

data = json.dumps(recs, ensure_ascii=False)

PAGE = r"""<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Essay data viewer</title>
<style>
body{margin:0;font:15px/1.5 -apple-system,Helvetica,Arial,sans-serif;display:flex;height:100vh;color:#222}
#side{width:340px;min-width:340px;overflow:auto;border-right:1px solid #ddd;background:#fafafa}
#side .item{padding:8px 12px;border-bottom:1px solid #eee;cursor:pointer;font-size:13px}
#side .item:hover{background:#eef} #side .item.sel{background:#dde}
#side .item .t{font-weight:600} #side .item .m{color:#666}
#main{flex:1;overflow:auto;padding:20px 32px;max-width:900px}
h2{font-size:18px;margin:0 0 6px} .temat{background:#f3f3ff;padding:10px 14px;border-left:4px solid #88a;margin-bottom:16px}
.par{margin:14px 0;padding:10px 14px;border-left:3px solid #ccc}
.par .lab{font-size:12px;color:#666;margin-bottom:4px} .par p{margin:0 0 8px}
.facts{font-size:13px;color:#444;margin-top:6px;padding-left:16px} .facts li.unused{color:#b55}
.y-ok{background:#d6f5d6;border-radius:3px;padding:0 2px} .y-bad{background:#f8c8c8;border-radius:3px;padding:0 2px}
.chk{font-family:Menlo,monospace;font-size:12px;background:#f5f5f5;padding:8px 12px;margin:10px 0}
.bad{color:#b00;font-weight:600} .ok{color:#080}
#filter{width:100%;box-sizing:border-box;padding:8px;border:0;border-bottom:1px solid #ddd;font-size:13px}
.meta{font-size:12px;color:#777;margin-bottom:10px}
</style></head><body>
<div id="side"><input id="filter" placeholder="filter (dział, stance, file, flag)…"><div id="list"></div></div>
<div id="main"><p>Wybierz wypracowanie z listy.</p></div>
<script>
const R = __DATA__;
const esc = s => String(s??'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function factYears(mat){ const s=new Set(); for(const m of mat){ for(const y of ((m.data||'')+' '+(m.fakt||'')).match(/\b\d{3,4}\b/g)||[]) s.add(y);} return s; }
function mark(text, ys){ return esc(text).replace(/\b(\d{3,4})\b/g, (m,y)=> `<span class="${ys.has(y)?'y-ok':'y-bad'}">${y}</span>`); }
function flags(r){ const c=r.check||{}; const f=[]; if(c.extra_years&&c.extra_years.length) f.push('extra_years'); if(c.bad_ids&&c.bad_ids.length) f.push('bad_ids'); if(c.markdown) f.push('markdown'); if(c.words&&(c.words<300)) f.push('short'); if(c.words&&(c.words>420)) f.push('long'); return f; }
function render(i){
  const r=R[i], e=r.essay||{}, mat=r.material||[], ys=factYears(mat);
  document.querySelectorAll('.item').forEach((el,j)=>el.classList.toggle('sel', j===i));
  const used=new Set((e.akapity||[]).flatMap(a=>a.fakty_uzyte||[]));
  let h=`<div class="meta">${esc(r._file)}:${r._line} · ${esc(r.model||'')} · reasoning=${esc(r.reasoning||'')} · prompt v${esc(r.prompt_version||'1')} · stance=${esc(r.stance||'')}</div>`;
  h+=`<h2>${esc(r.dzial||'')}</h2><div class="temat"><b>Temat:</b> ${esc(e.temat||'')}</div>`;
  const c=r.check||{}; const fl=flags(r);
  h+=`<div class="chk">words=${c.words??'?'} · paragraphs=${c.n_par??'?'} · aspects_ok=${c.aspects_ok} · extra_years=${JSON.stringify(c.extra_years||[])} · bad_ids=${JSON.stringify(c.bad_ids||[])} · markdown=${c.markdown} → <span class="${fl.length?'bad':'ok'}">${fl.length?fl.join(', '):'clean'}</span></div>`;
  const J=r.judge; if(J && J.suma!=null){
    h+=`<div class="chk" style="background:#eef7ee"><b>SĘDZIA (${esc(J._judge_model||'')}, blind): ${J.suma}/15</b> · A=${J.A} (przed odjęciem ${J.A_przed_odjeciem}, −${J.odjecie}) · B=${J.B}<br>`;
    for(const a of (J.aspekty||[])) h+=`• <b>${esc(a.nazwa)}</b>: ${esc(a.poziom)} / ${esc(a.funkcjonalnosc)} — ${esc(a.uzasadnienie)}<br>`;
    if((J.bledy_merytoryczne||[]).length) h+=`<span class="bad">Błędy: ${esc(J.bledy_merytoryczne.join(' | '))}</span><br>`;
    h+=`B: ${esc(J.B_uzasadnienie||'')}<br><i>Największa słabość:</i> ${esc(J.najwazniejsza_slabosc||'')}</div>`; }
  h+=`<div class="par"><div class="lab">WSTĘP</div><p>${mark(e.wstep||'',ys)}</p></div>`;
  for(const a of (e.akapity||[])){
    h+=`<div class="par"><div class="lab">AKAPIT · ${esc(a.aspekt)} · fakty: ${esc((a.fakty_uzyte||[]).join(', '))}</div><p>${mark(a.tekst||'',ys)}</p>`;
    const fm=mat.filter(m=>m.aspekt===a.aspekt);
    h+=`<ul class="facts">`+fm.map(m=>`<li class="${used.has(m.id)?'':'unused'}"><b>${esc(m.id)}</b> ${esc(m.fakt)} ${m.data?'('+esc(m.data)+')':''}${used.has(m.id)?'':' — <i>nieużyty</i>'}</li>`).join('')+`</ul></div>`;
  }
  h+=`<div class="par"><div class="lab">ZAKOŃCZENIE</div><p>${mark(e.zakonczenie||'',ys)}</p></div>`;
  const full=[e.wstep,...(e.akapity||[]).map(a=>a.tekst),e.zakonczenie].join('\n\n');
  h+=`<details><summary>Pełny tekst (do kopiowania)</summary><pre style="white-space:pre-wrap;font:14px/1.5 Georgia,serif">${esc(full)}</pre></details>`;
  h+=`<details><summary>Surowy rekord JSON</summary><pre style="white-space:pre-wrap;font-size:11px">${esc(JSON.stringify(r,null,1))}</pre></details>`;
  document.getElementById('main').innerHTML=h; document.getElementById('main').scrollTop=0;
}
function list(){
  const q=(document.getElementById('filter').value||'').toLowerCase();
  const L=document.getElementById('list'); L.innerHTML='';
  R.forEach((r,i)=>{ const fl=flags(r); const s=`${r.dzial} ${r.stance} ${r._file} v${r.prompt_version||1} ${fl.join(' ')} ${(r.judge||{}).suma??''}`.toLowerCase(); if(q&&!s.includes(q)) return;
    const d=document.createElement('div'); d.className='item'; d.onclick=()=>render(i);
    const J=r.judge||{}; const sc=J.suma!=null?`<b>${J.suma}/15</b> (A${J.A} B${J.B}) · `:'';
    d.innerHTML=`<div class="t">${esc(r.dzial||'?')}</div><div class="m">${sc}v${esc(r.prompt_version||'1')} · ${esc(r._file).replace('_judged','')}:${r._line} · ${(r.check||{}).words??'?'} w · <span class="${fl.length?'bad':'ok'}">${fl.length?fl.join(','):'clean'}</span></div>`;
    L.appendChild(d); });
}
document.getElementById('filter').oninput=list; list(); if(R.length) render(0);
</script></body></html>"""

out = os.path.join(ROOT, "viewer.html")
open(out, "w", encoding="utf-8").write(PAGE.replace("__DATA__", data))
print(f"{len(recs)} records -> {out}")
