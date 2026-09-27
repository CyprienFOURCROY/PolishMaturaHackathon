#!/usr/bin/env python3
"""Run the frozen baseline, validate the submission, and package an audit archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exam', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8080')
    p.add_argument('--essay-ids', nargs='+')
    p.add_argument('--no-essays', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Validate inputs, no generation')
    p.add_argument('--package-only', action='store_true', help='Validate/package an existing completed run')
    a = p.parse_args()
    if a.essay_ids and a.no_essays: p.error('Conflicting essay options')
    if a.check_only and a.package_only: p.error('Conflicting modes')
    exam_path = a.exam.resolve()
    out = a.out.resolve()
    if not exam_path.is_file(): p.error('Exam not found: '+str(exam_path))
    if out == HERE or out == exam_path.parent:
        p.error('Use a dedicated output directory')
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(HERE/'run_exam_gemma_final.py'),
           '--exam', str(exam_path), '--out', str(out),
           '--timeline', str(HERE/'timeline/events.jsonl'),
           '--context-notes', str(HERE/'historical_context.jsonl'),
           '--base-url', a.base_url, '--temperature', '0',
           '--max-tokens', '768', '--essay-tokens', '3072']
    if a.essay_ids: cmd += ['--essay-ids', *a.essay_ids]
    if a.no_essays: cmd += ['--no-essays']
    if not a.package_only:
        subprocess.run(cmd+['--validate-only'], check=True)
        with urllib.request.urlopen(a.base_url.rstrip('/')+'/health', timeout=10) as r:
            health=json.load(r)
        if health.get('status') != 'ok': raise RuntimeError('Server not ready: '+str(health))
        print('Server ready', flush=True)
        model_root = Path(os.environ.get('GEMMA_MODEL_ROOT', '/workspace/gemma-matura'))
        weights = [model_root/'model/gemma3-12b-q4ks/google_gemma-3-12b-it-Q4_K_S.gguf',
                   model_root/'model/gemma3-12b-iq3/mmproj-gemma-3-12b-it-f16.gguf']
        inventory = [{'path':str(f),'bytes':f.stat().st_size if f.is_file() else None} for f in weights]
        (out/'local-weight-inventory.json').write_text(json.dumps({
            'note':'Expected local files, not proof of the model loaded by the running server',
            'files':inventory},indent=2))
        print('Expected local weight bytes:',sum(x['bytes'] or 0 for x in inventory), flush=True)
        if a.check_only: return
        with (out/'client.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1)
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():
                raise SystemExit('Run interrupted. Completed answers remain saved. Repeat the same command to resume.')
    import run_exam_gemma_final as client
    exam=json.loads(exam_path.read_text())
    answers=json.loads((out/'answers.json').read_text())
    records=json.loads((out/'records.json').read_text())
    config=json.loads((out/'run-config.json').read_text())
    errors,warnings=[],[]
    if config.get('exam_sha256')!=hashlib.sha256(exam_path.read_bytes()).hexdigest():
        errors.append('Exam hash differs from the run')
    if answers.get('exam_id')!=exam.get('exam_id'): errors.append('exam_id mismatch')
    expected=[str(i['id']) for i in exam['items']]
    rows=answers.get('answers',[])
    actual=[str(i.get('id')) for i in rows]
    if actual!=expected or len(set(actual))!=len(actual): errors.append('Missing, extra, duplicate or reordered IDs')
    if set(records)!=set(expected): errors.append('Records do not cover the full exam')
    essays=set(config.get('essay_ids',[]))
    items={str(i['id']):i for i in exam['items']}
    for row in rows:
        key=str(row.get('id'))
        answer=row.get('answer')
        if not isinstance(answer,str) or not answer.strip():
            errors.append(key+': empty answer');continue
        record=records.get(key,{})
        if record.get('answer')!=answer: errors.append(key+': answer differs from records')
        if record.get('truncated') or record.get('finish_reason')=='length':
            errors.append(key+': truncated answer')
        if record.get('finish_reason')!='stop':
            errors.append(key+': unexpected finish_reason')
        if any(t in answer for t in ('<think>','</think>','<|channel>','<channel|>')):
            errors.append(key+': reasoning markers in answer')
        if key in essays and key in items:
            low,high=client.word_bounds(items[key]['question'])
            count=len(answer.split())
            if low is not None and count<low: errors.append(f'{key}: {count} words, minimum {low}')
            if high is not None and count>high: errors.append(f'{key}: {count} words, maximum {high}')
            if low is None and high is None: warnings.append(key+': no explicit word limit parsed; inspect task manually')
    report={'status':'PASS' if not errors else 'FAIL','errors':errors,'warnings':warnings,
            'answer_count':len(rows),'essay_ids':sorted(essays),
            'note':'Structural checks only; not a historical correctness grade'}
    (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    submission=out/'submission.json'
    if not errors: submission.write_bytes((out/'answers.json').read_bytes())
    elif submission.exists(): submission.unlink()
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t: t.add(out,arcname=out.name)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Archive:',archive)
    if errors: raise SystemExit('Validation failed. Inspect validation.json; submission.json was not created.')
    print('READY TO UPLOAD:',submission)

if __name__=='__main__':
    main()
