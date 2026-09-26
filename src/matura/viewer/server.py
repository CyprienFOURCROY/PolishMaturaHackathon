"""Interactive viewer: python -m matura.viewer.server [--port 8765]

Serves the viewer on localhost and runs single pipeline stages on request. A stage that already has a
cached result is NOT re-run unless the request says refresh=true; instead the server answers 409
"already_run" and the page shows a warning with a Re-run button.
"""
import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from .. import cache, config
from .. import export
from ..eval import judge
from ..schema import Prepared, load_exam
from ..stages import answer, final, ocr, retrieve, translate, vlm
from . import build

STAGES = ("ocr", "vlm", "translate", "answer", "answer_pl", "judge")
_lock = threading.Lock()  # one model run at a time
_items = {i.id: i for i in load_exam(config.EXAM_PATH)[1]}


class StageError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


def _joined_ocr(item) -> str:
    return "\n".join(t for t in (cache.get("ocr", n) for n in item.images) if t)


def already_run(item, stage: str) -> bool:
    if stage == "ocr":
        return bool(item.images) and all(cache.exists("ocr", n) for n in item.images)
    return cache.exists(stage, item.id)


def run_stage(item_id: str, stage: str, refresh: bool) -> None:
    item = _items.get(item_id)
    if item is None or stage not in STAGES:
        raise StageError("bad_request", f"Unknown item or stage: {item_id!r}/{stage!r}")
    if stage in ("ocr", "vlm") and not item.images:
        raise StageError("skipped", "This question has no image, so this stage does not apply.")
    if already_run(item, stage) and not refresh:
        raise StageError("already_run", f"'{stage}' was already run for question {item_id} (result is cached). "
                                        "Re-running will overwrite it.", 409)
    # prerequisites: upstream results must exist (stages never silently run their parents)
    if stage in ("vlm", "translate") and item.images and not already_run(item, "ocr"):
        raise StageError("prereq", "Run OCR first: this stage uses the OCR text.")
    if stage == "answer" and not cache.exists("translate", item.id):
        raise StageError("prereq", "Run the translation first: the answer stage reads the English text.")

    if stage == "answer_pl" and not cache.exists("answer", item.id):
        raise StageError("prereq", "Run the answer stage first: this translates its English answer into Polish.")

    if stage == "judge":
        if item.id == "26":
            raise StageError("skipped", "The essay is graded separately (not implemented yet).")
        if judge.final_answer(item.id) is None:
            raise StageError("prereq", "No final answer to grade yet: run the answer and final translation stages first.")

    with _lock:
        try:
            if stage == "ocr":
                ocr.run(item, refresh=True)
            elif stage == "vlm":
                vlm.run(item, _joined_ocr(item), refresh=True)
            elif stage == "judge":
                judge.grade(item, judge.final_answer(item.id))
            elif stage == "answer_pl":
                cache.put("answer_pl", item.id, final.run(item, cache.get("answer", item.id)))
            else:
                p = Prepared(item=item, ocr=_joined_ocr(item), vlm=cache.get("vlm", item.id) or "")
                translate.run(p, refresh=(stage == "translate"))
                if stage == "answer":
                    retrieve.run(p)
                    cache.put("answer", item.id, answer.run(p))
        except StageError:
            raise
        except Exception as e:  # missing weights, OOM, ...
            raise StageError("failed", f"{type(e).__name__}: {e}", 500) from e


def clear_stage(item_id: str, stage: str) -> list[str]:
    """Delete cached results so the stage can be re-run. stage='all' clears every stage of the item.
    Downstream results are kept (they may now be stale); clear them too if you want a clean rerun."""
    item = _items.get(item_id)
    if item is None or (stage not in STAGES and stage != "all"):
        raise StageError("bad_request", f"Unknown item or stage: {item_id!r}/{stage!r}")
    cleared = []
    for st in STAGES if stage == "all" else (stage,):
        keys = item.images if st == "ocr" else [item.id]
        if any([cache.delete(st, k) for k in keys]):
            cleared.append(st)
    return cleared


# ---- batch runner: runs the whole pipeline for several questions on a background thread ----
_job_lock = threading.Lock()
_job = {"running": False, "stop": False, "queue": [], "index": 0, "id": None, "stage": None, "ok": [], "failed": [],
        "skipped": [], "grade": False, "refresh": False, "from_stage": None, "ran": 0, "cached": 0, "tick": 0, "note": ""}


def stages_for(item, grade: bool) -> list[str]:
    return (["ocr", "vlm"] if item.images else []) + ["translate", "answer", "answer_pl"] + (["judge"] if grade else [])


def status() -> dict:
    with _job_lock:
        return {**_job, "total": len(_job["queue"])}


def start_batch(ids: list[str], refresh: bool, grade: bool, from_stage: str | None = None) -> None:
    unknown = [i for i in ids if i not in _items]
    if from_stage is not None and from_stage not in STAGES:
        raise StageError("bad_request", f"Unknown stage: {from_stage!r}")
    if not ids or unknown:
        raise StageError("bad_request", f"No questions selected or unknown ids: {unknown}")
    with _job_lock:
        if _job["running"]:
            raise StageError("busy", "A batch is already running.", 409)
        _job.update(running=True, stop=False, queue=list(ids), index=0, id=None, stage=None, ok=[], failed=[],
                    skipped=[], grade=grade, refresh=refresh, from_stage=from_stage, ran=0, cached=0,
                    tick=_job["tick"] + 1, note="")
    threading.Thread(target=_run_batch, daemon=True).start()


def stop_batch() -> None:
    with _job_lock:
        if _job["running"]:
            _job["stop"] = True
            _job["note"] = "Stopping after the current stage…"


def _set(**kw) -> None:
    with _job_lock:
        _job.update(kw)
        _job["tick"] += 1


def _run_batch() -> None:
    last_err, repeats = None, 0
    try:
        for n, iid in enumerate(_job["queue"]):
            if _job["stop"]:
                break
            item = _items[iid]
            _set(index=n, id=iid, stage=None)
            if iid == "26":  # essay: not implemented yet
                with _job_lock:
                    _job["skipped"].append({"id": iid, "reason": "essay (not implemented yet)"})
                continue
            failure = None
            for st in stages_for(item, _job["grade"]):
                if _job["stop"]:
                    break
                _set(stage=st)
                # refresh everything, or only the stages from `from_stage` on (e.g. redo the answer, keep OCR/VLM)
                fs = _job["from_stage"]
                redo = _job["refresh"] and (fs is None or STAGES.index(st) >= STAGES.index(fs))
                try:
                    run_stage(iid, st, refresh=redo)
                    with _job_lock:
                        _job["ran"] += 1
                except StageError as e:
                    if e.code == "already_run":  # done before: skipped silently, that is the point of a smooth batch
                        with _job_lock:
                            _job["cached"] += 1
                        continue
                    failure = {"id": iid, "stage": st, "error": str(e)}
                    break
                _set()
            with _job_lock:
                if failure:
                    _job["failed"].append(failure)
                elif not _job["stop"]:
                    _job["ok"].append(iid)
            if failure:
                key = (failure["stage"], failure["error"])
                repeats = repeats + 1 if key == last_err else 1
                last_err = key
                if repeats >= 3:  # same error three times in a row (e.g. missing weights): stop instead of spamming
                    _set(stop=True, note=f"Stopped: the same error happened {repeats} times in a row.")
                    break
            else:
                last_err, repeats = None, 0
    finally:
        _set(running=False, id=None, stage=None,
             note="Stopped by you." if _job["note"].startswith("Stopping") else _job["note"])


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _data(self) -> dict:
        return build.collect(lambda n: f"/images/{n}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, build.render(self._data(), live=True).encode(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json(200, status())
        elif self.path == "/api/data":
            self._json(200, self._data())
        elif self.path.startswith("/images/"):
            f = config.IMAGES_DIR / unquote(self.path[len("/images/"):]).split("/")[-1]  # basename only
            if f.is_file():
                self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/stop":
            stop_batch()
            return self._json(200, {"ok": True})
        if self.path == "/api/batch":
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                start_batch([str(i) for i in req.get("ids", [])], bool(req.get("refresh")), bool(req.get("grade")),
                            req.get("from_stage"))
                return self._json(200, {"ok": True})
            except StageError as e:
                return self._json(e.status, {"ok": False, "code": e.code, "error": str(e)})
        if self.path == "/api/export":
            return self._json(200, {"ok": True, **export.write()})
        if self.path not in ("/api/run", "/api/delete"):
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/delete":
                return self._json(200, {"ok": True, "cleared": clear_stage(str(req.get("id")), str(req.get("stage")))})
            run_stage(str(req.get("id")), str(req.get("stage")), bool(req.get("refresh")))
            self._json(200, {"ok": True})
        except StageError as e:
            self._json(e.status, {"ok": False, "code": e.code, "error": str(e)})
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "code": "bad_request", "error": "Invalid JSON"})

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)  # localhost only
    print(f"Viewer on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
