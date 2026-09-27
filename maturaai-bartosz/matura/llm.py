"""Klient OpenAI-compatible (llama-server) z obsługą obrazów."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests


def obraz_url(p: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def czat(url: str, system: str, user: str, obrazy: list[str] | None = None, *, max_tokens: int = 512,
         temperature: float = 0.2, bez_myslenia: bool = True, timeout: int = 600,
         probkowanie: dict | None = None) -> tuple[str, float]:
    """Zwraca (tekst odpowiedzi, sekundy). Zmienna MATURA_TEMPERATURA (np. 0) nadpisuje temperaturę wszystkich wywołań
    (zadania krótkie, akapity eseju): pomiar A/B temperatury bez zmiany domyślnych wartości.
    probkowanie: dodatkowe pola llama-server (np. {"dry_multiplier": 0.8}) dla esejów uczniów SFT; domyślnie brak."""
    if os.environ.get("MATURA_TEMPERATURA"):
        temperature = float(os.environ["MATURA_TEMPERATURA"])
    tresc: list | str = user
    if obrazy:
        tresc = [{"type": "text", "text": user}] + [{"type": "image_url", "image_url": {"url": obraz_url(o)}} for o in obrazy]
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": tresc}],
            "max_tokens": max_tokens, "temperature": temperature}
    if bez_myslenia:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if probkowanie:
        body.update(probkowanie)
    t0 = time.time()
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or "").strip(), time.time() - t0
