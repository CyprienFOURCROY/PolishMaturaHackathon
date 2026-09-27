"""Testy matura/llm.py: nadpisanie temperatury zmienną MATURA_TEMPERATURA (pomiar A/B)."""
from matura import llm


class _Odp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def _przechwyc(monkeypatch):
    wyslane = {}

    def post(url, json, timeout):
        wyslane.update(json)
        return _Odp()

    monkeypatch.setattr(llm.requests, "post", post)
    return wyslane


def test_domyslna_temperatura_bez_zmiennej(monkeypatch):
    monkeypatch.delenv("MATURA_TEMPERATURA", raising=False)
    w = _przechwyc(monkeypatch)
    llm.czat("http://x", "s", "u", temperature=0.7)
    assert w["temperature"] == 0.7


def test_zmienna_nadpisuje_temperature(monkeypatch):
    monkeypatch.setenv("MATURA_TEMPERATURA", "0")
    w = _przechwyc(monkeypatch)
    llm.czat("http://x", "s", "u", temperature=0.7)
    assert w["temperature"] == 0.0
