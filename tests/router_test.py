"""The decision router in shadow mode: rules, the classifier's failure handling, and that core.chat_events logs a
route without changing the answer. No Ollama: a fake model (core._stream) and a fake classifier (core._post).

Usage: .venv/bin/python tests/router_test.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402
import router  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  -> {detail}"))
    if not ok:
        FAILS.append(name)


def says(kind, confidence="high"):
    return lambda path, body, **kw: {"message": {"content": json.dumps({"kind": kind, "confidence": confidence})}}


def boom(path, body, **kw):
    raise OSError("no server")


def never(path, body, **kw):
    raise AssertionError("classifier called")


# Rules: no model call, and an emergency word wins.
r = router.route(never, "m", "He is choking, what do I do?")
check("emergency rule", (r.kind, r.source, r.rule) == ("urgent", "rule", "emergency-words"), r)
check("greeting rule", router.route(never, "m", "hello!").kind == "chitchat")
check("urgent beats greeting", router.route(never, "m", "hi, 911 isn't answering").kind == "urgent")

# Classifier.
check("classifier label", router.route(says("lookup"), "m", "How long is thawed chicken safe?").kind == "lookup")
check("low confidence steps aside", router.route(says("open", "low"), "m", "x").kind == router.UNKNOWN)
check("low-confidence urgent stays urgent", router.route(says("urgent", "low"), "m", "x").kind == "urgent")
check("error steps aside", router.route(boom, "m", "x").kind == router.UNKNOWN)
check("bad label steps aside", router.route(says("banana"), "m", "x").kind == router.UNKNOWN)
check("unknown changes nothing", router.route(boom, "m", "x").would == {"must_search": False, "reasoning": None})
check("urgent never turns reasoning off", router.EFFECTS["urgent"]["reasoning"] != "off")


# chat_events: same events with the router off and in shadow mode; the shadow run writes one log line.
class Mem:
    ai_writes = False


def run(mode, path):
    for var, val in ((router.MODE, mode), (router.LOG, path)):
        os.environ.pop(var, None)
        if val:
            os.environ[var] = val
    core._stream = lambda *a, **k: iter([{"message": {"content": "Boil it."}}, {"done": True, "eval_count": 3}])
    core._post = says("lookup")
    core.capabilities = lambda model: set()
    events = list(core.chat_events("m", None, [{"role": "user", "content": "How do I purify water?"}], caps=set(),
                                   memory=Mem()))
    return [e["type"] for e in events], [e for e in events if e["type"] == "done"][0]["answer"]


with tempfile.TemporaryDirectory() as tmp:
    log = os.path.join(tmp, "router.jsonl")
    off = run("off", log)
    check("off writes no log", not os.path.exists(log))
    shadow = run("shadow", log)
    check("shadow leaves the answer alone", shadow == off, (shadow, off))
    lines = open(log).read().splitlines() if os.path.exists(log) else []
    entry = json.loads(lines[0]) if lines else {}
    check("shadow logs one route", len(lines) == 1 and entry.get("kind") == "lookup"
          and entry.get("question") == "How do I purify water?", lines)
    os.remove(log)
    run("shadow", "")
    check("no log path, no log", not os.path.exists(log))

sys.exit(1 if FAILS else 0)
