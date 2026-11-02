"""A decision step ahead of the model: what kind of question is this?

Shadow mode only, for now: route() labels the person's question and core.chat_events logs the label (and what the
router would change) after the answer is out. Nothing about the answer changes. See the "decision router" design issue.

Rules come first (RULES: no model call, so no wait); a question no rule matches goes to one tiny, constrained call to
the chat model that is already loaded. If that fails or is unsure, the route is "unknown" and today's behaviour stands.
Off unless SURVIVAL_LLM_ROUTER=shadow. The log is also opt-in, and stays on this computer: SURVIVAL_LLM_ROUTER_LOG
names the file (one JSON line per question).
"""
import json
import os
import re
import time
from dataclasses import asdict, dataclass

KINDS = ("lookup", "urgent", "personal", "open", "chitchat")
UNKNOWN = "unknown"
MODE = "SURVIVAL_LLM_ROUTER"
LOG = "SURVIVAL_LLM_ROUTER_LOG"
CLASSIFIER_TIMEOUT = 10  # seconds; past this the router steps aside

# What each kind would change once the router is allowed to act. Settings the person chose are ceilings: the router
# only ever lowers them (reasoning: the cap on its tokens; None leaves their setting alone).
EFFECTS = {
    "lookup": {"must_search": True, "reasoning": "short"},
    "urgent": {"must_search": True, "reasoning": "short"},
    "personal": {"must_search": False, "reasoning": None},
    "open": {"must_search": False, "reasoning": None},
    "chitchat": {"must_search": False, "reasoning": "off", "tools": False},
    UNKNOWN: {"must_search": False, "reasoning": None},
}

# (name, pattern, kind). Checked in order; urgent first, so a rule for anything else never hides an emergency. Grow
# this from the log (evals/router_eval.py --mine), one reviewed rule at a time.
RULES = [
    ("emergency-words", re.compile(
        r"\b(911|not breathing|stopped breathing|can'?t breathe|choking|unconscious|unresponsive|seizure|overdos\w*|"
        r"severe(?:ly)? bleeding|bleeding (?:heavily|won'?t stop)|heart attack|stroke|anaphyla\w*|poison(?:ed|ing)?|"
        r"carbon monoxide|snake ?bite|drowning|hypothermi\w*|heat ?stroke)\b", re.I), "urgent"),
    ("greeting", re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|good (?:morning|evening))\W*$", re.I),
     "chitchat"),
]

CLASSIFIER_PROMPT = """Classify the person's message to an offline emergency and survival assistant. Answer with JSON.
kind:
- urgent: someone is hurt or sick now, or there is an immediate danger (injury, poisoning, fire, gas, downed power line)
- lookup: a factual or how-to question the reference library can answer (food safety, water, first aid, repairs)
- personal: they tell the assistant about themselves, their family or their equipment, or ask it to remember or forget
- open: planning, judgment, estimates or comparisons that need real thinking
- chitchat: a greeting, thanks or small talk
confidence: "high" or "low". When torn between urgent and anything else, answer urgent."""

SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": list(KINDS)},
    "confidence": {"type": "string", "enum": ["high", "low"]}},
    "required": ["kind", "confidence"]}


@dataclass
class Route:
    kind: str = UNKNOWN
    source: str = "none"  # "rule" or "classifier"
    rule: str = ""
    confidence: str = ""
    seconds: float = 0.0
    would: dict = None

    def as_dict(self):
        return asdict(self)


def mode():
    return os.environ.get(MODE, "off").strip().lower()


def match_rule(question):
    """(rule name, kind) for the first rule the question matches, else None."""
    for name, pattern, kind in RULES:
        if pattern.search(question):
            return name, kind
    return None


def classify(post, model, question):
    """Ask the loaded model for one label. post(path, body) is core._post. Returns (kind, confidence); anything
    that goes wrong, or low confidence, comes back as (UNKNOWN, ...), except that unsure about urgent stays urgent."""
    body = {"model": model, "stream": False, "think": False, "keep_alive": "15m", "format": SCHEMA,
            "options": {"temperature": 0, "num_predict": 40, "num_ctx": 2048},
            "messages": [{"role": "system", "content": CLASSIFIER_PROMPT},
                         {"role": "user", "content": question[:1000]}]}
    try:
        reply = json.loads(post("/api/chat", body, timeout=CLASSIFIER_TIMEOUT)["message"]["content"])
        kind, confidence = reply.get("kind"), reply.get("confidence")
    except Exception:  # noqa: BLE001 - a router that can't decide steps aside
        return UNKNOWN, "error"
    if kind not in KINDS:
        return UNKNOWN, "error"
    if confidence != "high" and kind != "urgent":
        return UNKNOWN, "low"
    return kind, confidence


def route(post, model, question):
    """Label the question: rules first, then the classifier. Never raises."""
    start = time.monotonic()
    hit = match_rule(question)
    if hit:
        found = Route(kind=hit[1], source="rule", rule=hit[0], confidence="high")
    else:
        kind, confidence = classify(post, model, question)
        found = Route(kind=kind, source="classifier", confidence=confidence)
    found.seconds = round(time.monotonic() - start, 3)
    found.would = EFFECTS[found.kind]
    return found


def log(found, question, model, reasoning):
    """Append one line to the opt-in log, if SURVIVAL_LLM_ROUTER_LOG names a file. Local only."""
    path = os.environ.get(LOG)
    if not path:
        return
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "reasoning": reasoning,
             "question": question, **found.as_dict()}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # a log that can't be written never gets in the way
