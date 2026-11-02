"""Score the decision router (router.py) on labelled questions, or mine its log for candidate rules.

Usage: python3 evals/router_eval.py MODEL            routing accuracy, urgent recall, rule coverage and latency
       python3 evals/router_eval.py --mine LOG       questions the classifier labelled, most common first, to review
                                                      and (by hand) turn into rules in router.RULES
Talks to the Ollama server on OLLAMA_HOST (default 127.0.0.1:11434); stdlib only. Add questions to LABELLED as the
router meets new kinds.
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402
import router  # noqa: E402

LABELLED = [
    ("My toddler swallowed a button battery, what do I do?", "urgent"),
    ("A power line is down in my front yard. What should I do?", "urgent"),
    ("I smell gas in the kitchen", "urgent"),
    ("My dad is having chest pain and his left arm is numb", "urgent"),
    ("There is a fire in the garage", "urgent"),
    ("He is choking", "urgent"),
    ("How long will the food in my freezer stay safe?", "lookup"),
    ("How do I make tap water safe with bleach?", "lookup"),
    ("How do I treat a minor burn?", "lookup"),
    ("What is the boiling time for water at altitude?", "lookup"),
    ("How do I reset a tripped GFCI outlet?", "lookup"),
    ("My daughter is allergic to penicillin", "personal"),
    ("Remember that I have a Honda EU2200i generator", "personal"),
    ("Forget where I live", "personal"),
    ("Make me a two-week plan for a family of four if the grid is down after a hurricane", "open"),
    ("Should we shelter in place or evacuate, given a wildfire 20 miles away?", "open"),
    ("What are my odds of dying driving to the store versus staying home?", "open"),
    ("hello", "chitchat"),
    ("thanks!", "chitchat"),
]


def score(model):
    rows, total = [], 0.0
    for question, want in LABELLED:
        found = router.route(core._post, model, question)
        total += found.seconds
        rows.append((question, want, found))
        print(f"{'ok  ' if found.kind == want else 'MISS'} {want:9} -> {found.kind:9} {found.source:10} "
              f"{found.seconds:5.2f}s  {question[:60]}")
    n = len(rows)
    right = sum(f.kind == w for _, w, f in rows)
    urgent = [(w, f) for _, w, f in rows if w == "urgent"]
    print(f"\naccuracy {right}/{n}; urgent recall {sum(f.kind == 'urgent' for _, f in urgent)}/{len(urgent)}; "
          f"rule hits {sum(f.source == 'rule' for _, _, f in rows)}/{n}; "
          f"unknown {sum(f.kind == router.UNKNOWN for _, _, f in rows)}/{n}; mean {total / n:.2f}s per question")
    wrong_urgent = [q for q, w, f in rows if w == "urgent" and f.kind != "urgent"]
    if wrong_urgent:
        print("URGENT MISSED:", *wrong_urgent, sep="\n  ")
        sys.exit(1)


def mine(path):
    seen = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("source") == "classifier":
                seen[(entry["kind"], entry["question"].strip().lower())] += 1
    for (kind, question), count in seen.most_common(50):
        print(f"{count:3}  {kind:9}  {question}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--mine":
        mine(sys.argv[2])
    elif len(sys.argv) == 2:
        score(sys.argv[1])
    else:
        sys.exit(__doc__)
