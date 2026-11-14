"""Measure accuracy and confidence calibration against the hand-labelled files in eval/gold.

    python eval/run_eval.py            # LLM mode (uses .cache/llm, so re-runs are free)
    python eval/run_eval.py --offline  # pattern-only baseline

Prints the results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from co_extract import FIELDS  # noqa: E402
from co_extract.normalize import id_key, tokens  # noqa: E402
from co_extract.pipeline import extract_file  # noqa: E402

BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0001)]


def _match_one(name: str, pred, gold) -> bool:
    if gold is None or pred is None:
        return gold is None and pred is None
    if name == "amount":
        return abs(float(pred) - float(gold)) < 0.01
    if name in ("contract_number", "change_order_number"):
        return id_key(pred) == id_key(gold)
    if name == "description":  # gold is a keyword string; need >=50% of its keywords
        g = tokens(gold)
        return bool(g) and len(g & tokens(pred)) / len(g) >= 0.5
    if name in ("project_name", "requested_by"):
        g, p = tokens(gold), tokens(pred)
        if not g or not p:
            return False
        inter = len(g & p)
        return inter / len(g) >= 0.6 and inter / len(p) >= 0.5
    return pred == gold


def is_correct(name: str, pred, gold) -> bool:
    alts = gold if isinstance(gold, list) else [gold]
    return any(_match_one(name, pred, g) for g in alts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--samples", type=int, default=int(os.environ.get("LLM_SAMPLES", "3")))
    ap.add_argument("--gold-dir", default=str(ROOT / "eval" / "gold"))
    ap.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--today", default=None, help="pin 'today' (YYYY-MM-DD) for date-range checks")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    client = None
    if not args.offline:
        from co_extract.providers import make_client
        client = make_client(args.provider, model=args.model,
                             cache_dir=os.environ.get("LLM_CACHE_DIR", ROOT / ".cache/llm"))
    today = dt.date.fromisoformat(args.today) if args.today else None

    rows, docs = [], []
    for gold_path in sorted(Path(args.gold_dir).glob("*.json")):
        gold = json.loads(gold_path.read_text())
        doc_path = ROOT / gold["file"]
        if not doc_path.exists():
            print(f"skip {gold_path.name}: {doc_path} missing", file=sys.stderr)
            continue
        out = extract_file(doc_path, client=client, n_samples=args.samples, today=today)
        doc_errors = []
        for name in FIELDS:
            pred, conf = out[name]["value"], out[name]["confidence"]
            ok = is_correct(name, pred, gold["expected"][name])
            rows.append({"doc": gold_path.stem, "field": name, "pred": pred,
                         "gold": gold["expected"][name], "conf": conf, "ok": ok})
            if not ok:
                doc_errors.append(name)
        docs.append({"doc": gold_path.stem, "errors": doc_errors, "needs_review": out["needs_review"],
                     "reasons": out["review_reasons"]})

    if not rows:
        print("no gold files evaluated", file=sys.stderr)
        return 1

    n = len(rows)
    acc = sum(r["ok"] for r in rows) / n
    brier = sum((r["conf"] - r["ok"]) ** 2 for r in rows) / n
    label = "offline patterns" if args.offline else f"{client.provider} · {client.model}"
    lines = [f"# Eval ({label})", "",
             f"Documents: {len(docs)} | field predictions: {n} | field accuracy: **{acc:.1%}** | "
             f"Brier score: **{brier:.3f}** (lower is better; 0.25 = coin flip)", "",
             "## Per-field accuracy", "", "| field | accuracy | mean confidence |", "|---|---|---|"]
    for name in FIELDS:
        fr = [r for r in rows if r["field"] == name]
        lines.append(f"| {name} | {sum(r['ok'] for r in fr)}/{len(fr)} | "
                     f"{sum(r['conf'] for r in fr) / len(fr):.2f} |")

    lines += ["", "## Calibration", "",
              "If confidence is meaningful, accuracy should rise with the confidence bucket and roughly "
              "match the mean confidence in each bucket.", "",
              "| confidence bucket | n | mean confidence | observed accuracy |", "|---|---|---|---|"]
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["conf"] < hi]
        if b:
            lines.append(f"| {lo:.2f}-{min(hi, 1):.2f} | {len(b)} | {sum(r['conf'] for r in b) / len(b):.2f} "
                         f"| {sum(r['ok'] for r in b) / len(b):.0%} |")

    caught = sum(1 for d in docs if d["errors"] and d["needs_review"])
    missed = sum(1 for d in docs if d["errors"] and not d["needs_review"])
    false_alarm = sum(1 for d in docs if not d["errors"] and d["needs_review"])
    clean = sum(1 for d in docs if not d["errors"] and not d["needs_review"])
    lines += ["", "## needs_review as an error detector (document level)", "",
              "| | has a wrong field | all fields correct |", "|---|---|---|",
              f"| needs_review = true | {caught} (caught) | {false_alarm} (extra review) |",
              f"| needs_review = false | **{missed} (missed)** | {clean} (auto-accepted) |", "",
              "The number that matters most is *missed*: a wrong value that was auto-accepted.", "",
              "## Errors", "", "| doc | field | predicted | gold | confidence | flagged |", "|---|---|---|---|---|---|"]
    flagged = {d["doc"]: d["needs_review"] for d in docs}
    for r in rows:
        if not r["ok"]:
            lines.append(f"| {r['doc']} | {r['field']} | {json.dumps(r['pred'])} | {json.dumps(r['gold'])} "
                         f"| {r['conf']:.2f} | {flagged[r['doc']]} |")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
