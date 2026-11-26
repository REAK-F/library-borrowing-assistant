"""Evaluate ranking and complete conversations against a fixed case file."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import sys

from assistant import BorrowingGuide, ROOT


CANDIDATE_THRESHOLDS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30)


def load_cases(part):
    with (ROOT / "data" / "cases.csv").open(newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["split"] == part]


def has_required_output(guide, expected_rule, expected_action, response):
    text = guide.render(response)
    if expected_action == "answer":
        rule = guide.rules[expected_rule]
        urls = [guide.data["sources"][key]["url"] for key, _ in rule["sources"]]
        return "官方出处" in text and "1. " in text and all(url in text for url in urls)
    if expected_action == "refer":
        return guide.data["sources"]["contact"]["url"] in text
    return "1 到 200" in text


def run_cases(guide, cases, method="tfidf"):
    results = []
    for case in cases:
        query = case["query"]
        expected_choice = int(case["choice"]) if case["choice"] else None
        keyword = guide.rank(query, "keyword")[0]
        tfidf = guide.rank(query)[0]
        response = guide.start(query, method)
        final = response
        if response.action == "ask" and case["reply"]:
            final = guide.follow_up(response, case["reply"])

        correct_action = response.action == case["action"]
        correct_rule = response.rule_id == case["rule_id"]
        expected_final_action = "answer" if case["action"] == "ask" else case["action"]
        success = (
            correct_action and correct_rule and final.action == expected_final_action
            and final.choice == expected_choice
            and has_required_output(guide, case["rule_id"], expected_final_action, final)
        )
        results.append({
            **case,
            "keyword_top": keyword.rule_id if keyword.score > 0 else "",
            "keyword_score": round(keyword.score, 6),
            "tfidf_top": tfidf.rule_id if tfidf.score > 0 else "",
            "tfidf_score": round(tfidf.score, 6),
            "predicted_action": response.action,
            "predicted_rule": response.rule_id,
            "final_action": final.action,
            "final_choice": final.choice,
            "correct": int(success),
        })
    return results


def count(predicate, rows):
    return sum(predicate(row) for row in rows)


def summarize(rows):
    judged = [row for row in rows if row["rule_id"]]
    referral = [row for row in rows if row["action"] == "refer"]
    followup = [row for row in rows if row["action"] == "ask"]
    groups = {row["category"]: {"cases": 0, "correct": 0} for row in rows}
    for row in rows:
        group = groups[row["category"]]
        group["cases"] += 1
        group["correct"] += row["correct"]
    return {
        "cases": len(rows),
        "answerable": len(judged),
        "keyword_top1": count(lambda row: row["keyword_top"] == row["rule_id"], judged),
        "tfidf_top1": count(lambda row: row["tfidf_top"] == row["rule_id"], judged),
        "workflow_correct": count(lambda row: row["correct"], rows),
        "refer_cases": len(referral),
        "refer_correct": count(lambda row: row["correct"], referral),
        "followup_cases": len(followup),
        "followup_correct": count(lambda row: row["correct"], followup),
        "by_category": groups,
    }


def choose_threshold(guide, cases, method="tfidf"):
    default = guide.threshold
    candidates = []
    for threshold in CANDIDATE_THRESHOLDS:
        guide.threshold = threshold
        summary = summarize(run_cases(guide, cases, method))
        candidates.append({"threshold": threshold, **summary})
    best = max(candidates, key=lambda row: (row["workflow_correct"], row["refer_correct"], row["threshold"]))
    guide.threshold = default
    return best["threshold"], candidates


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-only", action="store_true", help="只跑开发集来选阈值")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    guide = BorrowingGuide()
    initial = BorrowingGuide(enhanced=False)
    development = load_cases("dev")
    first_test = load_cases("test")
    first_threshold, first_candidates = choose_threshold(initial, development)
    threshold, candidates = choose_threshold(guide, development + first_test)
    keyword_threshold, keyword_candidates = choose_threshold(guide, development + first_test, "keyword")
    if args.dev_only:
        print(json.dumps({"initial_threshold": first_threshold, "selected_threshold": threshold,
                          "refinement": candidates}, indent=2))
        return

    if guide.threshold != threshold:
        raise SystemExit(f"rules.json threshold {guide.threshold} != dev selection {threshold}")
    if initial.threshold != first_threshold:
        raise SystemExit("The stored initial threshold differs from the initial development selection.")
    audit = load_cases("audit")
    first_rows = run_cases(initial, first_test)
    initial_audit_rows = run_cases(initial, audit)
    rows = run_cases(guide, audit)
    guide.threshold = keyword_threshold
    keyword_rows = run_cases(guide, audit, "keyword")
    # Remove the score threshold only. Explicit scope checks and follow-ups still apply.
    guide.threshold = 0.0
    no_threshold_rows = run_cases(guide, audit)
    output = {
        "threshold": threshold,
        "python": sys.version.split()[0],
        "scikit_learn": version("scikit-learn"),
        "initial_threshold": first_threshold,
        "initial_development": first_candidates,
        "refinement": candidates,
        "keyword_threshold": keyword_threshold,
        "keyword_development": keyword_candidates,
        "first_test": summarize(first_rows),
        "initial_on_audit": summarize(initial_audit_rows),
        "audit": summarize(rows),
        "audit_keyword_workflow": summarize(keyword_rows),
        "audit_without_threshold": summarize(no_threshold_rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_predictions(args.out / "first_test.csv", first_rows)
    write_predictions(args.out / "initial_on_audit.csv", initial_audit_rows)
    write_predictions(args.out / "predictions.csv", rows)
    write_predictions(args.out / "keyword_workflow.csv", keyword_rows)
    write_predictions(args.out / "without_threshold.csv", no_threshold_rows)
    for key in ("first_test", "initial_on_audit", "audit", "audit_keyword_workflow", "audit_without_threshold"):
        print(key, json.dumps(output[key], indent=2))
    for row in rows:
        if not row["correct"]:
            print(row["id"], row["query"], row["action"], row["rule_id"], "=>",
                  row["predicted_action"], row["predicted_rule"], row["tfidf_score"])


if __name__ == "__main__":
    main()
