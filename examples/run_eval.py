"""DuoRAG evaluation test suite.

Runs end-to-end evaluation tests against real documents using an LLM judge
for subjective quality assessment and deterministic checks for objective facts.

Usage:
    uv run --group eval python examples/run_eval.py [--reset] [--verbose] [--save-report eval_report.json]
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field

import yaml
from dotenv import load_dotenv

load_dotenv()

import openai

from duo_rag import DuoRAG, MetadataField


@dataclass
class TestResult:
    id: str
    stage: str
    question: str
    answer: str
    sql_used: str | None
    judge_score: float
    judge_reasoning: str
    deterministic_checks: dict[str, bool] = field(default_factory=dict)
    passed: bool = False


def judge_response(
    client: openai.OpenAI,
    judge_model: str,
    question: str,
    answer: str,
    sql_used: str | None,
    test_type: str,
    judge_criteria: str,
) -> tuple[float, str]:
    """Use an LLM judge to score the response quality."""
    dimensions = "correctness, completeness, relevance"
    extra_instruction = ""
    if test_type == "schema_evolution":
        dimensions += ", appropriate_response"
        extra_instruction = (
            "For appropriate_response, check whether the system correctly "
            "acknowledged it cannot answer precisely and indicated a schema gap. "
        )

    system_prompt = (
        f"You are an evaluation judge. Score the following answer on these dimensions: {dimensions}.\n"
        f"Each dimension is scored 0.0 to 1.0. Return the AVERAGE as 'overall_score'.\n"
        f"{extra_instruction}\n"
        f"Evaluation criteria: {judge_criteria}\n\n"
        f"Return JSON with keys: overall_score (float), reasoning (string)."
    )

    user_content = f"Question: {question}\nAnswer: {answer}"
    if sql_used:
        user_content += f"\nSQL used: {sql_used}"
    user_content += f"\nTest type: {test_type}"

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)
    return float(raw.get("overall_score", 0.0)), raw.get("reasoning", "")


def run_deterministic_checks(test_case: dict, answer: str, sql_used: str | None, rag: DuoRAG) -> dict[str, bool]:
    """Run non-LLM assertions on the response."""
    checks: dict[str, bool] = {}
    answer_lower = answer.lower()

    if test_case.get("expect_sql"):
        checks["sql_was_used"] = sql_used is not None and len(sql_used) > 0

    for kw in test_case.get("expected_keywords", []):
        checks[f"keyword_{kw}"] = kw.lower() in answer_lower

    for name in test_case.get("expected_names", []):
        checks[f"expected_{name}"] = name.lower() in answer_lower

    for name in test_case.get("excluded_names", []):
        checks[f"excluded_{name}"] = name.lower() not in answer_lower

    if test_case.get("expect_gap_detected"):
        # After an evolve query, check if the schema now has a field matching the expected pattern
        expected_like = test_case.get("expected_new_field_like", "")
        if expected_like and rag.schema:
            field_names = [f.name for f in rag.schema.fields]
            checks["gap_detected_new_field"] = any(expected_like in fn for fn in field_names)
        else:
            checks["gap_detected_new_field"] = False

    return checks


class EvalRunner:
    def __init__(self, config: dict, stages: list[dict], verbose: bool = False):
        self.config = config
        self.stages = stages
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.conversation_histories: dict[str, list[dict]] = {}
        self.client = openai.OpenAI()

        schema = [
            MetadataField(name=s["name"], type=s["type"], description=s["description"])
            for s in config["schema"]
        ]
        self.rag = DuoRAG(
            llm_model=config["llm_model"],
            schema=schema,
            data_dir=config["data_dir"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

    def setup(self):
        """Ingest documents if not already done."""
        db_path = os.path.join(self.config["data_dir"], "metadata.db")
        if not os.path.exists(db_path):
            print("Ingesting documents...")

            def progress(current: int, total: int) -> None:
                print(f"  Ingesting file {current}/{total}...", end="\r")

            stats = self.rag.ingest(self.config["documents_path"], on_progress=progress)
            print(f"\nDone. new={stats['new']}  changed={stats['changed']}  unchanged={stats['unchanged']}")
        else:
            print(f"Using existing data in {self.config['data_dir']}/")
        print(f"Schema fields: {[f.name for f in self.rag.schema.fields]}")

    def run(self):
        """Run all stages in order."""
        for stage in self.stages:
            stage_name = stage["name"]
            print(f"\n{'='*60}")
            print(f"  {stage_name}")
            print(f"{'='*60}")

            # Handle stage-level actions
            if stage.get("action") == "backfill":
                print("Running backfill...")

                def backfill_progress(current: int, total: int) -> None:
                    print(f"  Backfilling chunk {current}/{total}...", end="\r")

                result = self.rag.backfill(on_progress=backfill_progress)
                print(f"\nBackfill complete. Populated: {result['populated']}  Pruned: {result['pruned']}")

            # Run tests in this stage
            for test_case in stage.get("tests", []):
                self._run_test(test_case, stage_name)

    def _run_test(self, test_case: dict, stage_name: str):
        """Execute a single test case."""
        test_id = test_case["id"]
        question = test_case["question"]
        test_type = test_case.get("type", "factual")
        evolve = test_case.get("evolve", False)

        # Handle conversation continuity
        history = None
        if test_case.get("continues_from"):
            history = self.conversation_histories.get(test_case["continues_from"])

        print(f"\n  [{test_id}] Q: {question}")

        # Execute query
        answer = self.rag.query(question, evolve=evolve, history=history)
        sql_used = self.rag.last_sql

        # Save conversation history if requested
        if test_case.get("save_history"):
            self.conversation_histories[test_id] = self.rag.last_history

        if self.verbose:
            print(f"  A: {answer}")
            if sql_used:
                print(f"  SQL: {sql_used}")

        # Judge evaluation
        judge_score, judge_reasoning = judge_response(
            client=self.client,
            judge_model=self.config["judge_model"],
            question=question,
            answer=answer,
            sql_used=sql_used,
            test_type=test_type,
            judge_criteria=test_case.get("judge_criteria", ""),
        )

        # Deterministic checks
        det_checks = run_deterministic_checks(test_case, answer, sql_used, self.rag)

        # Pass/fail: judge >= 0.7 AND all deterministic checks pass
        all_det_pass = all(det_checks.values()) if det_checks else True
        passed = judge_score >= 0.7 and all_det_pass

        result = TestResult(
            id=test_id,
            stage=stage_name,
            question=question,
            answer=answer,
            sql_used=sql_used,
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
            deterministic_checks=det_checks,
            passed=passed,
        )
        self.results.append(result)

        # Console output
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] judge={judge_score:.2f}", end="")
        if det_checks:
            failed_checks = [k for k, v in det_checks.items() if not v]
            if failed_checks:
                print(f"  failed_checks={failed_checks}", end="")
        print()

        if self.verbose and judge_reasoning:
            print(f"  Judge: {judge_reasoning}")

    def print_summary(self):
        """Print overall results summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        print(f"\n{'='*60}")
        print(f"  RESULTS: {passed}/{total} passed ({passed/total*100:.0f}%)")
        print(f"{'='*60}")

        avg_score = sum(r.judge_score for r in self.results) / total if total else 0
        print(f"  Average judge score: {avg_score:.2f}")

        failed = [r for r in self.results if not r.passed]
        if failed:
            print(f"\n  Failed tests:")
            for r in failed:
                print(f"    - {r.id} (judge={r.judge_score:.2f})")

    def save_report(self, path: str):
        """Save detailed JSON report."""
        report = {
            "config": self.config,
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "average_judge_score": (
                sum(r.judge_score for r in self.results) / len(self.results)
                if self.results
                else 0
            ),
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="DuoRAG evaluation suite")
    parser.add_argument("--reset", action="store_true", help="Delete eval data for a clean run")
    parser.add_argument("--verbose", action="store_true", help="Print answers and judge reasoning")
    parser.add_argument("--save-report", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    # Load test definitions
    yaml_path = os.path.join(os.path.dirname(__file__), "eval_tests.yaml")
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    config = spec["config"]
    stages = spec["stages"]

    # Reset if requested
    if args.reset and os.path.exists(config["data_dir"]):
        shutil.rmtree(config["data_dir"])
        print(f"Reset: deleted {config['data_dir']}/")

    runner = EvalRunner(config, stages, verbose=args.verbose)
    runner.setup()
    runner.run()
    runner.print_summary()

    if args.save_report:
        runner.save_report(args.save_report)

    # Exit with non-zero if any test failed
    if any(not r.passed for r in runner.results):
        sys.exit(1)


if __name__ == "__main__":
    main()
