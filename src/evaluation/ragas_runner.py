import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    metrics_dict: Dict[str, float]
    per_question_scores: List[Dict[str, Any]]
    overall_score: float
    num_questions: int = 0

    def __post_init__(self):
        self.num_questions = len(self.per_question_scores)


class RAGASEvaluator:
    METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm_model = llm_model
        self._ragas_available = False
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from datasets import Dataset
            self._evaluate = evaluate
            self._metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            self._Dataset = Dataset
            self._ragas_available = True
            logger.info("RAGAS library loaded successfully")
        except ImportError:
            logger.warning(
                "ragas or datasets not installed; evaluation will use mock scores"
            )

    def evaluate_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> EvaluationReport:
        if len(questions) != len(ground_truths) != len(answers) != len(contexts):
            raise ValueError("All input lists must have the same length")

        if self._ragas_available:
            return self._evaluate_with_ragas(questions, ground_truths, answers, contexts)
        else:
            return self._evaluate_mock(questions, ground_truths, answers, contexts)

    def _evaluate_with_ragas(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> EvaluationReport:
        data = {
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": contexts,
        }
        dataset = self._Dataset.from_dict(data)
        result = self._evaluate(dataset, metrics=self._metrics)
        result_df = result.to_pandas()

        metrics_dict: Dict[str, float] = {}
        for metric in self.METRICS:
            if metric in result_df.columns:
                metrics_dict[metric] = float(result_df[metric].mean())

        per_question: List[Dict[str, Any]] = []
        for i, row in result_df.iterrows():
            q_scores = {"question": questions[i]}
            for metric in self.METRICS:
                if metric in row:
                    q_scores[metric] = float(row[metric])
            per_question.append(q_scores)

        overall = float(np.mean(list(metrics_dict.values()))) if metrics_dict else 0.0
        return EvaluationReport(
            metrics_dict=metrics_dict,
            per_question_scores=per_question,
            overall_score=round(overall, 4),
        )

    def _evaluate_mock(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> EvaluationReport:
        rng = np.random.default_rng(42)
        metrics_dict = {m: float(rng.uniform(0.6, 0.95)) for m in self.METRICS}
        per_question = []
        for q in questions:
            q_scores = {"question": q}
            q_scores.update({m: float(rng.uniform(0.5, 1.0)) for m in self.METRICS})
            per_question.append(q_scores)
        overall = float(np.mean(list(metrics_dict.values())))
        return EvaluationReport(
            metrics_dict=metrics_dict,
            per_question_scores=per_question,
            overall_score=round(overall, 4),
        )

    def load_test_dataset(self, filepath: str) -> Dict[str, List]:
        questions, ground_truths = [], []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                questions.append(item["question"])
                ground_truths.append(item["ground_truth"])
        logger.info("Loaded %d test cases from %s", len(questions), filepath)
        return {"questions": questions, "ground_truths": ground_truths}

    def save_report(self, report: EvaluationReport, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_data = {
            "overall_score": report.overall_score,
            "num_questions": report.num_questions,
            "metrics": report.metrics_dict,
            "per_question_scores": report.per_question_scores,
        }
        json_path = out / "evaluation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info("Saved evaluation report JSON to %s", json_path)

        if report.metrics_dict:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            metrics = list(report.metrics_dict.keys())
            values = list(report.metrics_dict.values())
            axes[0].bar(metrics, values, color="steelblue")
            axes[0].set_ylim(0, 1)
            axes[0].set_title("RAGAS Metrics (Average)")
            axes[0].set_ylabel("Score")
            axes[0].tick_params(axis="x", rotation=30)
            for i, v in enumerate(values):
                axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

            if report.per_question_scores:
                for metric in metrics:
                    per_q = [s.get(metric, 0.0) for s in report.per_question_scores]
                    axes[1].plot(range(len(per_q)), per_q, marker="o", label=metric)
                axes[1].set_title("Per-Question Metric Scores")
                axes[1].set_xlabel("Question Index")
                axes[1].set_ylabel("Score")
                axes[1].legend()
                axes[1].set_ylim(0, 1)

            plt.tight_layout()
            plot_path = out / "evaluation_plots.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved evaluation plots to %s", plot_path)

# _r 20260523144003-6ff728af
