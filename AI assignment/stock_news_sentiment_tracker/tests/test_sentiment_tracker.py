from pathlib import Path
import json
import unittest

import pandas as pd

from src.sentiment_tracker import (
    classify_headlines,
    evaluate_predictions,
    normalize_headlines,
    run_pipeline,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class SentimentTrackerTests(unittest.TestCase):
    def test_generic_text_column_is_supported(self) -> None:
        headlines = normalize_headlines(pd.DataFrame({"title": ["Profit jumps on strong demand"]}))
        result = classify_headlines(headlines)
        self.assertEqual(result.loc[0, "predicted_sentiment"], "Positive")
        self.assertEqual(result.loc[0, "sentiment_score"], 1.0)

    def test_neutral_and_negative_examples(self) -> None:
        headlines = normalize_headlines(pd.DataFrame({
            "headline": ["Shares trade flat ahead of expiry", "Fire halts production for two weeks"]
        }))
        result = classify_headlines(headlines)
        self.assertEqual(result["predicted_sentiment"].tolist(), ["Neutral", "Negative"])

    def test_labels_evaluate_predictions_not_train_them(self) -> None:
        headlines = normalize_headlines(pd.DataFrame({
            "headline_id": ["a", "b", "c"],
            "headline": ["Profit jumps", "Shares trade flat", "Factory fire halts production"],
        }))
        classified = classify_headlines(headlines)
        labels = pd.DataFrame({
            "headline_id": ["a", "b", "c"],
            "true_sentiment": ["Positive", "Neutral", "Negative"],
        })
        evaluated, metrics = evaluate_predictions(classified, labels)
        self.assertEqual(evaluated["predicted_sentiment"].tolist(), ["Positive", "Neutral", "Negative"])
        self.assertEqual(metrics["evaluation_rows"], 3)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIn("only after", metrics["method"])

    def test_end_to_end_outputs_use_packaged_data(self) -> None:
        results = run_pipeline(PROJECT_DIR)
        metrics_file = PROJECT_DIR / "outputs" / "model_metrics.json"
        persisted = json.loads(metrics_file.read_text(encoding="utf-8"))
        self.assertEqual(len(results["classified"]), 300)
        self.assertEqual(results["metrics"]["evaluation_rows"], 300)
        self.assertAlmostEqual(results["metrics"]["accuracy"], persisted["accuracy"])
        self.assertTrue((PROJECT_DIR / "outputs" / "stock_news_sentiment_outputs.xlsx").exists())
        self.assertTrue((PROJECT_DIR / "report.docx").exists())


if __name__ == "__main__":
    unittest.main()
