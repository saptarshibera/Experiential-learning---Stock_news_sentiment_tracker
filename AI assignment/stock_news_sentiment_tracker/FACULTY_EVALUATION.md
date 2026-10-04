# Faculty Evaluation Guide

## Launch the app

The app can be evaluated in a browser through the submitted Streamlit deployment link (add the deployed URL to the project README), or run locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Quick evaluation with the supplied synthetic data

Use the two CSVs in `data/`:

1. Upload `data/headlines.csv` as the headline file.
2. Upload `data/gold_labels.csv` in the separate evaluation-label field.
3. Inspect the displayed classification and evaluation summary.
4. Download the Excel workbook and check its `Classified Headlines`, `Evaluation`, `Metrics`, `Daily Sentiment`, `Company Summary`, `AI Use Log`, and `Sentiment Chart` sheets.

The pipeline can also generate the same static deliverables without the app:

```bash
python run_pipeline.py
```

## What to verify

- Expected demonstration input: 300 headlines with 300 matching labels.
- Expected final rule-based results: accuracy approximately 84.67%; macro-F1 approximately 82.41%. Small display rounding may differ.
- Sentiment scores: Positive `+1`, Neutral `0`, Negative `-1`.
- The company/date index is the mean of row sentiment scores; the company-level average weights by headline count.
- Predictions are created before labels are joined. The labels are used to evaluate, not train, the final classifier.
- The app also works without a labels file, but it must not display accuracy in that case.
- For a custom test file, use a CSV with a text column named `headline`, `text`, `title`, `news`, or `article`. Company, date, source, and headline IDs are optional.

## Known limitations and disclosure

This is an educational keyword/phrase baseline. It may miss negation, sarcasm, context, novel phrasing, or mixed sentiment. It is not investment advice.

An earlier AI-assisted prototype opened and used the assignment gold-label CSV to test a supervised classifier. That prototype was discarded, but this conflicts with the assignment instruction not to give gold labels to an AI tool. The final classifier does not train on the labels; this does not undo the earlier handling. This is disclosed in `README.md`, `report.md`, and `report.docx`.

## Submission notes for the student

- Add your actual public GitHub URL and deployed app URL to the README after publishing.
- Personalize the reflection in the report with your own genuine observations.
- Provide any required screen recording or live demo separately.
- Confirm the instructor accepts the disclosed gold-label handling before claiming the task fully complies with the instructions.
