# Stock News Sentiment Tracker

## Executive summary
This project classifies fictional business-news headlines as Positive, Neutral, or Negative, summarizes sentiment by company and date, and packages the results for review. It processed **300 headlines** for **10 companies** over **2025-04-02 to 2026-03-23**. The classifier is a fixed, keyword/phrase ruleset: Positive = +1, Neutral = 0, and Negative = -1. No gold labels are used to train it.

Predicted classes: **120 Positive**, **75 Neutral**, and **105 Negative**.

## 1. Objective and method
The business question is whether a simple tool can organize incoming company headlines into an interpretable first-pass sentiment signal. The workflow accepts a CSV, normalizes common column names, classifies headline text, produces a company/date index, and exports a workbook, CSVs, chart, and this report. Positive language includes improvements, contract wins, acquisitions, and upgrades; negative language includes downgrades, operational disruption, falling results, cost pressure, and compliance concerns. Unmatched or balanced signals default to Neutral.

This deliberately uses explicit rules rather than fitting a model to the evaluation sheet. It makes the behavior inspectable and prevents the gold labels from leaking into model training. The rules remain a small educational baseline, not a general-purpose language model.

## 2. Verification and results
The fixed rules were compared with 300 matched gold labels after classification. Accuracy was **84.7%** and macro-F1 was **82.4%**. No labels were used to fit or tune the classifier. Gold-label counts were {'Positive': 120, 'Negative': 107, 'Neutral': 73}.

### Confusion-matrix counts
Rows below are actual labels; columns are predicted labels.
- Actual Negative: predicted Negative 83, Neutral 24, Positive 0.
- Actual Neutral: predicted Negative 22, Neutral 51, Positive 0.
- Actual Positive: predicted Negative 0, Neutral 0, Positive 120.

### Error review
- NH0029: actual Negative; predicted Neutral; “Himal Cement CFO resigns citing personal reasons”
- NH0035: actual Neutral; predicted Negative; “Aarna Infotech completes routine plant maintenance shutdown”
- NH0038: actual Neutral; predicted Negative; “Ganga Power completes routine plant maintenance shutdown”
- NH0041: actual Negative; predicted Neutral; “Kiran Steel CFO resigns citing personal reasons”
- NH0043: actual Negative; predicted Neutral; “Kiran Steel CFO resigns citing personal reasons”
- NH0049: actual Negative; predicted Neutral; “Indra Telecom CFO resigns citing personal reasons”
- NH0051: actual Negative; predicted Neutral; “Kiran Steel CFO resigns citing personal reasons”
- NH0052: actual Negative; predicted Neutral; “Dhruv Bank CFO resigns citing personal reasons”

Accuracy alone can hide class-specific errors, so macro-F1 gives equal weight to each sentiment class. The full per-class precision/recall/F1 and row-level evaluation are in the workbook and `outputs/model_metrics.json`. Performance on this synthetic dataset should not be assumed to generalize to live news.

## 3. Company-level signal and recommendation
The daily sentiment index is the arithmetic mean of headline scores for each company/date; it ranges from -1 to +1. Company-level values weight each daily score by the number of headlines on that date. Highest average index: Himal Cement: +0.27 from 37 headlines, Kiran Steel: +0.27 from 30 headlines, Indra Telecom: +0.11 from 27 headlines. Lowest average index: Bhavya Pharma: -0.21 from 24 headlines, Dhruv Bank: -0.11 from 28 headlines, Aarna Infotech: -0.04 from 24 headlines.

Prioritize a review of positive headline activity at Himal Cement (+0.27, n=37), Kiran Steel (+0.27, n=30), Indra Telecom (+0.11, n=27). Monitor potential downside signals at Bhavya Pharma (-0.21, n=24), Dhruv Bank (-0.11, n=28), Aarna Infotech (-0.04, n=24). Use the index to triage news for human review, not as a standalone trading or investment signal.

For a manager, use this as a triage queue: review companies with multiple negative signals, verify the original reporting and business context, and monitor whether positive signals persist across dates. A single headline should not trigger a purchase or sale. The index does not include stock returns, volume, valuation, materiality, or source credibility, so it is not a prediction of market performance.

## 4. Iteration record
1. **Initial prototype:** a TF-IDF classifier was explored to see whether text classification was feasible. The initial full-data score was rejected as a validation result because training on evaluation labels can inflate reported performance.
2. **Leakage control:** the project changed to fixed, documented rules. Predictions are generated before the optional gold-label file is merged; labels are used only for the final evaluation report.
3. **Reliability and usability:** CSV column aliases, required headline checks, unique IDs, label validation, no-label behavior, CSV/workbook exports, and report generation were checked and tightened.

## 5. Responsible use, limitations, and reflection
The data supplied for this teaching task is fictional. The rules can miss context, negation, sarcasm, mixed sentiment, unfamiliar expressions, and the difference between a company's performance and the market's reaction. A headline may also be inaccurate or incomplete. If company/date fields are absent, the app supplies an explicit placeholder, reducing the usefulness of company and time aggregation. A real deployment would need human review, broader validation on independently sampled news, drift monitoring, and source-quality controls. No confidential or personal data should be uploaded to public AI services.

### Disclosure about evaluation labels
During an earlier AI-assisted prototype, the gold-label CSV was opened and used to test a supervised classifier. That prototype was discarded and is not part of this final rule-based classifier, but opening/using the evaluation labels conflicts with the assignment instruction not to give them to an AI tool. The final metric is from fixed rules applied before labels were joined; it does not undo the earlier handling. Disclose this to the instructor and follow their guidance rather than claiming the entire process complied with the restriction.

**Reflection to personalize before submission:** Review the classified rows, at least three correct and three incorrect/uncertain examples where available, and the error patterns. Add your own account of what you checked, what you learned, which limitation matters most, and what you would improve. Do not state that you performed a check unless you did.
