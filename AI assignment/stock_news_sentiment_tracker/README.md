# Stock News Sentiment Tracker

An end-to-end educational app that accepts stock-news headline CSVs, classifies headline tone as Positive / Neutral / Negative, calculates daily company sentiment, and exports evaluation and report deliverables.

## Try the hosted app

**Live app:** add the Streamlit Community Cloud URL here after deploying.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`. Upload a headline CSV. A separate label CSV is optional and is used only to measure evaluation metrics after predictions have been made.

To regenerate the included static deliverables:

```powershell
python run_pipeline.py
```

To run the automated checks:

```powershell
python -m unittest discover -s tests -v
```

## CSV format

The headlines CSV needs a text column called `headline`, `text`, `title`, `news`, or `article`. Optional fields:

| Field | Accepted alternatives | Purpose |
|---|---|---|
| `headline_id` | `id`, `news_id` | Match evaluation labels; generated if absent |
| `date` | `published_date`, `publish_date` | Daily aggregation |
| `company` | `firm`, `stock`, `ticker` | Company-level aggregation |
| `source` | `publisher`, `media` | Retained in classified output |

The optional label CSV needs `headline_id` and `true_sentiment` (alternatives: `id` / `news_id` and `sentiment` / `label`). Labels must be Positive, Neutral, or Negative. This is a stock-news application; arbitrary CSVs without headline text are not supported.

## Outputs

- `outputs/classified_headlines.csv` — all headline fields plus predicted sentiment and score.
- `outputs/daily_company_sentiment.csv` — company/date sentiment average and headline count.
- `outputs/company_summary.csv` — headline-count-weighted company-level averages.
- `outputs/evaluation_predictions.csv` — predictions matched to supplied gold labels.
- `outputs/model_metrics.json` — accuracy, macro-F1, class report, confusion matrix, evaluation method.
- `outputs/daily_sentiment_index.png` — overall time-series chart.
- `outputs/stock_news_sentiment_outputs.xlsx` — workbook with classified rows, evaluation, metrics, chart, AI-use log, and summaries.
- `report.docx` / `report.md` — report drafts. Add your personal reflection before submission.
- `ai_use_log.csv` — AI-assisted build activity summary; review it and keep it accurate.
- `FACULTY_EVALUATION.md` — quick-start and evaluation guide.

## Method and evaluation

The final classifier is a fixed, human-readable rule set; it is not fitted on gold labels. It creates predictions first and compares them with labels only afterward. With the supplied data, the final rules achieve **84.67% accuracy** and **82.41% macro-F1** over 300 labeled headlines. Without labels, the app shows no accuracy score.

Scores are Positive = +1, Neutral = 0, Negative = -1. The daily index is the mean headline score per company/date. Company averages are weighted by headline counts. This is a news-tone indicator, not a market-return forecast or investment recommendation.

## GitHub and hosted app setup

### 1. Create the GitHub repository

Create a new **public** repository on GitHub named `stock-news-sentiment-tracker`. Do not initialize it with a README, license, or `.gitignore` because those files are already in this project.

In PowerShell, run these commands from the project directory (replace `YOUR-USERNAME`):

```powershell
cd "C:\Users\sapta\Downloads\AI assignment\stock_news_sentiment_tracker"
git init -b main
git rev-parse --show-toplevel
git add .
git status
git commit -m "Submit stock news sentiment tracker"
git remote add origin https://github.com/YOUR-USERNAME/stock-news-sentiment-tracker.git
git push -u origin main
```

After `git rev-parse --show-toplevel`, confirm Git shows this project directory, not `C:\`. If it still prints `C:\`, do not run `git add`; follow the steps exactly from the project directory and initialize this project as its own repository. Inspect `git status` and confirm it lists only this project. Never run `git add .` from `C:\` or another project folder. The source dataset is synthetic course material; confirm your faculty permits publishing the supplied files before making the repository public.

After the push, open the repository URL and check that the README, app, `data/`, report, AI-use log, and workbook are present. GitHub Actions runs the automated checks on pushes and pull requests; confirm the workflow passes in the **Actions** tab.

### 2. Deploy the app for faculty

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app** / **Deploy an app**.
3. Select your `stock-news-sentiment-tracker` repository and `main` branch.
4. Set the main file path to `app.py`.
5. Deploy and wait for the app URL.
6. Test the public URL in a private/incognito browser window. Upload `data/headlines.csv`, then separately upload `data/gold_labels.csv` to check metrics and workbook downloads.
7. Replace the live app placeholder near the top of this README with the actual URL. Push that README edit to GitHub.

Keep the Streamlit app public if your faculty needs to access it without signing in. Free hosting policies and limits may change.

## Faculty evaluation

See [FACULTY_EVALUATION.md](./FACULTY_EVALUATION.md) for the exact demo steps, expected metrics, output checklist, and method notes.

## Responsible use and assignment disclosure

The provided assignment data is synthetic. The simple rules can miss negation, sarcasm, context, mixed sentiment, and unfamiliar language. The tool evaluates headline tone, not truth, business materiality, fundamentals, or stock performance. Do not upload confidential or personal data; do not use the output as financial advice.

**Gold-label disclosure:** an earlier AI-assisted prototype opened and used the assignment gold-label CSV to test a supervised classifier. That prototype was discarded, but this conflicts with the task instruction not to give gold labels to an AI tool. The final rules do not train on labels; this does not undo the earlier handling. This is disclosed in the report and evaluation guide. Follow your instructor's guidance and be transparent about it.

The report is a draft: personalize the reflection with your own genuine observations. Follow your instructor's requirements for a live demo or recording.
