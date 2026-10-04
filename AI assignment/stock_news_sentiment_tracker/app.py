from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st

from src.sentiment_tracker import (
    aggregate_daily_sentiment,
    classify_headlines,
    company_summary,
    create_ai_use_log,
    evaluate_predictions,
    generate_recommendation,
    make_sentiment_chart,
    normalize_headlines,
)


def build_workbook(
    classified: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    metrics: dict,
    ai_log: pd.DataFrame,
) -> bytes:
    from openpyxl.drawing.image import Image

    workbook_buffer = BytesIO()
    chart_buffer = BytesIO()
    figure = make_sentiment_chart(daily)
    figure.savefig(chart_buffer, format="png", dpi=140, bbox_inches="tight")
    chart_buffer.seek(0)
    import matplotlib.pyplot as plt
    plt.close(figure)

    with pd.ExcelWriter(workbook_buffer, engine="openpyxl") as writer:
        classified.to_excel(writer, sheet_name="Classified Headlines", index=False)
        daily.to_excel(writer, sheet_name="Daily Sentiment", index=False)
        summary.to_excel(writer, sheet_name="Company Summary", index=False)
        evaluation_rows.to_excel(writer, sheet_name="Evaluation", index=False)
        ai_log.to_excel(writer, sheet_name="AI Use Log", index=False)
        pd.DataFrame([
            {"metric": "Evaluation rows", "value": metrics.get("evaluation_rows", 0)},
            {"metric": "Accuracy", "value": metrics.get("accuracy")},
            {"metric": "Macro F1", "value": metrics.get("macro_f1")},
            {"metric": "Method", "value": metrics.get("method", "No gold labels supplied; accuracy not calculated.")},
        ]).to_excel(writer, sheet_name="Metrics", index=False)
        writer.book["Classified Headlines"].freeze_panes = "A2"
        writer.book["Daily Sentiment"].freeze_panes = "A2"
        chart_sheet = writer.book.create_sheet("Sentiment Chart")
        chart_sheet.add_image(Image(chart_buffer), "A1")

    return workbook_buffer.getvalue()


def create_report(classified: pd.DataFrame, summary: pd.DataFrame, metrics: dict) -> str:
    recommendation = generate_recommendation(summary)
    if metrics:
        validation = (
            f"Evaluation used {metrics['evaluation_rows']} matched rows. "
            f"Accuracy: {metrics['accuracy']:.2%}; macro-F1: {metrics['macro_f1']:.2%}. "
            "The labels were used only after predictions were produced."
        )
    else:
        validation = "No gold-label CSV was uploaded, so accuracy and F1 could not be calculated."
    counts = classified["predicted_sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )
    return f"""# Stock News Sentiment Analysis Report

## Objective and method
This report summarizes an analysis of {len(classified)} uploaded headlines. A transparent rule-based classifier assigns Positive, Neutral, or Negative using pre-defined headline wording patterns. It does not train on the evaluation labels. Sentiment scores are +1 for Positive, 0 for Neutral, and -1 for Negative.

## Results and validation
Predicted counts: Positive {counts['Positive']}, Neutral {counts['Neutral']}, and Negative {counts['Negative']}.

{validation}

## Business recommendation
{recommendation}

Use these results to prioritize headlines for analyst review. Confirm important claims against the original article and independent sources before decisions.

## Limitations and responsible use
Keyword rules can miss context, sarcasm, negation, mixed signals, and unfamiliar wording. The model assesses headline tone rather than factual truth, materiality, price impact, or a company's long-term health. Unknown dates are grouped separately; absent company names become "Unknown Company." This tool is not investment advice. The supplied assignment data is synthetic; do not upload personal or confidential information to public tools.

## Evaluation-label disclosure
An earlier AI-assisted prototype opened the assignment gold-label CSV and used it to test a supervised classifier. That prototype was discarded, but this conflicts with the assignment instruction not to give gold labels to an AI tool. The final rule-based classifier does not train on the labels, but this does not undo the earlier handling. Disclose this to the instructor and follow their guidance.

## Reflection
Review and personalize this section with your own experience before submitting. In particular, describe what you checked in the output, what surprised you, what you changed after testing, and what you would improve with more time. Do not claim checks or personal experiences you did not perform.
"""


st.set_page_config(page_title="CSV Stock News Sentiment App", page_icon="📊", layout="wide")
st.title("CSV Stock News Sentiment App")
st.caption(
    "Upload a CSV containing stock-news headlines. Optionally upload a separate gold-label CSV "
    "to evaluate predictions after classification."
)

with st.expander("Input format"):
    st.markdown(
        "- Headlines file: a text column named `headline`, `text`, `title`, `news`, or `article` is required.\n"
        "- Optional columns: `headline_id` (or `id`), `date`, `company` (or `ticker`), and `source`.\n"
        "- Evaluation file: `headline_id` plus `true_sentiment` (or `sentiment` / `label`) with Positive, Neutral, or Negative values.\n"
        "- This app is for stock-news headline CSVs, not arbitrary CSV data."
    )

headlines_file = st.file_uploader("Upload headline CSV", type=["csv"], key="headlines")
labels_file = st.file_uploader("Upload gold-label CSV (optional, evaluation only)", type=["csv"], key="labels")

if headlines_file is None:
    st.info("Upload a headline CSV to start.")
else:
    try:
        headlines = normalize_headlines(pd.read_csv(headlines_file))
        classified = classify_headlines(headlines)
        metrics = {}
        evaluation_rows = pd.DataFrame()
        if labels_file is not None:
            labels = pd.read_csv(labels_file)
            evaluation_rows, metrics = evaluate_predictions(classified, labels)
            normalized_labels = evaluation_rows[["headline_id", "true_sentiment"]]
            classified = classified.merge(normalized_labels, on="headline_id", how="left", validate="one_to_one")

        daily = aggregate_daily_sentiment(classified)
        summary = company_summary(daily)
        ai_log = create_ai_use_log()
        recommendation = generate_recommendation(summary)

        st.subheader("Classification")
        st.dataframe(classified, use_container_width=True, hide_index=True)
        st.download_button(
            "Download classified headlines CSV",
            classified.to_csv(index=False),
            file_name="classified_headlines.csv",
            mime="text/csv",
        )

        if metrics:
            st.subheader("Gold-label evaluation")
            first, second, third = st.columns(3)
            first.metric("Matched evaluation rows", metrics["evaluation_rows"])
            second.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            third.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
            st.caption(metrics["method"])
            st.dataframe(evaluation_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No accuracy is shown because no gold-label CSV was provided.")

        st.subheader("Daily sentiment index")
        chart = make_sentiment_chart(daily)
        st.pyplot(chart)
        st.dataframe(daily, use_container_width=True, hide_index=True)

        st.subheader("Business recommendation")
        st.write(recommendation)

        st.subheader("AI-use log")
        st.dataframe(ai_log, use_container_width=True, hide_index=True)

        report = create_report(classified, summary, metrics)
        workbook = build_workbook(classified, daily, summary, evaluation_rows, metrics, ai_log)
        st.subheader("Download deliverables")
        left, right = st.columns(2)
        left.download_button(
            "Download complete Excel workbook",
            workbook,
            file_name="stock_news_sentiment_outputs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        right.download_button(
            "Download report draft",
            report,
            file_name="report.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download AI-use log CSV",
            ai_log.to_csv(index=False),
            file_name="ai_use_log.csv",
            mime="text/csv",
        )
        st.warning(
            "The report includes a reflection prompt for you to personalize. Check the output and log "
            "and add your own genuine reflection before submitting."
        )
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Could not process the uploaded file: {exc}")
