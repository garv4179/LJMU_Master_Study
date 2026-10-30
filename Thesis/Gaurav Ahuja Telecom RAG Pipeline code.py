"""
Telecom RAG Thesis — End-to-End Experimental Framework
======================================================
Dataset:
https://www.kaggle.com/datasets/syncoraai/customer-support-conversations
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------

DATA_PATH = os.getenv("CUSTOMER_SUPPORT_DATA_PATH", "customer_support_data.csv")
OUTPUT_DIR = Path(os.getenv("TELECOM_RAG_OUTPUT_DIR", "telecom_rag_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K = 5
CANDIDATE_K = 10
OPTIMISED_TOP_K = 3
MAX_EVALUATION_QUERIES = 25

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Keep False for thesis-quality telecom-only experiments. Enable only when you
# intentionally want to test the code with a general support dataset.
ALLOW_NON_TELECOM_FALLBACK = False

USE_AZURE_OPENAI = all(
    os.getenv(name)
    for name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    )
)

TELECOM_TERMS = (
    "telecom", "telecommunications", "mobile network", "broadband",
    "sim card", "esim", "roaming", "5g", "4g", "network outage",
    "mobile data", "carrier", "phone service", "call drop", "sms", "otp",
)

# ---------------------------------------------------------------------
# 2. HELPERS: COLUMN NORMALISATION AND TEXT CLEANING
# ---------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """Convert a value to clean, single-line text; missing values become empty."""
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Find a column case-insensitively from a list of aliases."""
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def series_or_default(
    df: pd.DataFrame, column: Optional[str], default: str = "Unknown"
) -> pd.Series:
    if column is None:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column].fillna(default).astype(str)


def load_raw_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load a local CSV. Download the Kaggle file manually/API before running."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}\n"
            "Download the CSV from "
            "https://www.kaggle.com/datasets/syncoraai/customer-support-conversations "
            "and save it to DATA_PATH, or set CUSTOMER_SUPPORT_DATA_PATH."
        )
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(file_path)
    else:
        raw = pd.read_csv(file_path, low_memory=False)
    if raw.empty:
        raise ValueError(f"The dataset file is empty: {file_path}")
    print(f"Loaded {len(raw):,} rows from {file_path.resolve()}")
    print("Source columns:", ", ".join(map(str, raw.columns)))
    return raw


def normalise_dataset(raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Map common Syncora conversation/ticket schemas into one canonical schema.

    Canonical message-level columns:
      conv_id, turn_index, role, text, industry, product, issue_type,
      language, channel, overall_sentiment, overall_urgency, outcome,
      primary_intent
    """
    df = raw.copy()
    conv_col = find_column(df, [
        "conversation_id", "conversationid", "conv_id", "ticket_id",
        "case_id", "session_id", "id",
    ])
    turn_col = find_column(df, [
        "turn_id", "turn_index", "message_index", "sequence", "sequence_id",
        "message_id", "timestamp", "created_at",
    ])
    role_col = find_column(df, ["role", "sender", "author", "speaker", "from"])
    text_col = find_column(df, [
        "message", "text", "content", "utterance", "message_text",
        "customer_message", "body",
    ])
    subject_col = find_column(df, ["subject", "title", "ticket_subject"])
    answer_col = find_column(df, [
        "answer", "response", "resolution_text", "agent_response",
        "suggested_answer",
    ])

    industry_col = find_column(df, ["industry", "domain", "sector"])
    product_col = find_column(df, ["product", "service", "plan", "category"])
    issue_col = find_column(df, [
        "issue_type", "sub_category", "subcategory", "issue", "topic", "type",
        "queue",
    ])
    language_col = find_column(df, ["language", "locale", "lang"])
    channel_col = find_column(df, ["channel", "source", "platform"])
    sentiment_col = find_column(df, ["overall_sentiment", "sentiment"])
    urgency_col = find_column(df, ["overall_urgency", "urgency", "priority"])
    outcome_col = find_column(df, ["outcome", "status", "resolution_status"])
    intent_col = find_column(df, ["primary_intent", "intent", "user_intent"])

    # Ticket-style data often has one row with a question/body and an answer.
    if answer_col is not None and (text_col is not None or subject_col is not None):
        query_source = (
            df[text_col].fillna("").astype(str)
            if text_col is not None
            else pd.Series([""] * len(df), index=df.index)
        )
        if subject_col is not None:
            subjects = df[subject_col].fillna("").astype(str)
            query_source = (subjects.str.strip() + ". " + query_source.str.strip()).str.strip(". ")
        out = pd.DataFrame({
            "conv_id": (
                df[conv_col].astype(str) if conv_col is not None
                else pd.Series([f"ticket_{i}" for i in range(len(df))], index=df.index)
            ),
            "turn_index": 0,
            "role": "Customer",
            "text": query_source.map(clean_text),
            "industry": series_or_default(df, industry_col, "Unknown"),
            "product": series_or_default(df, product_col, "Unknown"),
            "issue_type": series_or_default(df, issue_col, "Unknown"),
            "language": series_or_default(df, language_col, "Unknown"),
            "channel": series_or_default(df, channel_col, "Unknown"),
            "overall_sentiment": series_or_default(df, sentiment_col, "Unknown"),
            "overall_urgency": series_or_default(df, urgency_col, "Unknown"),
            "outcome": series_or_default(df, outcome_col, "Unknown"),
            "primary_intent": series_or_default(df, intent_col, "Unknown"),
            "reference_answer": df[answer_col].map(clean_text),
        })
        out["reference_answer"] = out["reference_answer"].fillna("")
        out = out[(out["text"].str.len() > 0) & (out["reference_answer"].str.len() > 0)]
        return out.reset_index(drop=True), "ticket_question_answer"

    if text_col is None:
        raise ValueError(
            "Could not find a text/message column. Expected one of: "
            "message, text, content, utterance, body."
        )

    if role_col is None:
        raise ValueError(
            "Could not find a role/sender column for message-level data, and no "
            "answer column was found for ticket-style data."
        )

    out = pd.DataFrame({
        "conv_id": (
            df[conv_col].astype(str) if conv_col is not None
            else pd.Series([f"conversation_{i}" for i in range(len(df))], index=df.index)
        ),
        "turn_index": (
            df[turn_col] if turn_col is not None
            else pd.Series(np.arange(len(df)), index=df.index)
        ),
        "role": df[role_col].fillna("Unknown").astype(str),
        "text": df[text_col].map(clean_text),
        "industry": series_or_default(df, industry_col, "Unknown"),
        "product": series_or_default(df, product_col, "Unknown"),
        "issue_type": series_or_default(df, issue_col, "Unknown"),
        "language": series_or_default(df, language_col, "Unknown"),
        "channel": series_or_default(df, channel_col, "Unknown"),
        "overall_sentiment": series_or_default(df, sentiment_col, "Unknown"),
        "overall_urgency": series_or_default(df, urgency_col, "Unknown"),
        "outcome": series_or_default(df, outcome_col, "Unknown"),
        "primary_intent": series_or_default(df, intent_col, "Unknown"),
    })
    out = out[out["text"].str.len() > 0].copy()

    # Use a stable ordering even if turn identifiers are strings or timestamps.
    out["_turn_sort"] = out["turn_index"].astype(str)
    out = out.sort_values(["conv_id", "_turn_sort"], kind="stable").drop(columns="_turn_sort")
    out["turn_index"] = out.groupby("conv_id").cumcount()
    return out.reset_index(drop=True), "message_level"


# ---------------------------------------------------------------------
# 3. DATA SELECTION
# ---------------------------------------------------------------------

def select_research_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Select telecom rows when the source has an industry/category/domain field.
    If the dataset has no telecom-labelled rows, stop by default so the thesis
    does not misrepresent general customer-support data as telecom data.
    """
    searchable_cols = [
        col for col in ("industry", "product", "issue_type", "primary_intent", "text")
        if col in df.columns
    ]
    combined = df[searchable_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    telecom_mask = combined.apply(lambda value: any(term in value for term in TELECOM_TERMS))
    telecom = df[telecom_mask].copy()

    if not telecom.empty:
        print(f"Selected {len(telecom):,} telecom-related rows from {len(df):,} rows.")
        return telecom.reset_index(drop=True), "telecom_filtered"

    if ALLOW_NON_TELECOM_FALLBACK:
        print(
            "WARNING: No telecom-labelled rows found. Running in GENERAL CUSTOMER "
            "SUPPORT mode because ALLOW_NON_TELECOM_FALLBACK=True."
        )
        return df.copy().reset_index(drop=True), "general_customer_support_fallback"

    raise ValueError(
        "No telecom-related rows were found in this dataset. The Kaggle dataset is "
        "a general customer-support dataset and may not contain telecom examples. "
        "For a telecom thesis, combine it with telecom-specific FAQs, troubleshooting "
        "manuals, 3GPP standards, or a telecom-labelled dataset. For pipeline testing "
        "only, set ALLOW_NON_TELECOM_FALLBACK=True."
    )


# ---------------------------------------------------------------------
# 4. DATA PRE-PROCESSING AND CUSTOMER/AGENT QA PAIRS
# ---------------------------------------------------------------------

CUSTOMER_ROLES = {"customer", "user", "client", "consumer", "requester"}
AGENT_ROLES = {"agent", "assistant", "support", "bot", "representative", "advisor"}


def create_qa_pairs(df: pd.DataFrame, schema_type: str) -> pd.DataFrame:
    if schema_type == "ticket_question_answer":
        qa = df.copy()
        qa["query"] = qa["text"]
        qa["reference_answer"] = qa["reference_answer"].map(clean_text)
        return qa[[
            "conv_id", "query", "reference_answer", "industry", "product",
            "issue_type", "language", "channel", "overall_sentiment",
            "overall_urgency", "outcome", "primary_intent",
        ]].reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    for conversation_id, group in df.groupby("conv_id", sort=False):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            current = group.iloc[i]
            following = group.iloc[i + 1]
            current_role = clean_text(current["role"]).lower()
            next_role = clean_text(following["role"]).lower()

            if current_role in CUSTOMER_ROLES and next_role in AGENT_ROLES:
                records.append({
                    "conv_id": conversation_id,
                    "query": current["text"],
                    "reference_answer": following["text"],
                    "industry": current["industry"],
                    "product": current["product"],
                    "issue_type": current["issue_type"],
                    "language": current["language"],
                    "channel": current["channel"],
                    "overall_sentiment": current["overall_sentiment"],
                    "overall_urgency": current["overall_urgency"],
                    "outcome": following["outcome"],
                    "primary_intent": current["primary_intent"],
                })

    qa = pd.DataFrame(records)
    if qa.empty:
        raise ValueError(
            "No customer→agent QA pairs were extracted. Check the dataset's role labels "
            "and columns. Recognised customer roles: "
            f"{sorted(CUSTOMER_ROLES)}; agent roles: {sorted(AGENT_ROLES)}."
        )
    return qa.reset_index(drop=True)


def conversation_level_split(qa: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(qa) < 2:
        raise ValueError("At least two QA pairs are required for a train/test split.")
    ids = qa["conv_id"].drop_duplicates().tolist()
    if len(ids) < 2:
        # Ticket-style datasets may contain unique ticket ids; if only one ID remains,
        # split rows as a last-resort fallback and document it in the output.
        train, test = train_test_split(
            qa, test_size=min(TEST_SIZE, 0.5), random_state=RANDOM_STATE
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    train_ids, test_ids = train_test_split(
        ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train = qa[qa["conv_id"].isin(train_ids)].reset_index(drop=True)
    test = qa[qa["conv_id"].isin(test_ids)].reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------

def create_bar_plot(series: pd.Series, title: str, filename: str) -> None:
    values = series.fillna("Unknown").astype(str).value_counts().head(15).sort_values()
    if values.empty:
        return
    plt.figure(figsize=(10, 6))
    values.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close()



def _save_eda_plot(filename: str) -> None:
    """Save the current Matplotlib figure to the configured EDA output folder."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def _bar_plot(series: pd.Series, title: str, filename: str, *,
              horizontal: bool = True, top_n: int = 15, rotation: int = 0) -> None:
    """Create a readable categorical bar chart while handling missing values."""
    values = series.fillna("Unknown").astype(str).replace("", "Unknown")
    counts = values.value_counts().head(top_n)
    if counts.empty:
        return
    plt.figure(figsize=(11, max(5, min(10, 0.34 * len(counts) + 1.5))))
    if horizontal:
        counts.sort_values().plot(kind="barh")
        plt.xlabel("Number of records")
        plt.ylabel("")
    else:
        counts.plot(kind="bar", rot=rotation)
        plt.ylabel("Number of records")
        plt.xlabel("")
    plt.title(title)
    _save_eda_plot(filename)


def _stacked_bar_plot(df: pd.DataFrame, row_col: str, column_col: str,
                      title: str, filename: str, *,
                      top_rows: int = 10, top_columns: int = 8) -> None:
    """Create stacked bars to compare categories across groups."""
    if row_col not in df.columns or column_col not in df.columns or df.empty:
        return
    work = df[[row_col, column_col]].fillna("Unknown").astype(str)
    work = work.replace("", "Unknown")
    top_row_values = work[row_col].value_counts().head(top_rows).index
    top_col_values = work[column_col].value_counts().head(top_columns).index
    work = work[work[row_col].isin(top_row_values)].copy()
    work[column_col] = np.where(work[column_col].isin(top_col_values), work[column_col], "Other")
    table = pd.crosstab(work[row_col], work[column_col])
    if table.empty:
        return
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel(row_col.replace("_", " ").title())
    ax.set_ylabel("Number of records")
    ax.legend(title=column_col.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_eda_plot(filename)


def _plot_missingness(raw: pd.DataFrame) -> None:
    """Plot missing values per source column."""
    if raw.empty:
        return
    missing_pct = (raw.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(25)
    if missing_pct.empty:
        return
    plt.figure(figsize=(11, max(5, 0.32 * len(missing_pct) + 1)))
    missing_pct.sort_values().plot(kind="barh")
    plt.title("Missing Values by Source Column")
    plt.xlabel("Missing values (%)")
    plt.ylabel("Source column")
    _save_eda_plot("EDA_09_missing_values.png")


def _plot_numeric_correlations(raw: pd.DataFrame) -> None:
    """Plot a correlation heatmap when the source has at least two numeric features."""
    numeric = raw.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(corr.columns)), max(6, 0.7 * len(corr.columns))))
    image = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    _save_eda_plot("EDA_10_numeric_correlation_heatmap.png")


def _plot_term_frequency(texts: pd.Series, title: str, filename: str,
                         ngram_range=(1, 1)) -> None:
    """Plot frequent words or bigrams without an extra word-cloud dependency."""
    cleaned = texts.fillna("").astype(str)
    cleaned = cleaned[cleaned.str.strip().ne("")]
    if cleaned.empty:
        return
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=ngram_range, max_features=25,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
        )
        matrix = vectorizer.fit_transform(cleaned.tolist())
        totals = np.asarray(matrix.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        order = np.argsort(totals)[-20:]
        if len(order) == 0:
            return
        plt.figure(figsize=(11, 7))
        plt.barh(terms[order], totals[order])
        plt.title(title)
        plt.xlabel("Aggregate TF-IDF weight")
        plt.ylabel("Term")
        _save_eda_plot(filename)
    except ValueError:
        return


def run_dataset_eda(raw: pd.DataFrame, selected: pd.DataFrame, qa: pd.DataFrame) -> None:
    """Generate descriptive EDA charts and machine-readable summaries for the RAG study."""
    print("\\n========== DATASET EDA ==========")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data quality and source-level structure
    _plot_missingness(raw)
    _plot_numeric_correlations(raw)

    # Horizontal bar charts for category distributions
    categorical_plots = [
        ("industry", "Industry Distribution", "EDA_01_industry_bar.png"),
        ("product", "Product / Category Distribution", "EDA_02_product_category_bar.png"),
        ("issue_type", "Issue / Sub-category Distribution", "EDA_03_issue_subcategory_bar.png"),
        ("language", "Language / Locale Distribution", "EDA_04_language_bar.png"),
        ("channel", "Support Channel Distribution", "EDA_05_channel_bar.png"),
        ("overall_sentiment", "Customer Sentiment Distribution", "EDA_06_sentiment_bar.png"),
        ("overall_urgency", "Priority / Urgency Distribution", "EDA_13_priority_urgency_bar.png"),
        ("outcome", "Support Outcome / Status Distribution", "EDA_14_outcome_status_bar.png"),
        ("primary_intent", "Customer Intent Distribution", "EDA_15_intent_bar.png"),
        ("role", "Message Role Distribution", "EDA_16_message_role_bar.png"),
    ]
    for col, title, filename in categorical_plots:
        if col in selected.columns:
            _bar_plot(selected[col], title, filename, horizontal=True, top_n=20)

    # Vertical bars provide a second view of key distributions
    for col, title, filename in [
        ("channel", "Support Channel Distribution (Vertical)", "EDA_17_channel_vertical_bar.png"),
        ("overall_sentiment", "Sentiment Distribution (Vertical)", "EDA_18_sentiment_vertical_bar.png"),
    ]:
        if col in selected.columns:
            _bar_plot(selected[col], title, filename, horizontal=False, top_n=12, rotation=35)

    # Cross-tabulation and stacked bars
    for row_col, col_col, title, filename in [
        ("industry", "overall_sentiment", "Sentiment by Industry", "EDA_19_sentiment_by_industry_stacked.png"),
        ("channel", "overall_sentiment", "Sentiment by Support Channel", "EDA_20_sentiment_by_channel_stacked.png"),
        ("issue_type", "overall_urgency", "Priority by Issue Type", "EDA_21_priority_by_issue_stacked.png"),
        ("primary_intent", "outcome", "Outcome by Customer Intent", "EDA_22_outcome_by_intent_stacked.png"),
    ]:
        _stacked_bar_plot(selected, row_col, col_col, title, filename)

    # Message, query, and answer length analysis
    if "text" in selected.columns and not selected.empty:
        lengths = selected["text"].fillna("").astype(str).str.split().str.len()
        plt.figure(figsize=(9, 5))
        lengths.clip(upper=200).plot(kind="hist", bins=35, edgecolor="black")
        plt.title("Message Word-Count Distribution (values clipped at 200)")
        plt.xlabel("Words per message")
        plt.ylabel("Frequency")
        _save_eda_plot("EDA_07_message_length_histogram.png")

        plt.figure(figsize=(9, 5))
        plt.boxplot(lengths.clip(upper=200), orientation="horizontal", showfliers=False)
        plt.title("Message Length Boxplot (values clipped at 200)")
        plt.xlabel("Word count")
        _save_eda_plot("EDA_08_message_length_boxplot.png")

        _plot_term_frequency(selected["text"], "Frequent Terms in Support Messages",
                             "EDA_23_top_terms_messages.png", (1, 1))
        _plot_term_frequency(selected["text"], "Frequent Bigrams in Support Messages",
                             "EDA_24_top_bigrams_messages.png", (2, 2))

    if not qa.empty:
        qa = qa.copy()
        qa["query_length"] = qa["query"].fillna("").astype(str).str.split().str.len()
        qa["answer_length"] = qa["reference_answer"].fillna("").astype(str).str.split().str.len()

        plt.figure(figsize=(9, 5))
        plt.scatter(qa["query_length"].clip(upper=100), qa["answer_length"].clip(upper=150), alpha=0.35)
        plt.title("Customer Query Length vs Reference Answer Length")
        plt.xlabel("Query words (clipped at 100)")
        plt.ylabel("Answer words (clipped at 150)")
        _save_eda_plot("EDA_11_query_answer_length_scatter.png")

        plt.figure(figsize=(9, 5))
        plt.boxplot([qa["query_length"].clip(upper=100), qa["answer_length"].clip(upper=150)],
                    tick_labels=["Customer query", "Reference answer"], showfliers=False)
        plt.title("Query and Reference Answer Length Comparison")
        plt.ylabel("Word count (clipped for readability)")
        _save_eda_plot("EDA_12_query_answer_length_boxplot.png")
        qa[["query_length", "answer_length"]].describe().round(3).to_csv(
            OUTPUT_DIR / "EDA_text_length_statistics.csv"
        )
        _plot_term_frequency(qa["query"], "Frequent Terms in Customer Queries",
                             "EDA_25_top_terms_queries.png", (1, 1))
        _plot_term_frequency(qa["reference_answer"], "Frequent Terms in Reference Answers",
                             "EDA_26_top_terms_answers.png", (1, 1))

    # Optional temporal trend when the source includes a timestamp
    timestamp_col = find_column(raw, ["timestamp", "created_at", "date", "datetime", "created"])
    if timestamp_col is not None:
        dates = pd.to_datetime(raw[timestamp_col], errors="coerce", utc=True).dropna()
        if not dates.empty:
            monthly = dates.dt.strftime("%Y-%m").value_counts().sort_index()
            if not monthly.empty:
                plt.figure(figsize=(11, 5))
                monthly.plot(kind="line", marker="o")
                plt.title("Conversation / Message Volume Over Time")
                plt.xlabel("Month")
                plt.ylabel("Number of records")
                plt.xticks(rotation=45, ha="right")
                _save_eda_plot("EDA_27_monthly_volume_trend.png")

    # Pie charts for compact summaries of small category sets
    for col, title, filename in [
        ("overall_sentiment", "Customer Sentiment Share", "EDA_28_sentiment_pie.png"),
        ("channel", "Support Channel Share", "EDA_29_channel_pie.png"),
    ]:
        if col in selected.columns:
            counts = selected[col].fillna("Unknown").astype(str).value_counts().head(8)
            if len(counts) >= 2:
                plt.figure(figsize=(8, 7))
                plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
                plt.title(title)
                _save_eda_plot(filename)

    # Export summary statistics and frequency tables for thesis reporting
    selected_copy = selected.copy()
    if "text" in selected_copy.columns:
        selected_copy["word_count"] = selected_copy["text"].fillna("").astype(str).str.split().str.len()

    summary = {
        "raw_rows": int(len(raw)),
        "raw_columns": list(map(str, raw.columns)),
        "selected_rows": int(len(selected)),
        "selected_conversations": int(selected["conv_id"].nunique()) if "conv_id" in selected else None,
        "qa_pairs": int(len(qa)),
        "unique_industries": int(selected["industry"].nunique()) if "industry" in selected else None,
        "unique_products_or_categories": int(selected["product"].nunique()) if "product" in selected else None,
        "unique_issues_or_subcategories": int(selected["issue_type"].nunique()) if "issue_type" in selected else None,
        "unique_languages": int(selected["language"].nunique()) if "language" in selected else None,
        "missing_cells_raw": int(raw.isna().sum().sum()),
        "missing_cells_raw_percent": round(float(raw.isna().sum().sum() / max(1, raw.size) * 100), 3),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "eda_plot_directory": str(OUTPUT_DIR.resolve()),
    }
    if "word_count" in selected_copy.columns and not selected_copy.empty:
        summary["message_word_count_descriptive_statistics"] = selected_copy["word_count"].describe().round(3).to_dict()
    if not qa.empty:
        summary["query_word_count_mean"] = round(float(qa["query_length"].mean()), 3)
        summary["answer_word_count_mean"] = round(float(qa["answer_length"].mean()), 3)

    (OUTPUT_DIR / "EDA_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    freq_rows = []
    for col in ["industry", "product", "issue_type", "language", "channel",
                "overall_sentiment", "overall_urgency", "outcome", "primary_intent", "role"]:
        if col in selected.columns:
            counts = selected[col].fillna("Unknown").astype(str).value_counts(dropna=False)
            freq_rows.extend({"column": col, "category": str(cat), "count": int(count)}
                             for cat, count in counts.items())
    if freq_rows:
        pd.DataFrame(freq_rows).to_csv(OUTPUT_DIR / "EDA_category_frequency_tables.csv", index=False)

    print(json.dumps(summary, indent=2, default=str))
    print(f"EDA plots and tables saved to: {OUTPUT_DIR.resolve()}")


# ---------------------------------------------------------------------
# 6. TRADITIONAL RETRIEVAL: TF-IDF
# ---------------------------------------------------------------------

class TFIDFRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=150_000,
            sublinear_tf=True, strip_accents="unicode",
        )
        self.matrix = None

    def fit(self, documents: Sequence[str]) -> None:
        safe_docs = [clean_text(doc) or "empty document" for doc in documents]
        self.matrix = self.vectorizer.fit_transform(safe_docs)

    def search(self, query: str, top_k: int = TOP_K) -> Tuple[List[int], List[float]]:
        if self.matrix is None:
            raise RuntimeError("TF-IDF retriever has not been fitted.")
        query_vector = self.vectorizer.transform([clean_text(query)])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        k = min(max(1, top_k), len(scores))
        indices = np.argsort(-scores)[:k]
        return indices.astype(int).tolist(), scores[indices].astype(float).tolist()


# ---------------------------------------------------------------------
# 7. VECTOR DATABASE DEPLOYMENT: BGE-M3 + OPTIONAL FAISS
# ---------------------------------------------------------------------

class VectorRetriever:
    """
    Uses BGE-M3 sentence embeddings when sentence-transformers is available.
    Otherwise, uses a TF-IDF-based semantic-search fallback (not random vectors).
    FAISS is optional; NumPy cosine/dot-product search is used if FAISS is absent.
    """
    def __init__(self) -> None:
        self.model = None
        self.index = None
        self.embeddings = None
        self.fallback_vectorizer = None
        self.backend = "uninitialised"

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.backend = f"SentenceTransformer:{EMBEDDING_MODEL}"
        except Exception as exc:
            print(f"Embedding model unavailable ({exc.__class__.__name__}); using TF-IDF fallback.")
            self.fallback_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=100_000, sublinear_tf=True,
            )
            self.backend = "TF-IDF fallback"

    def fit(self, documents: Sequence[str]) -> None:
        safe_docs = [clean_text(doc) or "empty document" for doc in documents]
        if self.model is not None:
            self.embeddings = self.model.encode(
                safe_docs, batch_size=32, normalize_embeddings=True,
                show_progress_bar=True, convert_to_numpy=True,
            ).astype("float32")
        else:
            matrix = self.fallback_vectorizer.fit_transform(safe_docs)
            self.embeddings = matrix.toarray().astype("float32")
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

        try:
            import faiss
            self.index = faiss.IndexFlatIP(int(self.embeddings.shape[1]))
            self.index.add(self.embeddings)
            self.backend += " + FAISS"
        except Exception:
            self.index = None
            self.backend += " + NumPy search"

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(
                [clean_text(query)], normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
        vec = self.fallback_vectorizer.transform([clean_text(query)]).toarray().astype("float32")
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def search(self, query: str, top_k: int = TOP_K) -> Tuple[List[int], List[float]]:
        if self.embeddings is None:
            raise RuntimeError("Vector retriever has not been fitted.")
        q = self._encode_query(query)
        k = min(max(1, top_k), len(self.embeddings))
        if self.index is not None:
            scores, indices = self.index.search(q, k)
            valid = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
            return [x[0] for x in valid], [x[1] for x in valid]
        scores = self.embeddings @ q[0]
        indices = np.argsort(-scores)[:k]
        return indices.astype(int).tolist(), scores[indices].astype(float).tolist()


# ---------------------------------------------------------------------
# 8. HYBRID RETRIEVAL + RERANKING
# ---------------------------------------------------------------------

def row_metadata_match(a: pd.Series, b: pd.Series) -> bool:
    """A weak relevance proxy using available labels; not a human relevance judgement."""
    checks = []
    for field in ("product", "issue_type", "primary_intent"):
        av, bv = clean_text(a.get(field, "")).lower(), clean_text(b.get(field, "")).lower()
        if av and bv and av != "unknown" and bv != "unknown":
            checks.append(av == bv)
    return bool(checks) and any(checks)


class OptimisedRetriever:
    def __init__(
        self, train: pd.DataFrame, vector_retriever: VectorRetriever,
        tfidf_retriever: TFIDFRetriever,
    ) -> None:
        self.train = train
        self.vector = vector_retriever
        self.tfidf = tfidf_retriever
        self.reranker = None
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
            print(f"Loaded reranker: {RERANKER_MODEL}")
        except Exception as exc:
            print(f"Reranker unavailable ({exc.__class__.__name__}); using hybrid score.")

    def search(self, row: pd.Series, top_k: int = OPTIMISED_TOP_K) -> Tuple[List[int], List[float]]:
        query = row["query"]
        ti, ts = self.tfidf.search(query, CANDIDATE_K)
        vi, vs = self.vector.search(query, CANDIDATE_K)
        candidates = list(dict.fromkeys(ti + vi))

        # Apply metadata filtering only when it leaves enough candidates.
        filtered = [
            idx for idx in candidates
            if row_metadata_match(row, self.train.iloc[idx])
        ]
        if len(filtered) >= min(top_k, 2):
            candidates = filtered

        if self.reranker is not None and candidates:
            pairs = [[query, self.train.iloc[idx]["reference_answer"]] for idx in candidates]
            scores = np.asarray(self.reranker.predict(pairs)).reshape(-1)
            order = np.argsort(-scores)[:top_k]
            return [candidates[i] for i in order], [float(scores[i]) for i in order]

        ti_map, vi_map = dict(zip(ti, ts)), dict(zip(vi, vs))
        combined = [
            (idx, 0.35 * ti_map.get(idx, 0.0) + 0.65 * vi_map.get(idx, 0.0))
            for idx in candidates
        ]
        combined.sort(key=lambda pair: pair[1], reverse=True)
        selected = combined[:top_k]
        return [x[0] for x in selected], [float(x[1]) for x in selected]


# ---------------------------------------------------------------------
# 9. RAG INTEGRATION AND CONFIGURATION / RESPONSE GENERATION
# ---------------------------------------------------------------------

def extractive_generation(query: str, contexts: Sequence[str]) -> str:
    if not contexts:
        return "I could not find sufficient information in the retrieved support knowledge."
    query_terms = set(re.findall(r"\b\w+\b", clean_text(query).lower()))
    best_context, best_score = contexts[0], -1.0
    for context in contexts:
        terms = set(re.findall(r"\b\w+\b", clean_text(context).lower()))
        score = len(query_terms & terms) / max(1, len(query_terms))
        if score > best_score:
            best_context, best_score = context, score
    return best_context


def generate_rag_answer(query: str, contexts: Sequence[str]) -> Tuple[str, str]:
    """Return (answer, generation_backend). Never silently claim Azure was used."""
    if USE_AZURE_OPENAI:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
            context = "\n\n".join(f"[Retrieved source {i+1}]\n{c}" for i, c in enumerate(contexts))
            response = client.chat.completions.create(
                model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a customer-support assistant. Answer only using the supplied "
                            "retrieved context. Do not invent policies, prices, or troubleshooting "
                            "steps. If the context is insufficient, say so and recommend escalation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Customer query:\n{query}\n\nRetrieved context:\n{context}",
                    },
                ],
            )
            answer = response.choices[0].message.content or ""
            return answer.strip(), "Azure OpenAI"
        except Exception as exc:
            print(f"Azure OpenAI generation failed ({exc.__class__.__name__}); using extractive fallback.")
    return extractive_generation(query, contexts), "Extractive fallback"


# ---------------------------------------------------------------------
# 10. EVALUATION METRICS
# ---------------------------------------------------------------------

def relevance_labels(indices: Sequence[int], train: pd.DataFrame, test_row: pd.Series) -> List[int]:
    # Proxy labels based on metadata; report them as proxy relevance, not ground truth.
    return [int(row_metadata_match(test_row, train.iloc[idx])) for idx in indices]


def precision_at_k(labels: Sequence[int], k: int) -> float:
    return float(sum(labels[:k]) / max(k, 1))


def reciprocal_rank(labels: Sequence[int]) -> float:
    for rank, label in enumerate(labels, start=1):
        if label:
            return 1.0 / rank
    return 0.0


def recall_at_k(labels: Sequence[int], total_relevant: int) -> float:
    return float(sum(labels) / max(total_relevant, 1))


def lexical_f1(prediction: str, reference: str) -> float:
    pred = set(re.findall(r"\b\w+\b", clean_text(prediction).lower()))
    ref = set(re.findall(r"\b\w+\b", clean_text(reference).lower()))
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    precision, recall = overlap / len(pred), overlap / len(ref)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def response_relevance(query: str, answer: str, vector_model: Any) -> float:
    try:
        if vector_model is not None:
            emb = vector_model.encode(
                [clean_text(query), clean_text(answer)],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return float(np.dot(emb[0], emb[1]))
    except Exception:
        pass
    # Lexical fallback is only a rough proxy.
    q = set(re.findall(r"\b\w+\b", query.lower()))
    a = set(re.findall(r"\b\w+\b", answer.lower()))
    return len(q & a) / max(1, len(q))


def grounding_score(answer: str, contexts: Sequence[str]) -> float:
    answer_words = set(re.findall(r"\b\w+\b", clean_text(answer).lower()))
    context_words = set(re.findall(r"\b\w+\b", " ".join(contexts).lower()))
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def total_relevant_proxy(train: pd.DataFrame, row: pd.Series) -> int:
    count = sum(row_metadata_match(row, train.iloc[i]) for i in range(len(train)))
    return max(1, int(count))


# ---------------------------------------------------------------------
# 11. MODEL EVALUATION AND PERFORMANCE ANALYSIS
# ---------------------------------------------------------------------

def run_experiment(
    train: pd.DataFrame, test: pd.DataFrame, tfidf: TFIDFRetriever,
    vector: VectorRetriever, optimised: OptimisedRetriever,
    max_queries: int = MAX_EVALUATION_QUERIES,
) -> Tuple[Dict[str, List[Dict[str, Any]]], pd.DataFrame]:
    test = test.head(max_queries)
    results: Dict[str, List[Dict[str, Any]]] = {
        "Traditional Retrieval (TF-IDF)": [],
        "Vector Retrieval": [],
        "RAG": [],
        "Optimised Hybrid RAG": [],
    }
    predictions: List[Dict[str, Any]] = []

    for n, (_, row) in enumerate(test.iterrows(), start=1):
        query, reference = row["query"], row["reference_answer"]
        total_rel = total_relevant_proxy(train, row)

        # Traditional retrieval
        start = time.perf_counter()
        idx, scores = tfidf.search(query, TOP_K)
        latency = (time.perf_counter() - start) * 1000
        labels = relevance_labels(idx, train, row)
        results["Traditional Retrieval (TF-IDF)"].append({
            "precision": precision_at_k(labels, TOP_K),
            "recall": recall_at_k(labels, total_rel),
            "mrr": reciprocal_rank(labels),
            "latency_ms": latency,
            "scores": scores,
        })

        # Vector retrieval
        start = time.perf_counter()
        idx, scores = vector.search(query, TOP_K)
        contexts = [train.iloc[i]["reference_answer"] for i in idx]
        answer, backend = generate_rag_answer(query, contexts)
        latency = (time.perf_counter() - start) * 1000
        labels = relevance_labels(idx, train, row)
        results["Vector Retrieval"].append({
            "precision": precision_at_k(labels, TOP_K),
            "recall": recall_at_k(labels, total_rel),
            "mrr": reciprocal_rank(labels),
            "answer_similarity_f1": lexical_f1(answer, reference),
            "response_relevance": response_relevance(query, answer, vector.model),
            "grounding": grounding_score(answer, contexts),
            "latency_ms": latency,
            "scores": scores,
        })

        # Standard RAG uses vector retrieval contexts.
        results["RAG"].append({
            "precision": precision_at_k(labels, TOP_K),
            "recall": recall_at_k(labels, total_rel),
            "mrr": reciprocal_rank(labels),
            "answer_similarity_f1": lexical_f1(answer, reference),
            "response_relevance": response_relevance(query, answer, vector.model),
            "grounding": grounding_score(answer, contexts),
            "latency_ms": latency,
            "scores": scores,
        })

        # Optimised hybrid retrieval + reranking.
        start = time.perf_counter()
        opt_idx, opt_scores = optimised.search(row, OPTIMISED_TOP_K)
        opt_contexts = [train.iloc[i]["reference_answer"] for i in opt_idx]
        opt_answer, opt_backend = generate_rag_answer(query, opt_contexts)
        opt_latency = (time.perf_counter() - start) * 1000
        opt_labels = relevance_labels(opt_idx, train, row)
        results["Optimised Hybrid RAG"].append({
            "precision": precision_at_k(opt_labels, OPTIMISED_TOP_K),
            "recall": recall_at_k(opt_labels, total_rel),
            "mrr": reciprocal_rank(opt_labels),
            "answer_similarity_f1": lexical_f1(opt_answer, reference),
            "response_relevance": response_relevance(query, opt_answer, vector.model),
            "grounding": grounding_score(opt_answer, opt_contexts),
            "latency_ms": opt_latency,
            "scores": opt_scores,
        })

        predictions.append({
            "query": query,
            "reference_answer": reference,
            "industry": row.get("industry", "Unknown"),
            "issue_type": row.get("issue_type", "Unknown"),
            "intent": row.get("primary_intent", "Unknown"),
            "rag_answer": answer,
            "generation_backend": backend,
            "optimised_answer": opt_answer,
            "optimised_generation_backend": opt_backend,
        })

        if n % 25 == 0 or n == len(test):
            print(f"Evaluated {n}/{len(test)} test queries")

    return results, pd.DataFrame(predictions)


# ---------------------------------------------------------------------
# 12. MODEL-LEVEL EDA, COMPARISON AND OPTIMISATION
# ---------------------------------------------------------------------

def model_eda(experiment_results: Dict[str, List[Dict[str, Any]]], vector: VectorRetriever) -> None:
    latency_data = []
    for model_name, records in experiment_results.items():
        scores = [score for record in records for score in record.get("scores", [])]
        if scores:
            plt.figure(figsize=(8, 4.5))
            plt.hist(scores, bins=25)
            plt.title(f"{model_name}: retrieval score distribution")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"EDA_scores_{slugify(model_name)}.png", dpi=180)
            plt.close()
        for record in records:
            latency_data.append({"model": model_name, "latency_ms": record["latency_ms"]})

    latency_df = pd.DataFrame(latency_data)
    if not latency_df.empty:
        plt.figure(figsize=(10, 5))
        latency_df.boxplot(column="latency_ms", by="model", grid=False)
        plt.suptitle("")
        plt.title("Model Latency Comparison")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "EDA_model_latency.png", dpi=180)
        plt.close()

    if vector.embeddings is not None:
        # Normalized embeddings can have identical norms; avoid zero-width histogram bins.
        norms = np.asarray(np.linalg.norm(vector.embeddings, axis=1), dtype=float)
        norms = norms[np.isfinite(norms)]
        if norms.size:
            plt.figure(figsize=(8, 4.5))
            if np.allclose(norms.min(), norms.max()):
                plt.hist(norms, bins=1, edgecolor="black")
                plt.xlim(norms[0] - 0.05, norms[0] + 0.05)
            else:
                bin_count = min(25, max(1, int(np.unique(norms).size)))
                plt.hist(norms, bins=bin_count, edgecolor="black")
            plt.title("Embedding Vector Norms")
            plt.xlabel("Norm")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "EDA_embedding_norms.png", dpi=180)
            plt.close()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")


def compare_models(experiment_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for model_name, records in experiment_results.items():
        frame = pd.DataFrame(records)
        rows.append({
            "Model": model_name,
            "Precision@K": frame["precision"].mean(),
            "Recall@K_proxy": frame["recall"].mean(),
            "MRR_proxy": frame["mrr"].mean(),
            "Answer_Similarity_F1": frame["answer_similarity_f1"].mean()
                if "answer_similarity_f1" in frame else np.nan,
            "Response_Relevance": frame["response_relevance"].mean()
                if "response_relevance" in frame else np.nan,
            "Grounding": frame["grounding"].mean()
                if "grounding" in frame else np.nan,
            "Latency_ms": frame["latency_ms"].mean(),
        })

    comparison = pd.DataFrame(rows)
    higher_is_better = [
        "Precision@K", "Recall@K_proxy", "MRR_proxy",
        "Answer_Similarity_F1", "Response_Relevance", "Grounding",
    ]
    normalised_columns = []
    for metric in higher_is_better:
        if metric not in comparison or comparison[metric].isna().all():
            continue
        low, high = comparison[metric].min(), comparison[metric].max()
        col = f"{metric}_norm"
        comparison[col] = 1.0 if high == low else (comparison[metric] - low) / (high - low)
        normalised_columns.append(col)

    low, high = comparison["Latency_ms"].min(), comparison["Latency_ms"].max()
    comparison["Latency_norm"] = 1.0 if high == low else (high - comparison["Latency_ms"]) / (high - low)

    # Cost is not measured from a live bill here. Latency is used only as a proxy;
    # actual token/API cost should be added when a deployed LLM is used.
    available_weights = {
        "Precision@K_norm": 0.10,
        "Recall@K_proxy_norm": 0.10,
        "MRR_proxy_norm": 0.10,
        "Answer_Similarity_F1_norm": 0.20,
        "Response_Relevance_norm": 0.15,
        "Grounding_norm": 0.20,
        "Latency_norm": 0.15,
    }
    present = [col for col in available_weights if col in comparison.columns]
    total_weight = sum(available_weights[col] for col in present)
    comparison["Composite_Score"] = sum(
        comparison[col].fillna(0) * (available_weights[col] / total_weight)
        for col in present
    )
    comparison = comparison.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    comparison.insert(0, "Rank", range(1, len(comparison) + 1))
    return comparison


def create_comparison_plots(comparison: pd.DataFrame) -> None:
    for metric in [
        "Precision@K", "Recall@K_proxy", "MRR_proxy", "Answer_Similarity_F1",
        "Response_Relevance", "Grounding", "Composite_Score", "Latency_ms",
    ]:
        if metric not in comparison.columns or comparison[metric].isna().all():
            continue
        plt.figure(figsize=(10, 5))
        plt.bar(comparison["Model"], comparison[metric])
        plt.title(f"Model Comparison — {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"Comparison_{slugify(metric)}.png", dpi=180)
        plt.close()


# ---------------------------------------------------------------------
# 13. ETHICAL CONSIDERATIONS AND FINAL ANALYSIS
# ---------------------------------------------------------------------

def ethical_analysis(data_mode: str) -> Dict[str, str]:
    ethics = {
        "Data suitability": (
            "Verify that selected examples are genuinely telecom-related. General customer-support "
            "records must not be reported as telecom evidence."
        ),
        "Bias and coverage": (
            "Report performance by available language, industry, category, priority and channel; "
            "small groups may not support reliable comparisons."
        ),
        "Privacy": "Remove or mask names, email addresses, phone numbers, account IDs and other PII before model/API use.",
        "Transparency": "Keep retrieved context and model/backend metadata for auditability; indicate when evidence is insufficient.",
        "Security": "Store secrets in environment variables; restrict knowledge-base access and test prompt-injection resilience.",
        "Hallucination": "Ground answers in retrieved evidence and route unsupported or high-risk queries to a human.",
        "Evaluation validity": (
            "Precision/recall labels in this prototype are metadata-based proxies, not human-annotated relevance judgements. "
            "Lexical F1 is answer similarity, not factual correctness."
        ),
        "Cost": "Measure actual input/output tokens and Azure charges for a production cost analysis; latency alone is not cost.",
        "Synthetic data": "If the source is synthetic, validate findings on representative real-world or independently curated data before deployment.",
        "Data mode": data_mode,
        "Reproducibility": (
            f"Random seed={RANDOM_STATE}; test size={TEST_SIZE}; embedding={EMBEDDING_MODEL}; "
            f"reranker={RERANKER_MODEL}; top_k={TOP_K}; candidate_k={CANDIDATE_K}."
        ),
    }
    (OUTPUT_DIR / "ethical_analysis.json").write_text(json.dumps(ethics, indent=2), encoding="utf-8")
    return ethics


# ---------------------------------------------------------------------
# 14. MAIN PIPELINE — MIRRORS THE RESEARCH FLOWCHART
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("CUSTOMER SUPPORT RAG EXPERIMENT — TELECOM THESIS PIPELINE")
    print("Data → Retrieval → RAG → Evaluation → Resolution → Responsible AI")
    print("=" * 72)

    # DATA COLLECTION
    print("\n[1/11] DATA COLLECTION")
    raw = load_raw_dataset(DATA_PATH)

    # DATA PRE-PROCESSING / NORMALISATION
    print("\n[2/11] DATA PRE-PROCESSING AND SCHEMA NORMALISATION")
    normalised, schema_type = normalise_dataset(raw)
    print(f"Detected schema type: {schema_type}")
    print(f"Normalised rows: {len(normalised):,}")

    # DATA SELECTION
    print("\n[3/11] DATA SELECTION")
    try:
        selected, data_mode = select_research_data(normalised)
    except ValueError as selection_error:
        # Still generate dataset-level EDA for the downloaded general support dataset.
        # Then stop the RAG experiment rather than mislabelling non-telecom data as telecom.
        print("\nTelecom-only selection could not proceed. Generating general-dataset EDA first.")
        try:
            eda_qa = create_qa_pairs(normalised, schema_type)
        except ValueError:
            eda_qa = pd.DataFrame(columns=["query", "reference_answer"])
        run_dataset_eda(raw, normalised, eda_qa)
        raise selection_error
    print(f"Research data mode: {data_mode}")
    print(f"Selected rows: {len(selected):,}")

    # QA PAIR CREATION / TRAIN-TEST SPLIT
    print("\n[4/11] CUSTOMER QUERY / REFERENCE RESPONSE PREPARATION")
    qa = create_qa_pairs(selected, schema_type)
    train, test = conversation_level_split(qa)
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition; add more dataset examples.")
    print(f"QA pairs: {len(qa):,} | Train: {len(train):,} | Test: {len(test):,}")

    # DATA SELECTION AND EDA
    print("\n[5/11] EXPLORATORY DATA ANALYSIS")
    run_dataset_eda(raw, selected, qa)

    # RETRIEVAL DOCUMENTS
    documents = train["reference_answer"].fillna("").map(clean_text).tolist()
    if not documents or not any(documents):
        raise ValueError("No non-empty reference answers available to index.")

    # VECTOR DATABASE DEPLOYMENT
    print("\n[6/11] VECTOR DATABASE DEPLOYMENT")
    tfidf = TFIDFRetriever()
    tfidf.fit(documents)
    vector = VectorRetriever()
    vector.fit(documents)
    print("Vector backend:", vector.backend)
    print("Vector dimensions:", vector.embeddings.shape[1])

    # RAG MODEL DESIGN AND DEVELOPMENT
    print("\n[7/11] RAG MODEL DESIGN AND DEVELOPMENT")
    optimised = OptimisedRetriever(train, vector, tfidf)
    print("Embedding model configured:", EMBEDDING_MODEL)
    print("Reranker configured:", RERANKER_MODEL)
    print("Azure OpenAI configured:", USE_AZURE_OPENAI)

    # MODEL EVALUATION
    print("\n[8/11] MODEL EVALUATION AND PERFORMANCE ANALYSIS")
    experiment_results, predictions = run_experiment(
        train, test, tfidf, vector, optimised, max_queries=MAX_EVALUATION_QUERIES
    )

    # COMPARE / OPTIMISE
    print("\n[9/11] COMPARE AND OPTIMISE PERFORMANCE")
    model_eda(experiment_results, vector)
    comparison = compare_models(experiment_results)
    print(comparison.to_string(index=False))
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)
    create_comparison_plots(comparison)

    # CUSTOMER QUERY RESOLUTION / RESPONSE GENERATION
    print("\n[10/11] CUSTOMER QUERY RESOLUTION AND RESPONSE GENERATION")
    if not predictions.empty:
        sample = predictions.iloc[0]
        print("\nSample customer query:")
        print(sample["query"])
        print("\nGenerated response:")
        print(sample["rag_answer"])
        print("\nGeneration backend:", sample["generation_backend"])

    # ETHICAL ANALYSIS
    print("\n[11/11] ETHICAL CONSIDERATIONS AND FINAL ANALYSIS")
    ethics = ethical_analysis(data_mode)
    print(json.dumps(ethics, indent=2))

    run_summary = {
        "dataset_path": str(Path(DATA_PATH).resolve()),
        "schema_type": schema_type,
        "data_mode": data_mode,
        "raw_rows": int(len(raw)),
        "selected_rows": int(len(selected)),
        "qa_pairs": int(len(qa)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "vector_backend": vector.backend,
        "azure_openai_enabled": USE_AZURE_OPENAI,
        "best_model_by_composite_score": str(comparison.iloc[0]["Model"]),
        "best_composite_score": float(comparison.iloc[0]["Composite_Score"]),
        "warning": (
            "Proxy metrics are experimental indicators, not a substitute for human evaluation. "
            "Confirm telecom relevance before using results in the thesis."
        ),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETED")
    print("Best model:", run_summary["best_model_by_composite_score"])
    print("Results directory:", OUTPUT_DIR.resolve())
    print("=" * 72)


if __name__ == "__main__":
    main()
