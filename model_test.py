import os
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from datasets import Dataset
from sklearn.metrics import confusion_matrix
import torch
import re
from io import BytesIO
import base64
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(__file__)
hf_model_path = os.path.join(BASE_DIR, "saved_model", "xlm-roberta-sentiment")
prophet_model_path = os.path.join(BASE_DIR, "prophet_model.pkl")
csv_path = os.path.join(BASE_DIR, "test_set.csv")

# Load models
tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)

with open(prophet_model_path, "rb") as f:
    prophet_model = pickle.load(f)

df = pd.read_csv(csv_path)


# Streamlit app configuration
st.set_page_config(
    page_title="Sentiment Analysis & Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

# Custom CSS for enhanced UI/UX
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stSidebar {background-color: #e9ecef;}
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #ced4da;
    }
    .stSelectbox, .stSlider, .stRadio, .stMultiselect {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stMarkdown h1, h2, h3 {
        color: #343a40;
        font-family: 'Arial', sans-serif;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .sidebar .sidebar-content {
        padding: 20px;
    }
    .css-1lcbmhc {padding: 1rem 2rem;}
    .sentiment-positive {color: #28a745; font-weight: bold;}
    .sentiment-neutral {color: #ffc107; font-weight: bold;}
    .sentiment-negative {color: #dc3545; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title(" Sentiment Analysis & Forecasting Dashboard")
st.markdown("""
Welcome to an interactive dashboard powered by XLM-RoBERTa for sentiment classification and Prophet for forecasting trends in English and Swahili messages. 
Analyze sentiments, predict future trends, and explore insightful visualizations.
""", unsafe_allow_html=True)

# Load models
@st.cache_resource(show_spinner="Loading XLM-RoBERTa model...")
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("saved_model/xlm-roberta-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("saved_model/xlm-roberta-sentiment")
        return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        st.error(f"Error loading XLM-RoBERTa model: {e}")
        return None

@st.cache_resource(show_spinner="Loading Prophet model...")
def load_prophet_model():
    try:
        with open("prophet_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Prophet model not found. Ensure 'prophet_model.pkl' is in the directory.")
        return None

classifier = load_sentiment_model()
prophet_model = load_prophet_model()
if classifier is None or prophet_model is None:
    st.stop()

# Load and prepare data
@st.cache_data(show_spinner="Loading datasets...")
def load_data():
    try:
        english_df = pd.read_csv("roberta_labeled_dataset.csv")
        swahili_df = pd.read_csv("swahili_labeled.csv")
        english_df = english_df[["time", "date", "message", "sentiment"]].drop_duplicates().dropna(subset=["message"])
        swahili_df = swahili_df[["time", "date", "translated_message", "sentiment"]].rename(columns={"translated_message": "message"}).drop_duplicates().dropna(subset=["message"])
        combined_df = pd.concat([english_df, swahili_df[["time", "date", "message", "sentiment"]]])
        combined_df["datetime"] = pd.to_datetime(combined_df["date"] + " " + combined_df["time"], errors="coerce", utc=True).dt.tz_convert("Africa/Nairobi")
        combined_df = combined_df.dropna(subset=["datetime"])
        combined_df["week"] = combined_df["datetime"].dt.to_period("W").apply(lambda x: x.start_time)
        combined_df["label"] = combined_df["sentiment"].map({"positive": 2, "neutral": 1, "negative": 0})
        return english_df, swahili_df, combined_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

english_df, swahili_df, combined_df = load_data()
if combined_df is None:
    st.stop()

# Prepare time-series data
@st.cache_data(show_spinner="Preparing time-series data...")
def prepare_time_series_data(df):
    time_series_df = df.groupby(["week", "sentiment"]).size().unstack().fillna(0)
    time_series_df["total"] = time_series_df.sum(axis=1)
    time_series_df["positive_pct"] = time_series_df["positive"] / time_series_df["total"].replace(0, np.nan)
    time_series_df["neutral_pct"] = time_series_df["neutral"] / time_series_df["total"].replace(0, np.nan)
    time_series_df["negative_pct"] = time_series_df["negative"] / time_series_df["total"].replace(0, np.nan)
    time_series_df = time_series_df.reset_index()
    time_series_df["datetime"] = time_series_df["week"]
    return time_series_df

time_series_df = prepare_time_series_data(combined_df)

# Load test set for confusion matrix
@st.cache_data(show_spinner="Loading test set...")
def load_test_set():
    try:
        test_df = pd.read_csv("test_set.csv")
        if "message" not in test_df.columns or "label" not in test_df.columns:
            st.error("test_set.csv must contain 'message' and 'label' columns.")
            return None
        return test_df
    except FileNotFoundError:
        st.error("test_set.csv not found. Generate it using the training split script.")
        return None
    except Exception as e:
        st.error(f"Error loading test_set.csv: {e}")
        return None

test_df = load_test_set()

st.sidebar.title("Navigation")
st.sidebar.markdown("Select a section to explore:")
section = st.sidebar.radio(
    "",
    ["Sentiment Prediction", "Sentiment Forecast", "Visualizations"],
    format_func=lambda x: f"📋 {x}",
    label_visibility="collapsed"
)

# Censor sensitive information
def censor_sensitive_info(text):
    text = re.sub(r'\b(254|\+254|0)?7\d{8}\b', "[PHONE NUMBER]", text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', "[EMAIL]", text)
    text = re.sub(r'\b[A-Za-z0-9]{2,}-\d{2}-\d{3,}/\d{4}\b', "[ADMISSION NO.]", text)
    return text

# Sentiment Prediction Section
if section == "Sentiment Prediction":
    st.header(" Predict Sentiment for Messages")
    with st.form("sentiment_form"):
        st.markdown("**Enter one or more messages (English or translated Swahili)**")
        user_input = st.text_area(
            "Messages (one per line for batch prediction):",
            placeholder="e.g., Great product!\nI love this service!",
            height=150
        )
        language_filter = st.selectbox("Filter Sample Messages:", ["All", "English", "Swahili"], help="Show sample messages from selected dataset.")
        submitted = st.form_submit_button("Predict Sentiment", use_container_width=True)
    
    if submitted and user_input:
        messages = [msg.strip() for msg in user_input.split("\n") if msg.strip()]
        if messages:
            with st.spinner("Predicting sentiments..."):
                results = classifier(messages, return_all_scores=True)
                st.subheader("Prediction Results")
                for msg, result in zip(messages, results):
                    pred_label = np.argmax([score["score"] for score in result])
                    label_map = {0: "negative", 1: "neutral", 2: "positive"}
                    sentiment = label_map[pred_label]
                    confidence = result[pred_label]["score"]
                    st.markdown(
                        f"""
                        **Message**: {censor_sensitive_info(msg)}  
                        **Sentiment**: <span class="sentiment-{sentiment}">{sentiment.capitalize()}</span> (Confidence: {confidence:.3f})
                        """,
                        unsafe_allow_html=True
                    )
                # Download predictions
                pred_df = pd.DataFrame({
                    "Message": messages,
                    "Sentiment": [label_map[np.argmax([score["score"] for score in res])] for res in results],
                    "Confidence": [res[np.argmax([score["score"] for score in res])]["score"] for res in results]
                })
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name=f"sentiment_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.error("Please enter at least one message.")
    
    st.subheader("Sample Messages with Similar Sentiment")
    sentiment_filter = st.selectbox("Select Sentiment:", ["positive", "neutral", "negative"], key="sample_sentiment")
    sample_size = st.slider("Number of Samples:", 1, 10, 3, help="Choose how many sample messages to display.")
    if language_filter == "English":
        sample_df = english_df[english_df["sentiment"] == sentiment_filter]
    elif language_filter == "Swahili":
        sample_df = swahili_df[swahili_df["sentiment"] == sentiment_filter]
    else:
        sample_df = combined_df[combined_df["sentiment"] == sentiment_filter]
    if not sample_df.empty:
        samples = sample_df["message"].sample(min(sample_size, len(sample_df))).tolist()
        for msg in samples:
            st.markdown(f"- {censor_sensitive_info(msg)}", unsafe_allow_html=True)
    else:
        st.warning(f"No {sentiment_filter} messages found in {language_filter} dataset.")

# Sentiment Forecast Section
if section == "Sentiment Forecast":
    st.header("📈 Positive Sentiment Trend Forecast")
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            periods = st.slider(
                "Forecast Period (Weeks):",
                1, 24, 4,
                help="Select how many weeks to forecast positive sentiment trends.",
                key="forecast_periods"
            )
        with col2:
            granularity = st.selectbox(
                "Time Granularity:",
                ["Weekly", "Daily"],
                help="Choose the time aggregation for forecasting."
            )
        with st.spinner("Generating forecast..."):
            freq = "W" if granularity == "Weekly" else "D"
            future = prophet_model.make_future_dataframe(periods=periods, freq=freq)
            forecast = prophet_model.predict(future)
        st.subheader(f"Positive Sentiment Forecast (Next {periods} {granularity})")
        fig = prophet_model.plot(forecast)
        plt.title(f"Positive Sentiment Forecast ({granularity})")
        plt.xlabel("Date")
        plt.ylabel("Positive Sentiment (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)
        st.subheader("Forecasted Values")
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).style.format({
                "ds": lambda x: x.strftime("%Y-%m-%d"),
                "yhat": "{:.2%}",
                "yhat_lower": "{:.2%}",
                "yhat_upper": "{:.2%}"
            }),
            use_container_width=True
        )
        # Download forecast
        csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).to_csv(index=False)
        st.download_button(
            label="Download Forecast",
            data=csv,
            file_name=f"forecast_{granularity.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Visualizations Section
if section == "Visualizations":
    st.header("📉 Data Visualizations")
    viz_option = st.selectbox(
        "Select Visualization:",
        ["Sentiment Distribution", "Temporal Trends", "Word Clouds", "Confusion Matrix"],
        help="Explore dataset insights or model performance."
    )

    if viz_option == "Sentiment Distribution":
        st.subheader("Sentiment Distribution")
        dataset_choice = st.radio("Select Dataset:", ["Combined", "English", "Swahili"], horizontal=True)
        fig = px.histogram(
            combined_df if dataset_choice == "Combined" else english_df if dataset_choice == "English" else swahili_df,
            x="sentiment",
            category_orders={"sentiment": ["positive", "neutral", "negative"]},
            color="sentiment",
            color_discrete_map={"positive": "#28a745", "neutral": "#ffc107", "negative": "#dc3545"},
            title=f"{dataset_choice} Sentiment Distribution"
        )
        fig.update_layout(xaxis_title="Sentiment", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        # Download distribution data
        dist_df = (combined_df if dataset_choice == "Combined" else english_df if dataset_choice == "English" else swahili_df)[["sentiment"]].value_counts().reset_index(name="count")
        csv = dist_df.to_csv(index=False)
        st.download_button(
            label="Download Distribution Data",
            data=csv,
            file_name=f"sentiment_distribution_{dataset_choice.lower()}.csv",
            mime="text/csv"
        )

    if viz_option == "Temporal Trends":
        st.subheader("Sentiment Trends Over Time")
        sentiment_types = st.multiselect(
            "Select Sentiments:",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
            help="Choose sentiments to plot over time."
        )
        granularity = st.radio("Time Granularity:", ["Weekly", "Daily"], horizontal=True)
        if sentiment_types:
            columns = [f"{s.lower()}_pct" for s in sentiment_types]
            if granularity == "Daily":
                temp_df = combined_df.copy()
                temp_df["day"] = temp_df["datetime"].dt.to_period("D").apply(lambda x: x.start_time)
                time_series_df = temp_df.groupby(["day", "sentiment"]).size().unstack().fillna(0)
                time_series_df["total"] = time_series_df.sum(axis=1)
                time_series_df["positive_pct"] = time_series_df["positive"] / time_series_df["total"].replace(0, np.nan)
                time_series_df["neutral_pct"] = time_series_df["neutral"] / time_series_df["total"].replace(0, np.nan)
                time_series_df["negative_pct"] = time_series_df["negative"] / time_series_df["total"].replace(0, np.nan)
                time_series_df = time_series_df.reset_index()
                time_series_df["datetime"] = time_series_df["day"]
            fig = px.line(
                time_series_df,
                x="datetime",
                y=columns,
                title=f"Sentiment Trends ({granularity})",
                color_discrete_map={"positive_pct": "#28a745", "neutral_pct": "#ffc107", "negative_pct": "#dc3545"}
            )
            fig.update_yaxes(tickformat=".0%", title="Percentage")
            fig.update_xaxes(title="Date")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Please select at least one sentiment.")

    if viz_option == "Word Clouds":
        st.subheader("Word Clouds by Sentiment")
        sentiment = st.selectbox("Select Sentiment:", ["positive", "neutral", "negative"])
        dataset_choice = st.radio("Select Dataset:", ["Combined", "English", "Swahili"], horizontal=True)
        data_df = combined_df if dataset_choice == "Combined" else english_df if dataset_choice == "English" else swahili_df
        text = " ".join(data_df[data_df["sentiment"] == sentiment]["message"])
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"{sentiment.capitalize()} Word Cloud ({dataset_choice})")
            st.pyplot(fig)
            # Download word cloud image
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            st.download_button(
                label="Download Word Cloud",
                data=buf.getvalue(),
                file_name=f"wordcloud_{sentiment}_{dataset_choice.lower()}.png",
                mime="image/png"
            )
        else:
            st.error(f"No {sentiment} messages found in {dataset_choice} dataset.")

    if viz_option == "Confusion Matrix":
        st.subheader("Confusion Matrix (Test Set)")
        dataset_choice = st.radio("Select Dataset for Confusion Matrix:", ["Combined", "English", "Swahili"], horizontal=True)
        if test_df is not None:
            test_subset = test_df if dataset_choice == "Combined" else test_df[test_df["message"].isin(english_df["message"])] if dataset_choice == "English" else test_df[test_df["message"].isin(swahili_df["message"])]
            if not test_subset.empty:
                with st.spinner("Computing confusion matrix..."):
                    def tokenize_function(examples):
                        return classifier.tokenizer(examples["message"], padding="max_length", truncation=True, max_length=128)
                    test_dataset = Dataset.from_pandas(test_subset[["message", "label"]])
                    test_dataset = test_dataset.map(tokenize_function, batched=True)
                    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
                    predictions = classifier(test_subset["message"], return_all_scores=True)
                    pred_labels = [np.argmax([score["score"] for score in pred]) for pred in predictions]
                    cm = confusion_matrix(test_subset["label"], pred_labels, labels=[0, 1, 2])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Negative", "Neutral", "Positive"],
                        yticklabels=["Negative", "Neutral", "Positive"],
                        ax=ax
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_title(f"Confusion Matrix ({dataset_choice})")
                    st.pyplot(fig)
                    st.markdown(
                        "**Interpretation**: The matrix shows true vs. predicted sentiment labels on the test set (~10% of data). "
                        "Diagonal values indicate correct predictions. Higher values on the diagonal suggest better model performance.",
                        unsafe_allow_html=True
                    )
                    # Display accuracy
                    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
                    st.write(f"**Accuracy**: {accuracy:.2%}")
            else:
                st.error(f"No test data available for {dataset_choice} dataset.")
        else:
            st.error("Cannot display confusion matrix: test_set.csv is missing or invalid. Run the generate_test_set.py script to create it.")

# Footer
st.sidebar.markdown("---")




