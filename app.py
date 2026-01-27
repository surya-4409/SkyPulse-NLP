import streamlit as st
import pandas as pd
import json
import os
import streamlit.components.v1 as components
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="SkyPulse NLP Dashboard",
    page_icon="✈️",
    layout="wide"
)

# Title and Header
st.title("✈️ SkyPulse: Airline Sentiment & Topic Analysis")
st.markdown("""
This dashboard analyzes public sentiment towards US Airlines using **Natural Language Processing**.
It showcases **Sentiment Classification** (Positive/Negative/Neutral) and **Topic Modeling** (LDA) to understand customer pain points.
""")

# --- Sidebar for Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to:", ["Dataset Overview", "Sentiment Analysis", "Topic Modeling"])

# --- Load Data Helper ---
@st.cache_data
def load_data():
    data_path = os.path.join('output', 'preprocessed_data.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def load_metrics():
    path = os.path.join('output', 'sentiment_metrics.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_topics():
    path = os.path.join('output', 'topics.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

df = load_data()

# --- Section 1: Dataset Overview ---
if options == "Dataset Overview":
    st.header("📊 Dataset Overview")
    if df is not None:
        st.write(f"**Total Tweets:** {len(df)}")
        st.write("### Raw Data Sample")
        st.dataframe(df[['tweet_id', 'airline_sentiment', 'cleaned_text']].head(10))
        
        st.write("### Sentiment Distribution (Raw Data)")
        fig = px.pie(df, names='airline_sentiment', title='Distribution of Sentiment in Training Data',
                     color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'gray'})
        st.plotly_chart(fig)
    else:
        st.error("Data not found. Please run the preprocessing step.")

# --- Section 2: Sentiment Analysis ---
elif options == "Sentiment Analysis":
    st.header("🤖 Sentiment Analysis Model")
    
    # Display Metrics
    metrics = load_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision_macro']:.2%}")
        col3.metric("Recall", f"{metrics['recall_macro']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1_score_macro']:.2%}")
    else:
        st.warning("Metrics file not found. Run the sentiment model training first.")
    
    st.markdown("---")
    
    # Display Predictions (if available)
    pred_path = os.path.join('output', 'sentiment_predictions.csv')
    if os.path.exists(pred_path):
        st.subheader("Model Predictions on Test Set")
        pred_df = pd.read_csv(pred_path)
        
        # Merge with original text for context (optional, based on tweet_id)
        if df is not None:
            # We convert tweet_id to string to ensure matching works
            pred_df['tweet_id'] = pred_df['tweet_id'].astype(str)
            df['tweet_id'] = df['tweet_id'].astype(str)
            display_df = pd.merge(pred_df, df[['tweet_id', 'cleaned_text']], on='tweet_id', how='left')
        else:
            display_df = pred_df
            
        st.dataframe(display_df.head(20))
        
        st.subheader("Predicted Sentiment Distribution")
        fig_pred = px.bar(pred_df, x='predicted_sentiment', color='predicted_sentiment', 
                          title="Count of Predictions by Class")
        st.plotly_chart(fig_pred)

# --- Section 3: Topic Modeling ---
elif options == "Topic Modeling":
    st.header("🔍 Topic Modeling (LDA)")
    st.markdown("Discovering hidden themes in the tweets.")
    
    # Display Topics
    topics = load_topics()
    if topics:
        st.subheader("Identified Topics & Top Words")
        cols = st.columns(len(topics))
        for idx, (topic, words) in enumerate(topics.items()):
            with cols[idx % 3]: # Display in grid of 3
                st.info(f"**{topic.upper()}**")
                st.write(", ".join(words))
    
    st.markdown("---")
    
    # Display Interactive Visualization
    st.subheader("Interactive LDA Visualization")
    html_path = os.path.join('output', 'lda_visualization.html')
    
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html_string = f.read()
        components.html(html_string, width=1300, height=800, scrolling=True)
    else:
        st.error("LDA Visualization HTML not found. Run the topic modeling step first.")