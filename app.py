import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# Title
st.title("🧠 Sentiment Visualizer - Product Reviews")

# Upload CSV
st.subheader("📁 Upload Review Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Sentiment Analysis
    st.subheader("🔍 Analyzing Sentiments...")
    df['Polarity'] = df['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment_Label'] = df['Polarity'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral'))

    # Show Data
    st.write(df.head())

    # Sentiment Pie Chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment_Label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Overall Sentiment")
    st.plotly_chart(fig)

    # Sentiment by Product
    st.subheader("🛍️ Sentiment by Product")
    prod_sentiment = df.groupby(['Product', 'Sentiment_Label']).size().reset_index(name='Count')
    fig2 = px.bar(prod_sentiment, x='Product', y='Count', color='Sentiment_Label', barmode='group')
    st.plotly_chart(fig2)

    # Optional: Word Cloud or Filter
    st.success("✅ Analysis Complete!")

else:
    st.info("👈 Upload your `sentiment_data.csv` file to get started.")
