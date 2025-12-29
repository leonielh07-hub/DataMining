import streamlit as st
import pandas as pd
from transformers import pipeline
import datetime
import altair as alt # Für das Diagramm mit Tooltips
import matplotlib.pyplot as plt # Für die WordCloud Anzeige
from wordcloud import WordCloud # Für die WordCloud Generierung

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Review AI Dashboard", layout="wide")

st.title("📊 E-Commerce Sentiment Analysis Dashboard")
st.markdown("Analysis of product reviews using Hugging Face Transformers.")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load the data.csv created by the scraper
        df = pd.read_csv("data.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# Check: If no data exists, show warning
if df.empty:
    st.error("⚠️ File 'data.csv' not found or empty. Please run 'scraper_selenium.py' first!")
    st.stop()

# --- 3. LOAD AI MODEL ---
@st.cache_resource
def load_sentiment_model():
    # Loads the AI model (cached after first run)
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
options = ["Products", "Testimonials", "Reviews"]
selection = st.sidebar.radio("Go to:", options)

# --- 5. MAIN AREA ---
if selection == "Products":
    st.header("🛍️ Products")
    products_df = df[df["type"] == "product"].copy() 
    
    if not products_df.empty:
        st.info(f"{len(products_df)} products found.")
        
        # --- DATA PREPERATION ---
        # 1. soliit the string at the beginning " - " . n=1 says where it is seperated. 
        # create a df with 2 columns (0 und 1)
        split_data = products_df["content"].str.split(" - ", n=1, expand=True)
        
        # second column assigned 
        # column 0 is the name 
        products_df["Product Name"] = split_data[0].str.replace("Product: ", "")
        
        # Column 1 is the price (change text into numbers)
        products_df["Price"] = pd.to_numeric(split_data[1], errors='coerce')
        
        # --- SHOW ---
        st.dataframe(
            products_df[["Product Name", "Price"]], 
            width="stretch",
            hide_index=True, 
            column_config={
                "Price": st.column_config.NumberColumn(
                    "Price ($)",
                    format="$%.2f" # formating as $ with two decimals
                )
            }
        )
    else:
        st.warning("No products found in data.csv.")

elif selection == "Testimonials":
    st.header("⭐ Customer Testimonials")
    testimonials_df = df[df["type"] == "testimonial"]
    
    if not testimonials_df.empty:
        st.info(f"{len(testimonials_df)} testimonials found.")
        # UPDATE: use_container_width -> width='stretch'
        st.dataframe(testimonials_df[["content", "rating"]], width="stretch")
    else:
        st.warning("No testimonials found in data.csv.")

elif selection == "Reviews":
    st.header("📝 Review Analysis (AI Powered)")
    
    reviews_df = df[df["type"] == "review"].copy()
    
    if reviews_df.empty:
        st.warning("No reviews found.")
    else:
        # --- A. FILTERS (Year & Month) ---
        col1, col2 = st.columns(2)
        
        with col1:
            # Dynamically get available years from data
            available_years = sorted(reviews_df["date"].dt.year.unique(), reverse=True)
            if not available_years:
                available_years = [2023] # Fallback
            selected_year = st.selectbox("Select Year:", options=available_years)

        with col2:
            months = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
            selected_month = st.select_slider("Select Month:", options=months)
            month_index = months.index(selected_month) + 1
        
        # Apply Filters
        filtered_reviews = reviews_df[
            (reviews_df["date"].dt.year == selected_year) & 
            (reviews_df["date"].dt.month == month_index)
        ].copy() # Copy to avoid SettingWithCopy warnings
        
        st.subheader(f"Reviews in {selected_month} {selected_year}")
        
        if filtered_reviews.empty:
            st.info("No reviews found for this time period.")
        else:
            st.write(f"Count: {len(filtered_reviews)}")
            
            # --- Preview of REVIEWS (before clicking on run analysis) ---
            st.markdown("### Preview of Reviews")
            st.dataframe(filtered_reviews[["date", "content"]], width="stretch")

            # --- B. AI ANALYSIS BUTTON ---
            if st.button("Start Sentiment Analysis 🚀"):
                with st.spinner("AI is analyzing..."):
                    # Apply AI
                    texts = filtered_reviews["content"].tolist()
                    # Truncate to 512 chars to prevent model errors
                    short_texts = [t[:512] for t in texts]
                    
                    results = sentiment_pipeline(short_texts)
                    
                    # Store results in DataFrame
                    filtered_reviews["sentiment"] = [r["label"] for r in results]
                    filtered_reviews["confidence"] = [r["score"] for r in results]
                    
                    # --- C. VISUALIZATION ---
                    
                    # 1. Bar Chart with Average Confidence Tooltip (using Altair)
                    st.markdown("### Sentiment Distribution")
                    
                    # We use Altair to calculate the mean confidence per sentiment for the tooltip
                    chart = alt.Chart(filtered_reviews).mark_bar().encode(
                        x=alt.X('sentiment', axis=alt.Axis(title='Sentiment')),
                        y=alt.Y('count()', axis=alt.Axis(title='Number of Reviews')),
                        color=alt.Color('sentiment', scale=alt.Scale(domain=['POSITIVE', 'NEGATIVE'], range=['#28a745', '#dc3545'])),
                        tooltip=[
                            'sentiment', 
                            'count()', 
                            alt.Tooltip('mean(confidence)', format='.2%', title='Avg. Confidence')
                        ]
                    ).properties(
                        title="Count by Sentiment with Avg Confidence Score"
                    )
                    # Hinweis: st.altair_chart nutzt oft noch use_container_width, 
                    # falls hier noch eine Warnung kommt, kann man es testweise entfernen.
                    st.altair_chart(chart, use_container_width=True)

                    # 2. Word Cloud
                    st.markdown("### Word Cloud")
                    try:
                        # Join all review texts into one big string
                        all_text = " ".join(filtered_reviews["content"])
                        if all_text:
                            # Create WordCloud object
                            wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(all_text)
                            
                            # Display using Matplotlib
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off") # Hide axes
                            st.pyplot(fig)
                        else:
                            st.warning("Not enough text for a word cloud.")
                    except Exception as e:
                        st.error(f"Could not generate Word Cloud: {e}")

                    # 3. Detailed Data Table
                    st.markdown("### Details")
                    # UPDATE: use_container_width -> width='stretch'
                    st.dataframe(
                        filtered_reviews[["date", "content", "sentiment", "confidence"]],
                        column_config={
                            "confidence": st.column_config.ProgressColumn(
                                "Confidence (AI)",
                                format="%.2f",
                                min_value=0,
                                max_value=1
                            )
                        },
                        width="stretch"
                    )