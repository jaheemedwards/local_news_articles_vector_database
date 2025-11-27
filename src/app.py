import streamlit as st
import json
import numpy as np
from utils.data_loader import load_news_data
from views.cluster_topics import show_cluster_topics
from views.similar_stories import show_similar_stories
from views.cluster_trends import show_cluster_trends

st.set_page_config(
    page_title="News Analytics App",
    layout="wide",
)

# ------------------------
# 1. Load news data once
# ------------------------
if "news_df" not in st.session_state:
    df = load_news_data()
    st.session_state["news_df"] = df
else:
    df = st.session_state["news_df"]

CLUSTER_INTERPRETATIONS = {
    0: "Government Affairs, Ministries & National Policy",
    1: "Courts, Judiciary & Legal Proceedings",
    2: "Crime Policy, National Security & Emergency Issues",
    3: "Police Operations, Raids, Arrests & Seizures",
    4: "Murders, Homicides & Fatal Incidents",
    5: "Social Issues, Education, Community & Human Interest",
    6: "Elections, Political Parties & Campaigns",
    7: "Weather, Flooding, Disaster Response & Infrastructure",
    8: "Energy Sector, Economy, Carnival & Regional Affairs",
}

# Extract UMAP embeddings
st.session_state['embedding_2d'] = df[['umap_x','umap_y']].to_numpy()
st.session_state['embedding_3d'] = df[['umap_3d_x','umap_3d_y','umap_3d_z']].to_numpy()

# ------------------------
# 2. Load cluster keywords
# ------------------------
if "cluster_keywords" not in st.session_state:
    with open("dataset/cluster_keywords_100.json", "r") as f:
        st.session_state["cluster_keywords"] = json.load(f)

st.title("📰 Trinidad & Tobago News Analytics Dashboard")
st.write("Welcome! Use the tabs below to explore clusters, similar stories, and trends.")

# ------------------------
# Tabs at the top
# ------------------------
tabs = st.tabs(["Home", "Cluster Topics", "Similar Stories", "Cluster Trends"])

# --- Home tab ---
with tabs[0]:
    st.markdown("### Overview / Methodology")

    # --- Overview / Methodology ---
    st.markdown(
        """
        Welcome to the interactive news analysis dashboard! This app allows you to explore
        news articles from Trinidad & Tobago using semantic embeddings, clustering, and
        trend analysis.

        **Pipeline Overview:**
        1. **Data Collection:** Newsday articles from 2018–2025 were scraped and stored locally.
        2. **Cleaning:** Articles were cleaned, dates standardized, and the most recent year was
           selected for efficient embedding.
        3. **Embedding:** Each article was converted into a high-dimensional vector using the
           `nomic-embed-text` model via Ollama, capturing semantic meaning of titles and bodies.
        4. **Clustering & Dimensionality Reduction:** KMeans identifies clusters representing
           topics like politics, crime, social issues, and economy. 2D & 3D UMAP projections
           visualize these clusters.
        5. **Exploration & Trends:** Users can explore clusters, see keyword wordclouds, example
           articles, and track daily/weekly trends of each topic over time.
        """
    )

    # Create 3 columns: left (spacer), middle (image), right (spacer)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/news_articles_picture.png", width='stretch')


# --- Cluster Topics tab ---
with tabs[1]:
    show_cluster_topics(df,
                        st.session_state['embedding_2d'],
                        st.session_state['embedding_3d'],
                        st.session_state['cluster_keywords'],
                        cluster_interpretations=CLUSTER_INTERPRETATIONS)

# --- Similar Stories tab ---
with tabs[2]:
    show_similar_stories(df, top_k=7)

# --- Cluster Trends tab ---
with tabs[3]:
    show_cluster_trends(df, CLUSTER_INTERPRETATIONS)


