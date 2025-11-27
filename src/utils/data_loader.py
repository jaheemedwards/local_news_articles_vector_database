import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_news_data():
    df = pd.read_parquet("dataset/news_last_1_year_clusters_umap.parquet")
    df["embedding"] = df["embedding"].apply(np.array)
    return df
