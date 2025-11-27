import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def show_similar_stories(df, top_k=7, preview_chars=300):
    """
    Display similar news stories based on cosine similarity of embeddings.

    Parameters:
    - df: DataFrame containing news articles with 'embedding'
    - top_k: number of similar stories to show
    - preview_chars: number of characters to display from body
    """

    st.title("🔍 Similar Stories Explorer")

    # --- Slider to pick article ---
    idx = st.slider("Select Article", 0, len(df) - 1, 0)
    selected = df.iloc[idx]

    st.subheader("📌 Selected Story")
    st.write(f"**Title:** {selected['title']}")
    st.write(f"**Author:** {selected['author']}")
    st.write(selected["body"][:preview_chars] + "...")

    # --- Similarity computation ---
    target = selected["embedding"].reshape(1, -1)
    all_embs = np.vstack(df["embedding"].to_numpy())

    sims = cosine_similarity(target, all_embs)[0]
    top_idx = np.argsort(sims)[::-1][1:top_k+1]  # skip the selected article itself

    similar_df = df.iloc[top_idx].copy()
    similar_df["similarity"] = sims[top_idx]

    st.subheader(f"📰 Top {top_k} Similar Stories")
    for _, row in similar_df.iterrows():
        st.markdown("### " + row["title"])
        st.write(f"Similarity: {row['similarity']:.4f}")
        st.write(row["body"][:preview_chars] + "...")
        st.markdown("---")

