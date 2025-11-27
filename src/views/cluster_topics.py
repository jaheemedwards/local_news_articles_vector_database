import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import numpy as np
from scipy.stats import zscore

def show_cluster_topics(
    df,
    embedding_2d,
    embedding_3d,
    cluster_keywords,
    cluster_interpretations,
    top_n_words=10,
    num_examples=3
):
    """
    Display 2D/3D UMAP cluster visualizations, word clouds, and example articles.
    
    Parameters:
    - df: DataFrame with news articles
    - embedding_2d: np.array of 2D UMAP coordinates
    - embedding_3d: np.array of 3D UMAP coordinates
    - cluster_keywords: dict of cluster -> keywords list
    - cluster_interpretations: dict of cluster -> descriptive labels
    - top_n_words: number of top words to display in wordclouds
    - num_examples: number of example articles per cluster
    """

    # ===============================
    # Map clusters to descriptive labels
    # ===============================
    df['cluster_label'] = df['cluster'].map(cluster_interpretations)

    # ===============================
    # 1. 2D UMAP
    # ===============================
    st.subheader("2D UMAP of Clusters")
    st.caption("Each colour represents a cluster. Hover to explore titles, authors, and cluster theme.")

    fig_2d = px.scatter(
        df,
        x='umap_x',
        y='umap_y',
        color='cluster_label',  # descriptive labels
        # hover_data=['title', 'author', 'cluster_label'],  # only show these
        title="2D UMAP of News Article Clusters"
    )
    fig_2d.update_traces(
        hovertemplate="<b>Cluster:</b> %{customdata[0]}<br><b>Title:</b> %{customdata[1]}<br><b>Author:</b> %{customdata[2]}<extra></extra>",
        customdata=df[['cluster_label', 'title', 'author']].values
    )

    st.plotly_chart(fig_2d)

    # ===============================
    # 2. 3D UMAP (cleaned)
    # ===============================
    st.subheader("3D UMAP of Clusters (Outliers removed)")
    st.caption("A 3D view of the clusters with extreme outliers filtered using z-score.")

    z_scores = np.abs(zscore(embedding_3d, axis=0))
    mask = (z_scores < 3).all(axis=1)
    df_3d_clean = df[mask]

    fig_3d = px.scatter_3d(
        df_3d_clean,
        x='umap_3d_x',
        y='umap_3d_y',
        z='umap_3d_z',
        color='cluster_label',
        # hover_data=['title', 'author', 'cluster_label'],  # only show these
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, zeroline=False, showbackground=False),
            yaxis=dict(showgrid=True, zeroline=False, showbackground=False),
            zaxis=dict(showgrid=True, zeroline=False, showbackground=False),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene_camera=dict(eye=dict(x=1.0, y=1.0, z=1.2))
    )

    fig_3d.update_traces(marker=dict(size=4, opacity=0.7))

    fig_3d.update_traces(
        hovertemplate="<b>Cluster:</b> %{customdata[0]}<br><b>Title:</b> %{customdata[1]}<br><b>Author:</b> %{customdata[2]}<extra></extra>",
        customdata=df_3d_clean[['cluster_label', 'title', 'author']].values
    )

    st.plotly_chart(fig_3d, width='stretch')

    # ===============================
    # 3. Word Clouds WITH interpretations
    # ===============================
    st.subheader("Cluster Themes & Keyword Clouds")

    for cluster_str, keywords_list in cluster_keywords.items():
        cluster = int(cluster_str)
        col1, col2 = st.columns([1, 1])

        # LEFT COLUMN: Interpretation + Keywords
        with col1:
            st.markdown(f"### Cluster {cluster}")
            st.markdown(f"**Theme:** {cluster_interpretations.get(cluster, 'No interpretation available')}")
            st.markdown("**Top Keywords:**")
            st.write(", ".join(keywords_list[:top_n_words]))

        # RIGHT COLUMN: Word Cloud
        with col2:
            wc = WordCloud(
                width=400,
                height=220,
                background_color='white'
            ).generate(' '.join(keywords_list[:top_n_words]))
            st.image(wc.to_array(), caption=f"Cluster {cluster} Word Cloud")

    # ===============================
    # 4. Example Articles with interpretations
    # ===============================
    st.subheader("Example Articles per Cluster")

    # Ensure df['cluster'] is integer type
    if df['cluster'].dtype != int:
        df['cluster'] = df['cluster'].astype(int)

    for cluster in sorted(df['cluster'].unique()):
        st.markdown(f"### Cluster {cluster}: {cluster_interpretations.get(cluster, '')}")
        articles = df[df['cluster'] == cluster]['title'].head(num_examples).tolist()
        for a in articles:
            st.write(f"- {a}")
        st.markdown("---")
