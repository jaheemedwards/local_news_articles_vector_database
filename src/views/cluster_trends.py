import streamlit as st
import pandas as pd
import plotly.express as px

def show_cluster_trends(df, cluster_interpretations):
    """
    Display trends of news article clusters over time.
    
    Parameters:
    - df: DataFrame containing 'date_iso' and 'cluster' columns
    - cluster_interpretations: dict mapping cluster ID -> descriptive label
    """
    st.title("📈 Cluster Trends Over Time")
    st.write("Explore how the number of articles in each cluster changes over time.")

    # --- Ensure date column is datetime ---
    if not pd.api.types.is_datetime64_any_dtype(df['date_iso']):
        df['date_iso'] = pd.to_datetime(df['date_iso'], errors='coerce')

    # Drop rows where date or cluster is missing
    df_clean = df.dropna(subset=['date_iso', 'cluster'])
    if df_clean.empty:
        st.warning("No data available to display trends.")
        return

    # --- Aggregate counts per cluster per day ---
    trend_df = df_clean.groupby([pd.Grouper(key='date_iso', freq='D'), 'cluster']).size().reset_index(name='count')

    # Map numeric cluster IDs to descriptive labels
    trend_df['cluster_label'] = trend_df['cluster'].map(cluster_interpretations)

    # --- Select clusters ---
    clusters = trend_df['cluster_label'].unique().tolist()
    selected_clusters = st.multiselect(
        "Select clusters to display",
        options=clusters,
        default=clusters
    )
    if not selected_clusters:
        st.warning("Please select at least one cluster to display.")
        return

    trend_df = trend_df[trend_df['cluster_label'].isin(selected_clusters)]

    # --- Plot daily trends ---
    st.subheader("Daily Article Trends")
    fig_daily = px.line(
        trend_df,
        x='date_iso',
        y='count',
        color='cluster_label',
        markers=True,
        title="Number of Articles per Cluster Over Time (Daily)",
        labels={'date_iso': 'Date', 'count': 'Article Count', 'cluster_label': 'Cluster'}
    )
    st.plotly_chart(fig_daily, width='stretch')

    # --- Plot smoothed trends (weekly) ---
    st.subheader("Smoothed Trends (Weekly)")
    trend_df.set_index('date_iso', inplace=True)

    smoothed_list = []
    for cluster_label in selected_clusters:
        df_cluster = trend_df[trend_df['cluster_label'] == cluster_label]
        weekly = df_cluster['count'].resample('W').sum().reset_index()
        weekly['cluster_label'] = cluster_label
        smoothed_list.append(weekly)

    smoothed_df = pd.concat(smoothed_list, ignore_index=True)

    fig_smooth = px.line(
        smoothed_df,
        x='date_iso',
        y='count',
        color='cluster_label',
        markers=True,
        title="Number of Articles per Cluster Over Time (Smoothed Weekly)",
        labels={'date_iso': 'Date', 'count': 'Article Count', 'cluster_label': 'Cluster'}
    )
    st.plotly_chart(fig_smooth, width='stretch')

    # --- Optional: show raw data ---
    if st.checkbox("Show raw data"):
        st.dataframe(trend_df.reset_index())
