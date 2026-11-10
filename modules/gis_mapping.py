"""
SlopeGuard AI — GIS Risk Mapping Module
Renders interactive spatial maps of Himachal Pradesh terrain and historical occurrences.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import HP_CENTER


def find_nearest_dataset_sample(lat: float, lon: float, df: pd.DataFrame) -> pd.Series:
    """
    Finds the closest environmental sample point in the 12,409 dataset using Haversine distance.
    """
    lat_r = np.radians(df['latitude'].values)
    lon_r = np.radians(df['longitude'].values)
    target_lat_r = np.radians(lat)
    target_lon_r = np.radians(lon)

    dlat = lat_r - target_lat_r
    dlon = lon_r - target_lon_r

    a = np.sin(dlat / 2.0)**2 + np.cos(target_lat_r) * np.cos(lat_r) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    dist_km = 6371.0 * c

    min_idx = np.argmin(dist_km)
    matched = df.iloc[min_idx].copy()
    matched['distance_to_target_km'] = round(float(dist_km[min_idx]), 3)
    return matched


def create_gis_risk_map(
    df_all: pd.DataFrame,
    selected_lat: float,
    selected_lon: float,
    selected_probability_pct: float = None,
    sample_limit: int = 1400,
    uploaded_dataset: bool = False
) -> go.Figure:
    """
    Builds an interactive map for the active dataset.
    """
    plot_df = df_all.dropna(subset=['latitude', 'longitude']).copy()
    if len(plot_df) > sample_limit:
        plot_df = plot_df.sample(n=sample_limit, random_state=42)

    if 'class' in plot_df:
        class_labels = (
            {1: 'Uploaded class 1', 0: 'Uploaded class 0'}
            if uploaded_dataset else
            {1: 'Landslide (Historical Failure)', 0: 'Pseudo-absence (Reference Terrain)'}
        )
        plot_df['Inventory Class'] = plot_df['class'].map(class_labels).fillna('Unclassified')
    else:
        plot_df['Inventory Class'] = 'Uploaded dataset sample' if uploaded_dataset else 'Dataset sample'

    hover_data = {
        column: format_spec
        for column, format_spec in (
            ('slope_deg', ':.1f'),
            ('elevation_m', ':.0f'),
            ('Rainfall_z', ':.2f'),
            ('NDVI_z', ':.2f'),
            ('lulc_name', True)
        )
        if column in plot_df.columns
    }

    fig = px.scatter_mapbox(
        plot_df,
        lat="latitude",
        lon="longitude",
        color="Inventory Class",
        color_discrete_map={
            'Landslide (Historical Failure)': '#e11d48',
            'Pseudo-absence (Reference Terrain)': '#0284c7',
            'Uploaded class 1': '#e11d48',
            'Uploaded class 0': '#0284c7',
            'Uploaded dataset sample': '#0284c7',
            'Dataset sample': '#0284c7',
            'Unclassified': '#64748b'
        },
        hover_data=hover_data,
        zoom=8,
        center={"lat": float(selected_lat), "lon": float(selected_lon)},
        mapbox_style="carto-positron",
        opacity=0.72,
        height=490
    )

    # Highlight currently selected location
    hover_text = f"📍 Selected Target Site<br>Lat: {selected_lat:.4f}°N, Lon: {selected_lon:.4f}°E"
    if selected_probability_pct is not None:
        hover_text += f"<br>AI Landslide Probability: {selected_probability_pct}%"

    fig.add_trace(go.Scattermapbox(
        lat=[float(selected_lat)],
        lon=[float(selected_lon)],
        mode='markers+text',
        marker=dict(size=24, color='#d97706', symbol='circle'),
        text=["📍 TARGET SITE"],
        textposition="top center",
        name="Target Location",
        hoverinfo="text",
        hovertext=hover_text
    ))

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(color="#0f172a", size=11, family="Inter")
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff"
    )

    return fig
