#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import joblib
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="ROSHN Community Intelligence",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PATHS
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# ============================================================================
# CUSTOM CSS - Auto Dark/Light Theme (System Preference)
# ============================================================================
st.markdown("""
<style>
    /* ---- Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Lato:wght@300;400;500;600;700&display=swap');

    /* ==================================================================
       CSS VARIABLES - Light Mode (Default)
       ================================================================== */
    /* ==================================================================
       THEME VARIABLES — Change these to restyle the entire dashboard
       
       HOW TO USE:
       - Change any value below and it cascades to ALL components
       - Colors use hex (#B8860B) or rgba for transparency
       - Sizes use px, rem, or % 
       - To change the brand color: update --accent-gold
       - To change the background: update --bg-primary
       - To change text darkness: update --text-primary
       ================================================================== */
    :root {
        /* ── BACKGROUNDS ─────────────────────────────────────────── */
        --bg-primary: #F7F5F0;        /* Main page background (warm off-white) */
        --bg-card: #FFFFFF;            /* KPI cards, chart wrappers, tables */
        --bg-sidebar: linear-gradient(180deg, #2C2C2C 0%, #1A1A1A 100%);  /* Sidebar gradient (hidden in current layout) */
        --bg-section: linear-gradient(90deg, #F2EDE3 0%, transparent 100%);  /* Section header background fade */
        --bg-chart: #FFFFFF;           /* Plotly chart container background */
        --bg-plot: #FDFCFA;            /* Plotly plot area inside chart */

        /* ── TEXT COLORS ─────────────────────────────────────────── */
        --text-primary: #1A1A1A;       /* Headings, KPI values, strong text */
        --text-secondary: #3D3D3D;     /* Body paragraphs, descriptions */
        --text-muted: #7A6C5D;         /* Labels, captions, nav items (inactive) */
        --text-heading2: #8B6914;      /* H2 subheadings (gold-brown) */
        --text-sidebar: #D4C5B0;       /* Sidebar text (light for dark sidebar bg) */
        --text-chart: #2A2A2A;         /* Chart axis labels, tick text */

        /* ── BORDERS ─────────────────────────────────────────────── */
        --border-main: #D8CDB8;        /* Card borders, dividers, table borders */
        --border-grid: #E8E0D4;        /* Chart gridlines (lighter than borders) */
        --border-sidebar: #3D3D3D;     /* Sidebar borders */

        /* ── BRAND ACCENT — The signature gold ──────────────────── */
        --accent-gold: #B8860B;        /* Gold accent: nav active, section bars, AI panel, badges */
        --accent-hover: rgba(184, 134, 11, 0.14);  /* Gold with transparency for hover states */

        /* ── ALERTS ──────────────────────────────────────────────── */
        --alert-bg: #FFFDF8;           /* Alert card background (warm white) */
        --alert-text: #2A2A2A;         /* Alert body text */

        /* ── SHADOWS ─────────────────────────────────────────────── */
        --shadow-sm: 0 2px 12px rgba(0,0,0,0.06);        /* Default card shadow */
        --shadow-hover: 0 12px 36px rgba(184, 134, 11, 0.16);  /* Hover shadow (gold-tinted) */

        /* ── INTERACTIVE STATES ──────────────────────────────────── */
        --hover-bg: #1A1A2E;           /* Hover background for buttons */
        --hover-text: #FAF8F5;         /* Hover text color */
    }

    /* ==================================================================
       Dark Mode - DISABLED (forced light mode for consistent demo)
       To re-enable: uncomment and wrap in @media (prefers-color-scheme: dark) { }
       ================================================================== */

    /* ==================================================================
       GLOBAL STYLES
       ================================================================== */
    /* ==================================================================
       GLOBAL STYLES — Base app appearance
       ================================================================== */
    .stApp {
        font-family: 'Lato', sans-serif;       /* Default font for all text */
        background-color: var(--bg-primary);    /* Page background color */
    }
    
    html { scroll-behavior: smooth; }           /* Smooth scroll between sections */
    
    /* Column gap between side-by-side charts/cards */
    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem;                           /* Reduce Streamlit's default 1rem gap */
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] label {
        color: var(--text-sidebar) !important;
        font-family: 'Lato', sans-serif !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: var(--accent-gold) !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }

    /* ==================================================================
       TYPOGRAPHY — Page headings (H1, H2, H3) and body text
       - H1: Playfair Display serif — used for page titles
       - H2: Playfair Display — used for section subtitles (gold-brown)
       - H3: Lato sans-serif — used inside section-header bars
       - Body: Lato — all paragraph and list text
       ================================================================== */
    h1 {
        color: var(--text-primary) !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    h2 {
        color: var(--text-heading2) !important;
        font-family: 'Playfair Display', serif !important;
        font-weight: 500 !important;
    }
    h3 {
        color: var(--text-primary) !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }

    /* ---- Text ---- */
    .stMarkdown p, .stMarkdown li {
        color: var(--text-secondary);
        font-family: 'Lato', sans-serif;
        line-height: 1.7;
    }

    /* ==================================================================
       KPI CARDS — The 6-column metric cards on every page
       
       STRUCTURE: .kpi-card > .kpi-label + .kpi-value + .kpi-delta
       
       TO CUSTOMIZE:
       - Card size: change min-height (currently 145px)
       - Value font size: change clamp() in .kpi-value (min 22px, max 32px)
       - Top color bar: set in Python via border-top inline style
       - Hover lift: change translateY in .kpi-card:hover (-5px)
       - Animation: change fadeInUp duration (0.5s)
       ================================================================== */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border-main);
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: var(--shadow-sm);
        min-height: 145px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--border-main), transparent);
        opacity: 0.5;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover);
    }
    /* #11: Fade-in animation for KPI cards */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .kpi-card {
        animation: fadeInUp 0.5s ease forwards;
    }
    .kpi-value {
        font-size: clamp(22px, 5vw, 32px);
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        margin: 8px 0;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .kpi-label {
        font-size: 10px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-bottom: 6px;
        font-family: 'Lato', sans-serif;
        font-weight: 700;
    }
    .kpi-delta {
        font-size: 11.5px;
        margin-top: 6px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .delta-up { color: #2E7D46; }
    .delta-down { color: #C4515A; }

    /* ==================================================================
       SECTION HEADERS — Gold-bordered bars above each chart group
       
       STRUCTURE: .section-header > h3
       - Left gold border (4px) acts as visual anchor
       - Gradient background fades from warm linen to transparent
       - ::after pseudo-element adds a subtle bottom hairline
       
       TO CUSTOMIZE:
       - Border color: change var(--accent-gold)
       - Background: change the gradient in .section-header
       - Font size: change font-size in .section-header h3 (13.5px)
       ================================================================== */
    .section-header {
        background: linear-gradient(90deg, #F5F0E6 0%, #FDFCFA 60%, transparent 100%);
        border-left: 4px solid var(--accent-gold);
        padding: 10px 20px;
        border-radius: 0 10px 10px 0;
        margin: 20px 0 14px 0;
        box-shadow: inset 4px 0 12px rgba(184, 134, 11, 0.1);
        position: relative;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 4px;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, var(--border-main), transparent 80%);
    }
    .section-header h3 {
        margin: 0 !important;
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    /* ==================================================================
       ALERT CARDS — Critical/Warning/Success notification banners
       
       Used for: Early Warning alerts, pipeline alerts, drill-down summaries
       - .alert-critical: Red left border (#C4515A) — urgent issues
       - .alert-warning: Gold left border (#C9920A) — caution items
       - .alert-success: Green left border (#2E7D46) — positive status
       ================================================================== */
    .alert-critical {
        background: var(--alert-bg);
        border-left: 4px solid #C4515A;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 10px 0;
        box-shadow: var(--shadow-sm);
    }
    .alert-critical span { color: var(--alert-text); }
    .alert-warning {
        background: var(--alert-bg);
        border-left: 4px solid #C9920A;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 10px 0;
        box-shadow: var(--shadow-sm);
    }
    .alert-success {
        background: var(--alert-bg);
        border-left: 4px solid #2E7D46;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 10px 0;
        box-shadow: var(--shadow-sm);
    }

    /* ==================================================================
       RISK BADGES — Colored pill badges for risk grades
       
       Used on: Resident Deep Dive profile card, tables
       - .risk-critical: Red — E grade (75-100% default prob)
       - .risk-high: Orange — D grade (50-75%)
       - .risk-medium: Gold — C grade (25-50%)
       - .risk-low: Teal — B grade (10-25%)
       - .risk-verylow: Green — A grade (0-10%)
       ================================================================== */
    .risk-badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-family: 'Lato', sans-serif;
    }
    .risk-critical { background: #C4515A22; color: #C4515A; border: 1px solid #C4515A44; }
    .risk-high { background: #C96B3022; color: #C96B30; border: 1px solid #C96B3044; }
    .risk-medium { background: #C9920A22; color: #C9920A; border: 1px solid #C9920A44; }
    .risk-low { background: #3D8B5E22; color: #3D8B5E; border: 1px solid #3D8B5E44; }
    .risk-verylow { background: #2E7D4622; color: #2E7D46; border: 1px solid #2E7D4644; }

    /* ==================================================================
       AI INSIGHT PANEL — "Next Best Actions" recommendation card
       
       Appears at the bottom of every page (except Resident Dive)
       
       STRUCTURE:
       .ai-insight-panel
         .ai-header > .ai-icon (gold gradient square) + .ai-title
         .ai-body > .ai-action (repeated for each recommendation)
           .action-icon (▸ arrow)
           .action-text > .ai-priority (CRITICAL/HIGH/MEDIUM/LOW badge) + text
       
       TO CUSTOMIZE:
       - Panel background: change gradient in .ai-insight-panel
       - Left accent: change border-left (5px solid #B8860B)
       - Priority badge colors: .priority-critical/high/medium/low
       ================================================================== */
    .ai-insight-panel {
        background: linear-gradient(135deg, #FFFDF5 0%, #FFF9EC 50%, #FFFDF5 100%);
        border: 1px solid #D4B06A;
        border-left: 5px solid #B8860B;
        border-radius: 12px;
        padding: 22px 26px;
        margin: 20px 0 28px 0;
        box-shadow: 0 4px 20px rgba(184, 134, 11, 0.08);
        position: relative;
    }
    .ai-insight-panel::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 120px;
        height: 120px;
        background: radial-gradient(circle at top right, rgba(184, 134, 11, 0.04), transparent 70%);
        border-radius: 0 12px 0 0;
    }
    .ai-insight-panel .ai-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
    }
    .ai-insight-panel .ai-header .ai-icon {
        background: linear-gradient(135deg, #B8860B, #D4A017);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 700;
    }
    .ai-insight-panel .ai-header .ai-title {
        font-family: 'Lato', sans-serif;
        font-size: 13px;
        font-weight: 700;
        color: #B8860B;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .ai-insight-panel .ai-body {
        font-family: 'Lato', sans-serif;
        font-size: 13.5px;
        color: #3D3D3D;
        line-height: 1.75;
    }
    .ai-insight-panel .ai-action {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 8px 0 4px 0;
    }
    .ai-insight-panel .ai-action .action-icon {
        color: #B8860B;
        font-size: 14px;
        margin-top: 2px;
        flex-shrink: 0;
    }
    .ai-insight-panel .ai-action .action-text {
        font-size: 13px;
        color: #4A4A4A;
    }
    .ai-insight-panel .ai-priority {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .priority-critical { background: #C4515A22; color: #C4515A; border: 1px solid #C4515A44; }
    .priority-high { background: #C96B3022; color: #C96B30; border: 1px solid #C96B3044; }
    .priority-medium { background: #C9920A22; color: #C9920A; border: 1px solid #C9920A44; }
    .priority-low { background: #2E7D4622; color: #2E7D46; border: 1px solid #2E7D4644; }

    /* ==================================================================
       TABS — In-page tab navigation (e.g., Payment Risk has 3 tabs)
       
       - Inactive: muted text, no border
       - Active: gold text + gold bottom border (3px)
       - Hover: gold text transition
       
       TO CUSTOMIZE:
       - Active color: change var(--accent-gold) references
       - Tab padding: change padding in [data-baseweb="tab"] (12px 28px)
       - Font size: change font-size (13px)
       ================================================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        border-bottom: 2px solid var(--border-main);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        color: var(--text-muted);
        border: none;
        padding: 12px 28px;
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        transition: color 0.3s ease, border-bottom-color 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: var(--accent-gold) !important;
        border-bottom: 3px solid var(--accent-gold) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-gold);
    }

    /* ==================================================================
       DATA TABLES — Styled st.dataframe() components
       - Rounded corners (10px), border, subtle shadow
       - Font family forced to Lato for consistency
       ================================================================== */
    .stDataFrame {
        border: 1px solid var(--border-main);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    /* Dataframe header row */
    .stDataFrame [data-testid="stDataFrameResizable"] {
        font-family: 'Lato', sans-serif !important;
    }

    /* ==================================================================
       DRILL-DOWN DROPDOWNS — st.selectbox() used for chart drill-downs
       - Styled border, rounded corners, Lato font
       - Label: muted color, uppercase, 12px
       ================================================================== */
    .stSelectbox > div > div {
        border-color: var(--border-main) !important;
        border-radius: 8px !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 13px !important;
    }
    .stSelectbox label {
        color: var(--text-muted) !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }

    /* ==================================================================
       DRILL-DOWN EXPANDERS — st.expander() panels that open on click
       - Gradient header background
       - Gold hover color on summary text
       - Border separator when open
       ================================================================== */
    .stExpander {
        border: 1px solid var(--border-main) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow-sm) !important;
        overflow: hidden;
        margin: 8px 0 16px 0 !important;
    }
    .stExpander > details > summary {
        font-family: 'Lato', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        background: linear-gradient(90deg, #F5F0E6 0%, var(--bg-card) 100%) !important;
    }
    .stExpander > details > summary:hover {
        color: var(--accent-gold) !important;
    }
    .stExpander > details[open] > summary {
        border-bottom: 1px solid var(--border-main) !important;
    }

    /* ==================================================================
       METRIC CARDS — st.metric() widgets inside drill-down panels
       - Playfair serif for values, Lato uppercase for labels
       - Used inside expanders for drill-down KPIs
       ================================================================== */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-main);
        border-radius: 10px;
        padding: 14px 12px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        font-family: 'Lato', sans-serif !important;
    }

    /* ---- Divider ---- */
    hr { border-color: var(--border-main) !important; }

    /* ==================================================================
       ANIMATIONS — Fade-in effects for page load
       
       - sectionFadeIn: Used on section headers, charts, tables, alerts
       - fadeInUp: Used on KPI cards (defined in KPI section above)
       - pulse: Used on the LIVE timestamp in header
       
       TO DISABLE ANIMATIONS: set animation: none; on each class
       ================================================================== */
    @keyframes sectionFadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .section-header {
        animation: sectionFadeIn 0.4s ease forwards;
    }
    .stPlotlyChart, .stDataFrame {
        animation: sectionFadeIn 0.5s ease forwards;
    }
    .alert-critical, .alert-warning, .alert-success {
        animation: sectionFadeIn 0.4s ease forwards;
    }

    /* ==================================================================
       PLOTLY CHART CONTAINERS — Wrapper around every Plotly chart
       
       - White background, rounded corners (14px), subtle shadow
       - Modebar (camera, zoom icons) hidden by default, shows on hover
       - 4px inner padding prevents chart touching the border
       
       TO CUSTOMIZE:
       - Border radius: change border-radius (14px)
       - Shadow: change box-shadow
       - Modebar visibility: change opacity values (0 → 0.6)
       ================================================================== */
    .stPlotlyChart {
        background: var(--bg-chart);
        border: 1px solid var(--border-main);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        padding: 4px;
    }
    /* Hide Plotly modebar until hover */
    .stPlotlyChart .modebar-container {
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .stPlotlyChart:hover .modebar-container {
        opacity: 0.6;
    }
    .stPlotlyChart .modebar-btn {
        font-size: 13px !important;
    }

    /* ==================================================================
       STREAMLIT OVERRIDES — Hide default Streamlit UI elements
       
       - Hides: hamburger menu, footer, sidebar, collapse button
       - Reduces top padding from Streamlit's default ~5rem to 1.5rem
       - This gives a clean, app-like appearance
       ================================================================== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    
    .stApp > header { display: none; }          /* Hide Streamlit's header bar */
    .block-container {
        padding-top: 1.5rem !important;         /* Top padding above ROSHN header */
        padding-bottom: 1rem !important;        /* Bottom padding after last element */
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1.5rem !important;         /* Backup selector for top padding */
    }

    /* ==================================================================
       TOP HEADER BAR — "ROSHN ◆ COMMUNITY INTELLIGENCE" brand bar
       
       STRUCTURE:
       .header-bar
         .header-brand > .diamond + .name + .subtitle
         .header-meta > "LIVE" + .time (pulsing gold timestamp)
       
       TO CUSTOMIZE:
       - Logo text size: change .header-brand .name font-size (24px)
       - Subtitle: change .header-brand .subtitle
       - Gold border: change border-bottom in .header-bar
       - Pulse speed: change animation duration in @keyframes pulse (2s)
       ================================================================== */
    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 10px 0;
        border-bottom: 2px solid var(--accent-gold);
        margin-bottom: 0;
        background: linear-gradient(180deg, rgba(184, 134, 11, 0.04) 0%, transparent 100%);
    }
    .header-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-brand .diamond { color: var(--accent-gold); font-size: 9px; text-shadow: 0 0 10px rgba(184, 134, 11, 0.5); }
    .header-brand .name {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 3px;
    }
    .header-brand .subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 10.5px;
        color: var(--accent-gold);
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-left: 16px;
        padding-left: 16px;
        border-left: 2px solid var(--accent-gold);
        font-weight: 700;
    }
    .header-meta {
        font-size: 10px;
        color: var(--text-muted);
        letter-spacing: 1.5px;
        text-align: right;
        text-transform: uppercase;
    }
    .header-meta .time { color: var(--accent-gold); font-weight: 700; letter-spacing: 0.5px; }
    .header-meta .time::before { content: ''; display: none; }
    
    /* ---- Live pulse indicator ---- */
    @keyframes pulse { 
        0%, 100% { opacity: 1; } 
        50% { opacity: 0.3; } 
    }
    .header-meta .time { animation: pulse 2s ease-in-out infinite; }

    /* ---- Navigation Strip ---- */
    .nav-strip {
        display: flex;
        gap: 0;
        border-bottom: 2px solid var(--border-main);
        margin-bottom: 8px;
        overflow-x: auto;
    }
    .nav-item {
        padding: 8px 22px;
        font-family: 'Lato', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--text-muted);
        cursor: pointer;
        border-bottom: 3px solid transparent;
        transition: all 0.2s ease;
        white-space: nowrap;
        text-decoration: none;
    }
    .nav-item:hover {
        color: var(--accent-gold);
        border-bottom-color: rgba(184, 134, 11, 0.4);
    }
    .nav-item.active {
        color: var(--accent-gold);
        border-bottom-color: var(--accent-gold);
    }

    /* ---- Override Streamlit button styles for nav ---- */
    .nav-buttons button,
    .nav-buttons button:focus,
    .nav-buttons button:active,
    .nav-buttons button:visited {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        border-bottom: 3px solid transparent !important;
        color: var(--text-muted) !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 10px 4px !important;
        margin: 0 !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.2 !important;
        outline: none !important;
    }
    .nav-buttons button:hover {
        color: var(--accent-gold) !important;
        border-bottom-color: rgba(184, 134, 11, 0.4) !important;
        background: transparent !important;
    }
    /* Active page — gold underline, NO red background */
    .nav-buttons [data-testid="stBaseButton-primary"],
    .nav-buttons [data-testid="stBaseButton-primary"]:focus,
    .nav-buttons [data-testid="stBaseButton-primary"]:active,
    .nav-buttons button[kind="primary"],
    .nav-buttons button[kind="primary"]:focus,
    .nav-buttons button[kind="primary"]:active {
        color: var(--accent-gold) !important;
        border-bottom: 3px solid var(--accent-gold) !important;
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    .nav-buttons [data-testid="stBaseButton-secondary"],
    .nav-buttons [data-testid="stBaseButton-secondary"]:focus {
        color: var(--text-muted) !important;
        border-bottom: 3px solid transparent !important;
        background: transparent !important;
        background-color: transparent !important;
    }

    /* ---- Filter Row ---- */
    .filter-row {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    /* ---- Compact spacing for nav area ---- */
    .nav-buttons {
        margin-bottom: -12px;
    }
    .nav-buttons + div {
        margin-top: -8px;
    }
    /* Reduce default Streamlit block gaps */
    .stApp > div > div > div > div:first-child .stMarkdown,
    .stApp > div > div > div > div:first-child .stMultiSelect {
        margin-bottom: 0;
    }
    .stMultiSelect > div { min-height: 36px; }
    .stMultiSelect label {
        font-size: 9px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: var(--accent-gold) !important;
        margin-bottom: 2px !important;
        font-weight: 600 !important;
        opacity: 0.7;
    }

    /* ---- #7: Download button styling ---- */
    .stDownloadButton button {
        background: transparent !important;
        color: var(--accent-gold) !important;
        border: 1.5px solid var(--accent-gold) !important;
        border-radius: 6px !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        background: var(--accent-gold) !important;
        color: var(--bg-primary, #FAF8F5) !important;
    }

    /* ---- Inputs ---- */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ---- Metric styling ---- */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-main);
        border-radius: 12px;
        padding: 16px;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Playfair Display', serif !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 11px !important;
    }

    /* ---- Radio buttons in sidebar ---- */
    [data-testid="stSidebar"] .stRadio > div { gap: 2px; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: background 0.2s !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: var(--accent-hover) !important;
    }

    /* ---- Gold accent line ---- */
    .gold-line {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
        margin: 10px auto;
        border-radius: 2px;
    }

    /* ---- Resident detail rows ---- */
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-main);
    }
    .detail-row .label { color: var(--text-muted); font-size: 13px; }
    .detail-row .value { color: var(--text-primary); font-weight: 500; font-size: 13px; }
</style>

<!-- Force light theme -->
<script>
    document.documentElement.setAttribute('data-theme', 'light');
</script>
""", unsafe_allow_html=True)

# ============================================================================
# DETECT THEME FOR PLOTLY
# ============================================================================
# Force light mode for consistent demo experience
IS_DARK = False

# ============================================================================
# PLOTLY THEME - Auto adapts to dark/light
# ============================================================================
if IS_DARK:
    PLOT_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#141620",
        font=dict(family="Lato", color="#D0D6E0", size=12),
        title="",
        xaxis=dict(gridcolor="#22252F", zerolinecolor="#2A2D38", linecolor="#2A2D38",
                   tickfont=dict(color="#B0B8C8")),
        yaxis=dict(gridcolor="#22252F", zerolinecolor="#2A2D38", linecolor="#2A2D38",
                   tickfont=dict(color="#B0B8C8")),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#D0D6E0", size=12)),
        hoverlabel=dict(bgcolor="#1A1C25", font_size=12, font_color="#EAEDF3"),
    )
    CHART_TEXT_COLOR = "#EAEDF3"
    CHART_MUTED_COLOR = "#9AA4B4"
else:
    PLOT_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FDFCFA",
        font=dict(family="Lato", color="#5A5A5A", size=12),
        title="",
        xaxis=dict(gridcolor="#F0EBE3", zerolinecolor="#E8E0D4", linecolor="#E8E0D4"),
        yaxis=dict(gridcolor="#F0EBE3", zerolinecolor="#E8E0D4", linecolor="#E8E0D4"),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#5A5A5A")),
        hoverlabel=dict(bgcolor="#2C2C2C", font_size=12, font_color="#FAF8F5"),
    )
    CHART_TEXT_COLOR = "#3D3D3D"
    CHART_MUTED_COLOR = "#9A8C7A"

COLORS = {
    "teal": "#3D8B5E",       # Rich emerald sage
    "coral": "#C4515A",      # Vivid rose
    "yellow": "#C9920A",     # Deep amber gold
    "green": "#2E7D46",      # Deep forest green
    "red": "#B5383D",        # Strong crimson
    "blue": "#3A6B8C",       # Deep teal blue
    "purple": "#7B5E8D",     # Rich amethyst
    "orange": "#C96B30",     # Vivid terracotta
    "navy": "#1A1A2E",       # Deep navy
    "light": "#FAF8F5",      # Warm white
    "gold": "#B8860B",       # Dark goldenrod — richer than champagne
    "sand": "#E8E0D4",       # Sand
    "taupe": "#7A6C5D",      # Deep taupe
    "charcoal": "#2A2A2A",   # Rich charcoal
}

RISK_COLORS = {
    "A - Very Low": "#2E7D46",
    "B - Low": "#3D8B5E",
    "C - Medium": "#C9920A",
    "D - High": "#C96B30",
    "E - Critical": "#C4515A",
}

SEVERITY_COLORS = {
    "Low": "#3D8B5E",
    "Medium": "#C9920A",
    "High": "#C96B30",
    "Critical": "#C4515A",
}

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_all_data():
    """Load all datasets."""
    residents = pd.read_csv(os.path.join(DATA_DIR, "roshn_residents_master.csv"))
    payments = pd.read_csv(os.path.join(DATA_DIR, "roshn_payment_transactions.csv"))
    complaints = pd.read_csv(os.path.join(DATA_DIR, "roshn_complaints.csv"))
    bookings = pd.read_csv(os.path.join(DATA_DIR, "roshn_facility_bookings.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "roshn_ai_interactions.csv"))
    
    # Load leads data if available
    leads_path = os.path.join(DATA_DIR, "roshn_leads.csv")
    leads_scored_path = os.path.join(OUTPUT_DIR, "roshn_leads_scored.csv")
    if os.path.exists(leads_scored_path):
        leads = pd.read_csv(leads_scored_path)
    elif os.path.exists(leads_path):
        leads = pd.read_csv(leads_path)
    else:
        leads = pd.DataFrame()
    
    if not leads.empty:
        leads["created_date_dt"] = pd.to_datetime(leads["created_date"], errors="coerce")
        leads["last_activity_dt"] = pd.to_datetime(leads["last_activity_date"], errors="coerce")
    
    # Load scored data if available
    scored_path = os.path.join(OUTPUT_DIR, "roshn_residents_scored.csv")
    if os.path.exists(scored_path):
        scored = pd.read_csv(scored_path)
        # Merge predicted columns into residents
        pred_cols = ["resident_id", "predicted_default_prob", "predicted_risk_grade"]
        available = [c for c in pred_cols if c in scored.columns]
        if len(available) > 1:
            residents = residents.merge(scored[available], on="resident_id", how="left")
    
    # Parse dates
    complaints["created_date"] = pd.to_datetime(complaints["created_date"], errors="coerce")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")
    payments["due_date"] = pd.to_datetime(payments["due_date"], errors="coerce")
    bookings["booking_date"] = pd.to_datetime(bookings["booking_date"], errors="coerce")
    
    return residents, payments, complaints, bookings, interactions, leads

@st.cache_resource
def load_model_metadata():
    """Load model metadata."""
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def kpi_card(label, value, delta=None, delta_dir="up", color="#4ECDC4"):
    """Render a KPI card with colored top accent."""
    delta_html = ""
    if delta:
        arrow = "▲" if delta_dir == "up" else "▼"
        delta_class = "delta-up" if delta_dir == "up" else "delta-down"
        delta_html = f'<div class="kpi-delta {delta_class}">{arrow} {delta}</div>'
    
    return f"""
    <div class="kpi-card" style="border-top: 4px solid {color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        {delta_html}
    </div>
    """

def section_header(title, icon=""):
    """Render a section header."""
    st.markdown(f"""
    <div class="section-header">
        <h3>{icon} {title}</h3>
    </div>
    """, unsafe_allow_html=True)

def render_ai_insights(actions):
    """Render an AI insight panel with prioritized next-best-actions.
    actions: list of dicts with keys: priority (critical/high/medium/low), text
    """
    actions_html = ""
    for a in actions:
        pri = a.get("priority", "medium")
        pri_label = pri.upper()
        pri_class = f"priority-{pri}"
        actions_html += f"""
        <div class="ai-action">
            <span class="action-icon">▸</span>
            <span class="action-text"><span class="ai-priority {pri_class}">{pri_label}</span> {a['text']}</span>
        </div>"""
    
    st.markdown(f"""
    <div class="ai-insight-panel">
        <div class="ai-header">
            <div class="ai-icon">AI</div>
            <div class="ai-title">Next Best Actions — AI Recommendations</div>
        </div>
        <div class="ai-body">{actions_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# AI INSIGHT ENGINES — Data-driven recommendation generators
# ============================================================================
def insights_overview(residents, payments, complaints, interactions):
    """Generate AI insights for the Executive Summary page."""
    actions = []
    default_rate = residents["default_flag"].mean() * 100
    total_residents = len(residents)
    
    # Always-on: Default rate assessment
    if default_rate > 15:
        actions.append({"priority": "critical", "text": f"Default rate is {default_rate:.1f}% — well above the 10% industry benchmark. Launch an immediate collections task force targeting the {len(residents[residents['risk_category'].isin(['High', 'Critical'])]):,} high-risk residents before Q2 close."})
    elif default_rate > 8:
        actions.append({"priority": "high", "text": f"Default rate at {default_rate:.1f}% is above target. Prioritize proactive payment plan outreach to residents in the 30-60 DPD window to prevent escalation to delinquency."})
    else:
        actions.append({"priority": "low", "text": f"Default rate at {default_rate:.1f}% is within healthy range. Maintain current monitoring cadence and focus collections resources on the {len(residents[residents['risk_category'].isin(['High', 'Critical'])]):,} high-risk accounts."})
    
    # Always-on: Complaint load
    active_complaints = len(complaints[complaints["status"].isin(["Open", "In Progress"])])
    complaint_ratio = active_complaints / max(total_residents, 1) * 100
    top_cat = complaints[complaints["status"].isin(["Open", "In Progress"])]["category"].value_counts()
    top_cat_name = top_cat.index[0] if len(top_cat) > 0 else "N/A"
    top_cat_count = top_cat.values[0] if len(top_cat) > 0 else 0
    actions.append({"priority": "high" if complaint_ratio > 30 else "medium", "text": f"{active_complaints:,} open complaints ({complaint_ratio:.0f}% of residents). Top backlog: '{top_cat_name}' with {top_cat_count:,} open cases — assign a dedicated resolution team to this category to reduce backlog by 50% in 2 weeks."})
    
    # Always-on: AI efficiency
    ai_rate = interactions["resolved_by_ai"].mean() * 100
    esc_count = int(interactions["escalated_to_human"].sum())
    actions.append({"priority": "medium" if ai_rate < 75 else "low", "text": f"AI resolves {ai_rate:.1f}% of interactions automatically ({esc_count:,} escalated to humans). " + ("Analyze top escalation reasons to push AI resolution above 80% — each 1% improvement saves ~{:.0f}K SAR annually.".format(len(interactions) * 0.01 * 15 / 1000) if ai_rate < 80 else "AI performance is strong. Focus on maintaining quality while expanding to new interaction types.")})
    
    # Always-on: Worst community
    comm_risk = residents.groupby("community")["default_flag"].mean()
    worst_comm = comm_risk.idxmax()
    worst_rate = comm_risk.max() * 100
    best_comm = comm_risk.idxmin()
    best_rate = comm_risk.min() * 100
    actions.append({"priority": "high" if worst_rate > default_rate * 1.3 else "medium", "text": f"Community risk spread: {worst_comm} ({worst_rate:.1f}% default) vs {best_comm} ({best_rate:.1f}%). " + (f"Deploy targeted support to {worst_comm} — the {worst_rate - default_rate:.1f}pp gap above average suggests community-specific issues." if worst_rate > default_rate * 1.2 else "Risk is relatively uniform across communities — maintain portfolio-wide monitoring.")})
    
    return actions


def insights_payment_risk(residents, payments):
    """Generate AI insights for the Payment Risk page."""
    actions = []
    avg_rate = residents["default_flag"].mean() * 100
    
    # Always-on: High-risk residents
    high_risk = residents[residents["risk_category"].isin(["High", "Critical"])]
    hr_balance = high_risk["outstanding_balance_aed"].sum() if len(high_risk) > 0 else 0
    hr_str = f"{hr_balance/1e9:.1f}B" if hr_balance >= 1e9 else f"{hr_balance/1e6:.0f}M"
    if len(high_risk) > 0:
        actions.append({"priority": "critical", "text": f"{len(high_risk):,} residents at high/critical risk holding {hr_str} SAR in outstanding balances. Start with the top 25 by risk score — personal outreach within 48 hours with tailored payment restructuring options."})
    
    # Always-on: DPD analysis
    dpd_0 = (residents["current_dpd"] == 0).sum()
    dpd_1_30 = ((residents["current_dpd"] > 0) & (residents["current_dpd"] <= 30)).sum()
    dpd_30_60 = ((residents["current_dpd"] > 30) & (residents["current_dpd"] <= 60)).sum()
    dpd_60_plus = (residents["current_dpd"] > 60).sum()
    if dpd_30_60 > 0:
        actions.append({"priority": "high", "text": f"DPD breakdown: {dpd_0:,} current, {dpd_1_30:,} early (1-30), {dpd_30_60:,} in danger zone (30-60), {dpd_60_plus:,} delinquent (60+). The {dpd_30_60:,} residents in the 30-60 window are the highest-ROI intervention target — automated reminders + agent calls before they cross 60 days."})
    
    # Always-on: Zone comparison
    zone_default = residents.groupby("zone")["default_flag"].mean().sort_values(ascending=False)
    worst_zone = zone_default.index[0]
    best_zone = zone_default.index[-1]
    worst_z_rate = zone_default.values[0] * 100
    best_z_rate = zone_default.values[-1] * 100
    actions.append({"priority": "high" if worst_z_rate > avg_rate * 1.3 else "medium", "text": f"Zone '{worst_zone}' has the highest default rate ({worst_z_rate:.1f}%) while '{best_zone}' has the lowest ({best_z_rate:.1f}%). " + (f"The {worst_z_rate - avg_rate:.1f}pp gap above average in {worst_zone} suggests zone-specific issues — cross-reference with complaint data for root cause." if worst_z_rate > avg_rate * 1.2 else f"Default risk is relatively even across zones. Focus on individual high-risk accounts rather than zone-level interventions.")})
    
    # Always-on: Payment pattern
    on_time_rate = (payments["payment_status"] == "Paid").mean() * 100
    late_rate = (payments["payment_status"] == "Paid Late").mean() * 100
    actions.append({"priority": "medium" if on_time_rate < 60 else "low", "text": f"Payment health: {on_time_rate:.0f}% on-time, {late_rate:.0f}% late. " + (f"With {late_rate:.0f}% paying late, introduce automated payment reminders 5 days before due date and offer auto-debit enrollment incentives." if late_rate > 15 else "On-time payment rates are healthy. Maintain current reminder cadence and reward consistent payers with loyalty benefits.")})
    
    return actions


def insights_complaints(complaints):
    """Generate AI insights for the Complaints page."""
    actions = []
    total = len(complaints)
    
    # Always-on: Top backlog category
    open_complaints = complaints[complaints["status"].isin(["Open", "In Progress"])]
    if len(open_complaints) > 0:
        top_cat = open_complaints["category"].value_counts()
        top_name = top_cat.index[0]
        top_count = top_cat.values[0]
        actions.append({"priority": "high", "text": f"'{top_name}' leads the open backlog with {top_count:,} unresolved complaints out of {len(open_complaints):,} total. Assign a dedicated team of 3 agents to clear this category within 2 weeks using batch resolution templates."})
    
    # Always-on: Escalation assessment
    escalated = len(complaints[complaints["status"] == "Escalated"])
    esc_pct = escalated / max(total, 1) * 100
    actions.append({"priority": "critical" if esc_pct > 5 else "high" if esc_pct > 3 else "medium", "text": f"Escalation rate is {esc_pct:.1f}% ({escalated:,} cases). " + (f"This exceeds the 3% target — review escalation criteria and empower front-line agents with higher resolution authority." if esc_pct > 3 else f"Below the 3% target threshold. Maintain current escalation protocols and monitor for any upward trends.")})
    
    # Always-on: Resolution time analysis
    avg_resolution = complaints["resolution_hours"].mean()
    resolution_by_cat = complaints.groupby("category")["resolution_hours"].mean().sort_values(ascending=False)
    slowest_cat = resolution_by_cat.index[0]
    slowest_hrs = resolution_by_cat.values[0]
    actions.append({"priority": "high" if avg_resolution > 72 else "medium", "text": f"Average resolution is {avg_resolution:.0f} hours. Slowest category: '{slowest_cat}' at {slowest_hrs:.0f} hours. " + (f"Create SOP fast-track workflows for this category to cut resolution time by 40%." if slowest_hrs > 80 else f"Consider setting a 48-hour SLA for all categories to drive continuous improvement.")})
    
    # Always-on: Severity distribution insight
    critical_count = len(complaints[complaints["severity"] == "Critical"])
    critical_pct = critical_count / max(total, 1) * 100
    if critical_count > 0:
        worst_zone = complaints[complaints["severity"] == "Critical"]["zone"].value_counts()
        wz_name = worst_zone.index[0] if len(worst_zone) > 0 else "N/A"
        wz_count = worst_zone.values[0] if len(worst_zone) > 0 else 0
        actions.append({"priority": "high" if critical_pct > 5 else "medium", "text": f"{critical_count:,} critical complaints ({critical_pct:.1f}% of total). Zone '{wz_name}' has the most ({wz_count:,}). Schedule an emergency review with the zone facilities team and create a 7-day action plan for the top 5 issues."})
    
    return actions


def insights_sentiment(residents, interactions):
    """Generate AI insights for the Sentiment page."""
    actions = []
    
    avg_sentiment = interactions["sentiment_score"].mean()
    negative_pct = (interactions["sentiment_score"] < -0.2).mean() * 100
    positive_pct = (interactions["sentiment_score"] > 0.2).mean() * 100
    avg_csat = interactions["csat_score"].mean()
    
    # Always-on: Sentiment health assessment
    if negative_pct > 20:
        actions.append({"priority": "critical", "text": f"{negative_pct:.0f}% of interactions have negative sentiment — over 1 in 5 residents leave conversations unhappy. Launch a Voice of Customer program to extract the top 5 negative themes and create targeted improvement plans."})
    elif negative_pct > 15:
        actions.append({"priority": "high", "text": f"{negative_pct:.0f}% of interactions carry negative sentiment. Implement real-time sentiment monitoring to flag negative interactions for supervisor review within the same business day."})
    else:
        neutral_pct = 100 - positive_pct - negative_pct
        if neutral_pct > 40:
            actions.append({"priority": "medium", "text": f"{neutral_pct:.0f}% of interactions have neutral sentiment — residents are not dissatisfied but also not delighted. Introduce proactive follow-up messages after service interactions to convert neutral experiences into positive ones."})
    
    # Always-on: Channel comparison (always useful)
    channel_sent = interactions.groupby("channel")["sentiment_score"].mean().sort_values()
    best_ch = channel_sent.idxmax()
    worst_ch = channel_sent.idxmin()
    gap = channel_sent.max() - channel_sent.min()
    if gap > 0.08:
        actions.append({"priority": "high" if gap > 0.15 else "medium", "text": f"'{best_ch}' channel leads in sentiment ({channel_sent.max():.2f}) while '{worst_ch}' lags ({channel_sent.min():.2f}). Study what makes {best_ch} successful (faster resolution? better scripts?) and replicate those practices in {worst_ch}."})
    
    # Always-on: Community satisfaction ranking
    comm_sat = residents.groupby("community")["satisfaction_score"].mean().sort_values()
    bottom_3 = comm_sat.head(3)
    top_3 = comm_sat.tail(3)
    worst_comm = comm_sat.index[0]
    worst_score = comm_sat.values[0]
    best_score = comm_sat.values[-1]
    spread = best_score - worst_score
    
    if spread > 5:
        actions.append({"priority": "high" if spread > 10 else "medium", "text": f"Satisfaction ranges from {worst_score:.1f} ({worst_comm}) to {best_score:.1f} ({comm_sat.index[-1]}) — a {spread:.1f}-point gap. Pair the bottom 3 communities ({', '.join(bottom_3.index.tolist())}) with mentors from the top 3 ({', '.join(top_3.index.tolist())}) to transfer best practices."})
    
    # CSAT distribution insight
    low_csat_pct = (interactions["csat_score"] <= 2).mean() * 100
    high_csat_pct = (interactions["csat_score"] >= 4).mean() * 100
    if low_csat_pct > 10:
        actions.append({"priority": "high" if low_csat_pct > 20 else "medium", "text": f"{low_csat_pct:.0f}% of interactions scored CSAT 1-2 while {high_csat_pct:.0f}% scored 4-5. Implement an automatic callback for every CSAT-1 rating — recovering these detractors has 5x the retention impact of delighting already-happy residents."})
    
    # Sentiment by purpose — find worst purpose
    purpose_sent = interactions.groupby("purpose")["sentiment_score"].mean().sort_values()
    worst_purpose = purpose_sent.index[0]
    worst_p_score = purpose_sent.values[0]
    if worst_p_score < 0:
        actions.append({"priority": "high", "text": f"'{worst_purpose}' interactions have negative average sentiment ({worst_p_score:.2f}). This interaction type is creating detractors — redesign the process flow, add proactive status updates, and train agents specifically for this scenario."})
    elif worst_p_score < 0.15:
        actions.append({"priority": "medium", "text": f"'{worst_purpose}' has the lowest sentiment ({worst_p_score:.2f}) among interaction types. While not negative, there's room to improve — consider adding automated follow-ups and satisfaction checks for this purpose."})
    
    return actions


def insights_leads(leads):
    """Generate AI insights for the Lead Intelligence page."""
    actions = []
    
    if leads.empty:
        return actions
    
    active = leads[~leads["stage"].isin(["Won", "Lost"])]
    conv_rate = (leads["stage"] == "Won").mean() * 100
    
    # Stale leads
    if "days_since_activity" in leads.columns:
        stale = active[active["days_since_activity"] > 30]
        if len(stale) > 100:
            actions.append({"priority": "high", "text": f"{len(stale):,} active leads have been inactive for 30+ days. Launch a re-engagement campaign: personalized email + WhatsApp message with a new property recommendation based on their original interest."})
    
    # Response time
    avg_resp = leads["response_time_hours"].mean()
    if avg_resp > 8:
        actions.append({"priority": "critical", "text": f"Average lead response time is {avg_resp:.1f} hours — industry best practice is under 2 hours. Implement an auto-assignment rule: new leads must receive first contact within 1 hour during business hours."})
    elif avg_resp > 4:
        actions.append({"priority": "high", "text": f"Average response time is {avg_resp:.1f} hours. Leads contacted within 2 hours convert at 3x the rate. Set up automated instant-response messages while agents prepare personalized follow-ups."})
    
    # Source optimization
    source_conv = leads.groupby("source").agg(
        total=("lead_id", "count"),
        won=("stage", lambda x: (x == "Won").sum()),
    ).reset_index()
    source_conv["rate"] = source_conv["won"] / source_conv["total"] * 100
    best_source = source_conv.loc[source_conv["rate"].idxmax()]
    worst_source = source_conv.loc[source_conv["rate"].idxmin()]
    if best_source["rate"] > worst_source["rate"] * 2:
        actions.append({"priority": "medium", "text": f"'{best_source['source']}' converts at {best_source['rate']:.1f}% while '{worst_source['source']}' converts at only {worst_source['rate']:.1f}%. Shift 20% of marketing budget from {worst_source['source']} to {best_source['source']} for immediate ROI improvement."})
    
    # Hot leads action
    if "lead_priority" in leads.columns:
        hot = active[active["lead_priority"] == "Hot"]
        unscheduled = hot[hot.get("follow_up_scheduled", pd.Series(False, index=hot.index)) == False] if "follow_up_scheduled" in hot.columns else hot
        if len(unscheduled) > 10:
            actions.append({"priority": "critical", "text": f"{len(unscheduled):,} hot leads (>75% conversion probability) don't have follow-ups scheduled. These are your highest-value opportunities — assign to top agents and schedule calls within 24 hours."})
    
    return actions


def insights_demand(bookings, complaints):
    """Generate AI insights for the Demand page."""
    actions = []
    
    no_show_rate = (bookings["status"] == "No-Show").mean() * 100
    if no_show_rate > 10:
        actions.append({"priority": "high", "text": f"No-show rate is {no_show_rate:.1f}%. Implement a 24-hour booking reminder (SMS + push notification) and a 3-strike policy — residents with 3+ no-shows get deprioritized in booking queues."})
    
    # Overbooked facilities
    fac_counts = bookings["facility"].value_counts()
    top_fac = fac_counts.index[0]
    top_count = fac_counts.values[0]
    bottom_fac = fac_counts.index[-1]
    bottom_count = fac_counts.values[-1]
    if top_count > bottom_count * 3:
        actions.append({"priority": "medium", "text": f"'{top_fac}' ({top_count:,} bookings) is 3x+ more popular than '{bottom_fac}' ({bottom_count:,}). Consider adding capacity for {top_fac} and running promotional campaigns to drive traffic to underutilized amenities."})
    
    # Weekend demand
    weekend = bookings[bookings["day_of_week"].isin(["Friday", "Saturday"])]
    weekday = bookings[~bookings["day_of_week"].isin(["Friday", "Saturday"])]
    if len(weekend) / 2 > len(weekday) / 5 * 1.5:
        actions.append({"priority": "medium", "text": "Weekend demand is 50%+ higher per day than weekdays. Staff amenity areas with 2x personnel on Friday-Saturday and consider extended hours for top facilities."})
    
    # Maintenance-related
    maint_cats = ["Plumbing/Water Leakage", "Electrical Issues", "HVAC/Air Conditioning"]
    maint = complaints[complaints["category"].isin(maint_cats)]
    if len(maint) > 0:
        maint_comm = maint.groupby("community").size().sort_values(ascending=False)
        top_maint_comm = maint_comm.index[0]
        actions.append({"priority": "high", "text": f"'{top_maint_comm}' leads in maintenance complaints ({maint_comm.values[0]:,}). Schedule a preventive maintenance audit for this community — proactive repairs cost 60% less than emergency fixes."})
    
    return actions


def insights_ai_performance(interactions, residents, leads):
    """Generate AI insights for the AI Performance page."""
    actions = []
    
    ai_rate = interactions["resolved_by_ai"].mean() * 100
    esc_rate = interactions["escalated_to_human"].mean() * 100
    
    if esc_rate > 20:
        actions.append({"priority": "high", "text": f"Escalation rate is {esc_rate:.1f}% — every 1% reduction saves ~{int(len(interactions) * 0.01 * 15):,} SAR annually. Analyze the top 3 escalation purposes and expand AI training data for those scenarios."})
    
    # CSAT gap between AI and human
    ai_csat = interactions[interactions["resolved_by_ai"] == True]["csat_score"].mean() if "resolved_by_ai" in interactions.columns else 0
    human_csat = interactions[interactions["resolved_by_ai"] == False]["csat_score"].mean() if "resolved_by_ai" in interactions.columns else 0
    if human_csat > ai_csat + 0.3:
        actions.append({"priority": "medium", "text": f"Human-resolved interactions score {human_csat:.1f}/5 CSAT vs AI's {ai_csat:.1f}/5. Focus AI improvement on empathy signals and complex query handling to close the gap."})
    
    # Lead AI ROI
    if not leads.empty:
        ai_leads = leads[leads["ai_assisted"] == True]
        manual_leads = leads[leads["ai_assisted"] == False]
        ai_conv = (ai_leads["stage"] == "Won").mean() * 100 if len(ai_leads) > 0 else 0
        man_conv = (manual_leads["stage"] == "Won").mean() * 100 if len(manual_leads) > 0 else 0
        if ai_conv > man_conv:
            actions.append({"priority": "medium", "text": f"AI-assisted leads convert at {ai_conv:.1f}% vs {man_conv:.1f}% for manual. Expand AI assistance to 100% of leads — current coverage is {leads['ai_assisted'].mean()*100:.0f}%."})
        elif man_conv > ai_conv * 1.2:
            actions.append({"priority": "high", "text": f"Manual leads outperform AI-assisted ({man_conv:.1f}% vs {ai_conv:.1f}%). Review AI lead scoring logic — it may be routing low-quality leads to agents, creating a misleading comparison."})
    
    return actions

def risk_badge(grade):
    """Return HTML risk badge."""
    badge_map = {
        "A - Very Low": "risk-verylow",
        "B - Low": "risk-low",
        "C - Medium": "risk-medium",
        "D - High": "risk-high",
        "E - Critical": "risk-critical",
    }
    css_class = badge_map.get(grade, "risk-low")
    return f'<span class="risk-badge {css_class}">{grade}</span>'

# ============================================================================
# TOP NAVIGATION
# ============================================================================
NAV_PAGES = [
    "Overview",
    "Payment Risk",
    "Complaints",
    "Sentiment",
    "Leads",
    "Demand",
    "AI Performance",
    "Resident Dive",
]

def render_top_nav():
    """Render luxury top navigation bar and filters."""

    # ---- Header Bar: Brand + Timestamp ----
    st.markdown(f"""
    <div class="header-bar">
        <div class="header-brand">
            <span class="diamond">◆</span>
            <span class="name">ROSHN</span>
            <span class="diamond">◆</span>
            <span class="subtitle">Community Intelligence</span>
        </div>
        <div class="header-meta">
            <span>LIVE</span>&ensp;
            <span class="time">◉ {datetime.now().strftime("%d %b %Y, %H:%M")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Navigation Strip ----
    page = st.session_state.get("nav_page", NAV_PAGES[0])

    st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
    cols = st.columns(len(NAV_PAGES))
    for i, (col, pg) in enumerate(zip(cols, NAV_PAGES)):
        with col:
            if st.button(pg, key=f"nav_{i}", use_container_width=True,
                        type="primary" if page == pg else "secondary"):
                st.session_state["nav_page"] = pg
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    page = st.session_state.get("nav_page", NAV_PAGES[0])

    # ---- Compact Filter Row ----
    fc1, fc2, fc3 = st.columns([5, 5, 2])
    with fc1:
        communities = st.multiselect(
            "Filter by Community",
            options=sorted(residents["community"].unique()),
            default=[],
            placeholder="All Communities",
        )
    with fc2:
        zones = st.multiselect(
            "Filter by Zone",
            options=sorted(residents["zone"].unique()),
            default=[],
            placeholder="All Zones",
        )
    with fc3:
        st.markdown("")  # spacing alignment

    return page, communities, zones

def apply_filters(df, communities, zones, comm_col="community", zone_col="zone"):
    """Apply community and zone filters."""
    if communities:
        df = df[df[comm_col].isin(communities)]
    if zones:
        df = df[df[zone_col].isin(zones)]
    return df

# ============================================================================
# PAGE: EXECUTIVE SUMMARY
# ============================================================================
def page_executive_summary(residents, payments, complaints, bookings, interactions):
    """Executive Summary Dashboard."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 4px 0 16px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 4px; text-transform: uppercase; font-family: 'Lato', sans-serif; font-weight: 600;">Enterprise AI for Real Estate Operations</div>
        <h1 style="font-size: 30px; margin: 8px 0 4px 0; font-family: 'Playfair Display', serif !important;">Community Intelligence Command Center</h1>
        <div class="gold-line"></div>
        <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Real-Time Analytics & Predictive Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPI Row 1 ----
    total_residents = len(residents)
    active_complaints = len(complaints[complaints["status"].isin(["Open", "In Progress"])])
    default_rate = residents["default_flag"].mean() * 100 if "default_flag" in residents.columns else 0
    avg_satisfaction = residents["satisfaction_score"].mean() if "satisfaction_score" in residents.columns else 0
    ai_resolution = interactions["resolved_by_ai"].mean() * 100 if "resolved_by_ai" in interactions.columns else 0
    total_portfolio = residents["property_value_aed"].sum()
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        st.markdown(kpi_card("Total Residents", f"{total_residents:,}", "", "up", COLORS["teal"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Portfolio Value", f"{total_portfolio/1e9:.1f}B SAR", "", "up", COLORS["blue"]), unsafe_allow_html=True)
    with c3:
        delta_dir = "down" if default_rate > 5 else "up"
        st.markdown(kpi_card("Default Rate", f"{default_rate:.1f}%", f"{default_rate:.1f}% of residents", delta_dir, COLORS["coral"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Active Complaints", f"{active_complaints:,}", f"{active_complaints/total_residents*100:.1f}% of residents", "down", COLORS["yellow"]), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Satisfaction Score", f"{avg_satisfaction:.1f}", "Out of 100", "up", COLORS["green"]), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("AI Resolution Rate", f"{ai_resolution:.1f}%", f"{len(interactions):,} interactions", "up", COLORS["purple"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Row 2: Charts ----
    col1, col2 = st.columns([3, 2])
    
    with col1:
        section_header("Risk Distribution by Community", "📊")
        
        if "predicted_risk_grade" in residents.columns:
            risk_comm = residents.groupby(["community", "predicted_risk_grade"]).size().reset_index(name="count")
            fig = px.bar(
                risk_comm, x="community", y="count", color="predicted_risk_grade",
                color_discrete_map=RISK_COLORS,
                barmode="stack"
            )
            fig.update_layout(**PLOT_LAYOUT, height=400, showlegend=True,
                            legend_title_text="Risk Grade",
                            xaxis_title="", yaxis_title="Residents")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            risk_comm = residents.groupby(["community", "risk_category"]).size().reset_index(name="count")
            fig = px.bar(risk_comm, x="community", y="count", color="risk_category", barmode="stack")
            fig.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # DRILL-DOWN: Community detail
        comm_list = sorted(residents["community"].unique())
        selected_comm = st.selectbox("🔍 Drill into community", ["Select a community..."] + comm_list, key="overview_comm_drill")
        if selected_comm != "Select a community...":
            comm_data = residents[residents["community"] == selected_comm]
            risk_col = "predicted_risk_grade" if "predicted_risk_grade" in residents.columns else "risk_category"
            with st.expander(f"📊 {selected_comm} — {len(comm_data):,} residents", expanded=True):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Residents", f"{len(comm_data):,}")
                dc2.metric("Default Rate", f"{comm_data['default_flag'].mean()*100:.1f}%")
                dc3.metric("Avg Credit Score", f"{comm_data['credit_score'].mean():.0f}")
                dc4.metric("Portfolio", f"{comm_data['property_value_aed'].sum()/1e9:.1f}B SAR")
                drill_risk = comm_data[risk_col].value_counts().sort_index()
                fig_d = go.Figure(data=[go.Bar(
                    x=drill_risk.index, y=drill_risk.values,
                    marker_color=[RISK_COLORS.get(k, COLORS["teal"]) for k in drill_risk.index],
                    text=drill_risk.values, textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                )])
                fig_d.update_layout(**{**PLOT_LAYOUT, "title": f"Risk Distribution — {selected_comm}"}, height=280, xaxis_title="", yaxis_title="Residents")
                st.plotly_chart(fig_d, use_container_width=True)
    
    with col2:
        section_header("Portfolio Risk Overview", "🎯")
        
        if "predicted_risk_grade" in residents.columns:
            risk_dist = residents["predicted_risk_grade"].value_counts().sort_index()
        else:
            risk_dist = residents["risk_category"].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_dist.index,
            values=risk_dist.values,
            hole=0.55,
            marker_colors=[RISK_COLORS.get(k, COLORS["teal"]) for k in risk_dist.index],
            textinfo="label+percent",
            textfont=dict(color=CHART_TEXT_COLOR, size=11),
        )])
        fig.update_layout(**PLOT_LAYOUT, height=400, showlegend=False)
        fig.add_annotation(text=f"<b>{total_residents:,}</b><br>Residents",
                          x=0.5, y=0.5, font_size=16, font_color=CHART_TEXT_COLOR,
                          showarrow=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- Row 3: Trends ----
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Complaint Trends (Monthly)", "📋")
        
        complaints["month"] = complaints["created_date"].dt.to_period("M").astype(str)
        monthly_complaints = complaints.groupby("month").size().reset_index(name="count")
        monthly_complaints = monthly_complaints.tail(18)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_complaints["month"], y=monthly_complaints["count"],
            mode="lines+markers",
            line=dict(color=COLORS["coral"], width=3),
            marker=dict(size=8, color=COLORS["coral"]),
            fill="tozeroy",
            fillcolor="rgba(196, 81, 90, 0.12)"
        ))
        fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title="", yaxis_title="Complaints",
                         showlegend=False)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Sentiment Trend (Monthly)", "😊")
        
        interactions["month"] = interactions["timestamp"].dt.to_period("M").astype(str)
        monthly_sentiment = interactions.groupby("month")["sentiment_score"].mean().reset_index()
        monthly_sentiment = monthly_sentiment.tail(18)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_sentiment["month"], y=monthly_sentiment["sentiment_score"],
            mode="lines+markers",
            line=dict(color=COLORS["teal"], width=3),
            marker=dict(size=8, color=COLORS["teal"]),
            fill="tozeroy",
            fillcolor="rgba(61, 139, 94, 0.12)"
        ))
        fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title="", yaxis_title="Avg Sentiment",
                         showlegend=False)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- Row 4: Zone Performance ----
    section_header("Zone Performance Summary", "🗺️")
    
    zone_metrics = residents.groupby("zone").agg(
        residents_count=("resident_id", "count"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_credit_score=("credit_score", "mean"),
        default_rate=("default_flag", "mean"),
        avg_property_value=("property_value_aed", "mean"),
    ).reset_index()
    zone_metrics["default_rate"] = (zone_metrics["default_rate"] * 100).round(1)
    zone_metrics["avg_satisfaction"] = zone_metrics["avg_satisfaction"].round(1)
    zone_metrics["avg_credit_score"] = zone_metrics["avg_credit_score"].round(0).astype(int)
    zone_metrics["avg_property_value"] = (zone_metrics["avg_property_value"] / 1e6).round(2)
    zone_metrics.columns = ["Zone", "Residents", "Avg Satisfaction", "Avg Credit Score", 
                           "Default Rate %", "Avg Property Value (M SAR)"]
    
    st.dataframe(zone_metrics, use_container_width=True, hide_index=True)
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_overview(residents, payments, complaints, interactions))


# ============================================================================
# PAGE: PAYMENT RISK INTELLIGENCE
# ============================================================================
def page_payment_risk(residents, payments):
    """Payment Default Risk Intelligence."""
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">Risk Analytics</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Payment Risk Intelligence</h1>
        <p style="color: var(--text-muted);">Predictive default scoring, early warning alerts, and portfolio risk analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPI Row ----
    has_predictions = "predicted_default_prob" in residents.columns
    
    if has_predictions:
        high_risk = len(residents[residents["predicted_risk_grade"].isin(["D - High", "E - Critical"])])
        avg_prob = residents["predicted_default_prob"].mean() * 100
        at_risk_value = residents[residents["predicted_risk_grade"].isin(["D - High", "E - Critical"])]["property_value_aed"].sum()
    else:
        high_risk = len(residents[residents["risk_category"].isin(["High", "Critical"])])
        avg_prob = residents["risk_score"].mean()
        at_risk_value = residents[residents["risk_category"].isin(["High", "Critical"])]["property_value_aed"].sum()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("High Risk Residents", f"{high_risk:,}", f"{high_risk/len(residents)*100:.1f}% of total", "down", COLORS["coral"]), unsafe_allow_html=True)
    with c2:
        at_risk_str = f"{at_risk_value/1e9:.1f}B SAR" if at_risk_value >= 1e9 else f"{at_risk_value/1e6:.0f}M SAR"
        st.markdown(kpi_card("At-Risk Portfolio", at_risk_str, f"{at_risk_value/residents['property_value_aed'].sum()*100:.1f}% of total", "down", COLORS["red"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Default Probability", f"{avg_prob:.1f}%", "", "down", COLORS["yellow"]), unsafe_allow_html=True)
    with c4:
        avg_dpd = residents["current_dpd"].mean()
        st.markdown(kpi_card("Avg Days Past Due", f"{avg_dpd:.0f} days", "", "down", COLORS["teal"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Tabs ----
    tab1, tab2, tab3 = st.tabs(["📊 Risk Distribution", "⚠️ Early Warning Alerts", "📋 Risk Portfolio"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Risk Grade Distribution", "📊")
            if has_predictions:
                risk_dist = residents["predicted_risk_grade"].value_counts().sort_index()
                grade_col = "predicted_risk_grade"
            else:
                risk_dist = residents["risk_category"].value_counts().sort_index()
                grade_col = "risk_category"
            
            fig = go.Figure(data=[go.Bar(
                x=risk_dist.index, y=risk_dist.values,
                marker_color=[RISK_COLORS.get(k, COLORS["teal"]) for k in risk_dist.index],
                text=risk_dist.values,
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Risk Grade", yaxis_title="Residents")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Default Probability Distribution", "📈")
            if has_predictions:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=residents["predicted_default_prob"],
                    nbinsx=50,
                    marker_color=COLORS["teal"],
                    opacity=0.8
                ))
                fig.add_vline(x=0.5, line_dash="dash", line_color=COLORS["coral"], 
                             annotation_text="High Risk Threshold")
                fig.update_layout(**PLOT_LAYOUT, height=400, 
                                 xaxis_title="Default Probability", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=residents["risk_score"], nbinsx=50, marker_color=COLORS["teal"]))
                fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Risk Score", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk by Community
        section_header("Risk Heatmap: Community × Zone", "🗺️")
        
        risk_col = "predicted_default_prob" if has_predictions else "risk_score"
        heatmap_data = residents.groupby(["zone", "community"])[risk_col].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index="zone", columns="community", values=risk_col)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale=[[0, "#5B8C64"], [0.3, "#6B8F71"], [0.6, "#D4A843"], [0.8, "#D47643"], [1, "#C1666B"]],
            text=np.round(heatmap_pivot.values * 100, 1) if has_predictions else np.round(heatmap_pivot.values, 1),
            texttemplate="%{text}%",
            textfont=dict(size=11, color=CHART_TEXT_COLOR),
            hovertemplate="Zone: %{y}<br>Community: %{x}<br>Risk: %{z:.3f}<extra></extra>",
            colorbar=dict(title=dict(text="Risk", font=dict(color=CHART_MUTED_COLOR)), tickfont=dict(color=CHART_MUTED_COLOR))
        ))
        fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # DRILL-DOWN: Zone × Community risk detail
        drill_cols_pr = st.columns(2)
        with drill_cols_pr[0]:
            zone_drill = st.selectbox("🔍 Drill into zone", ["Select a zone..."] + sorted(residents["zone"].unique().tolist()), key="risk_zone_drill")
        with drill_cols_pr[1]:
            comm_drill_r = st.selectbox("🔍 Drill into community", ["Select a community..."] + sorted(residents["community"].unique().tolist()), key="risk_comm_drill")
        
        if zone_drill != "Select a zone..." or comm_drill_r != "Select a community...":
            drill_res = residents.copy()
            drill_parts = []
            if zone_drill != "Select a zone...":
                drill_res = drill_res[drill_res["zone"] == zone_drill]
                drill_parts.append(zone_drill)
            if comm_drill_r != "Select a community...":
                drill_res = drill_res[drill_res["community"] == comm_drill_r]
                drill_parts.append(comm_drill_r)
            
            with st.expander(f"🛡️ {' — '.join(drill_parts)} — {len(drill_res):,} residents", expanded=True):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Residents", f"{len(drill_res):,}")
                dc2.metric("Default Rate", f"{drill_res['default_flag'].mean()*100:.1f}%")
                dc3.metric("Avg DPD", f"{drill_res['current_dpd'].mean():.0f} days")
                dc4.metric("Avg Credit", f"{drill_res['credit_score'].mean():.0f}")
                
                if has_predictions:
                    rg = drill_res["predicted_risk_grade"].value_counts().sort_index()
                else:
                    rg = drill_res["risk_category"].value_counts().sort_index()
                fig_rg = go.Figure(data=[go.Bar(
                    x=rg.index, y=rg.values,
                    marker_color=[RISK_COLORS.get(k, COLORS["teal"]) for k in rg.index],
                    text=rg.values, textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                )])
                fig_rg.update_layout(**PLOT_LAYOUT, height=260, xaxis_title="Risk Grade", yaxis_title="Residents")
                st.plotly_chart(fig_rg, use_container_width=True)
    
    with tab2:
        section_header("Early Warning Alerts — High Risk Residents", "⚠️")
        
        if has_predictions:
            all_high_risk = residents[residents["predicted_risk_grade"].isin(["D - High", "E - Critical"])].copy()
        else:
            all_high_risk = residents[residents["risk_category"].isin(["High", "Critical"])].copy()
        
        # Full counts from ALL high-risk residents
        total_high_risk = len(all_high_risk)
        critical = len(all_high_risk[all_high_risk.get("predicted_risk_grade", all_high_risk.get("risk_category", "")) == "E - Critical"]) if has_predictions else len(all_high_risk[all_high_risk["risk_category"] == "Critical"])
        total_balance = all_high_risk["outstanding_balance_aed"].sum()
        
        # Display top 50 sorted for the table
        high_risk_df = all_high_risk.sort_values("risk_score", ascending=False).head(50)
        
        st.markdown(f"""
        <div class="alert-critical">
            <span style="color: #C4515A; font-weight: 700; font-size: 13px; letter-spacing: 0.5px;">⚠ CRITICAL ALERT</span><br>
            <span style="color: var(--text-chart); font-size: 14px; line-height: 1.6;"> {total_high_risk} residents at high/critical default risk ({critical} critical). 
            Combined outstanding balance: {total_balance/1e9:.1f}B SAR. 
            Immediate intervention recommended.</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display table
        display_cols = ["resident_id", "first_name", "last_name", "community", "zone",
                       "property_value_aed", "outstanding_balance_aed", "current_dpd",
                       "credit_score", "risk_score", "payment_consistency_pct"]
        if has_predictions:
            display_cols += ["predicted_default_prob", "predicted_risk_grade"]
        
        available_cols = [c for c in display_cols if c in high_risk_df.columns]
        st.dataframe(high_risk_df[available_cols].head(25), use_container_width=True, hide_index=True)
        
        # Download full list
        download_df = all_high_risk[available_cols].copy()
        download_df.columns = [c.replace("_aed", "_sar").replace("_", " ").title() for c in download_df.columns]
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download All {len(all_high_risk)} High-Risk Residents (CSV)",
            data=csv_data,
            file_name="roshn_high_risk_residents.csv",
            mime="text/csv",
            use_container_width=False,
        )
    
    with tab3:
        section_header("Payment Status Analysis", "💳")
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_dist = payments["payment_status"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=status_dist.index, values=status_dist.values,
                hole=0.5,
                marker_colors=[COLORS["green"], COLORS["teal"], COLORS["yellow"], COLORS["coral"], COLORS["red"]],
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**{**PLOT_LAYOUT, "title": "Payment Status Distribution"}, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payment trends
            payments["month"] = payments["due_date"].dt.to_period("M").astype(str)
            pay_monthly = payments.groupby(["month", "payment_status"]).size().reset_index(name="count")
            pay_monthly = pay_monthly[pay_monthly["month"] >= "2024-06"]
            
            fig = px.bar(pay_monthly, x="month", y="count", color="payment_status",
                        barmode="stack",
                        color_discrete_sequence=[COLORS["green"], COLORS["teal"], COLORS["yellow"], COLORS["coral"], COLORS["red"]])
            fig.update_layout(**{**PLOT_LAYOUT, "title": "Monthly Payment Status"}, height=400,
                            xaxis_title="", yaxis_title="Transactions")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_payment_risk(residents, payments))


# ============================================================================
# PAGE: COMPLAINT INTELLIGENCE
# ============================================================================
def page_complaint_intelligence(residents, complaints):
    """Complaint Trend Analysis and Heatmaps."""
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">Service Analytics</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Complaint Intelligence</h1>
        <p style="color: var(--text-muted);">Complaint heatmaps, category analysis, resolution tracking, and trend detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPIs ----
    total = len(complaints)
    open_count = len(complaints[complaints["status"].isin(["Open", "In Progress"])])
    avg_resolution = complaints["resolution_hours"].dropna().mean()
    escalated = len(complaints[complaints["status"] == "Escalated"])
    avg_rating = complaints["satisfaction_rating"].dropna().mean()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Total Complaints", f"{total:,}", "", "down", COLORS["teal"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Open / In Progress", f"{open_count:,}", f"{open_count/total*100:.1f}%", "down", COLORS["coral"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Resolution", f"{avg_resolution:.0f} hrs", "", "up", COLORS["yellow"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Escalated", f"{escalated:,}", "", "down", COLORS["red"]), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Avg Satisfaction", f"{avg_rating:.1f}/5", "", "up", COLORS["green"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Complaint Heatmap: Zone × Category ----
    section_header("Complaint Heatmap: Zone × Category", "🗺️")
    
    heatmap = complaints.groupby(["zone", "category"]).size().reset_index(name="count")
    heatmap_pivot = heatmap.pivot(index="zone", columns="category", values="count").fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale=[[0, "#FAF8F5"], [0.25, "#E8E0D4"], [0.5, "#B8860B"], [0.75, "#D47643"], [1, "#C1666B"]],
        text=heatmap_pivot.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(size=10, color=CHART_TEXT_COLOR),
        colorbar=dict(title=dict(text="Count", font=dict(color=CHART_MUTED_COLOR)), tickfont=dict(color=CHART_MUTED_COLOR))
    ))
    fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="", yaxis_title="")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # DRILL-DOWN: Zone × Category
    drill_cols = st.columns(2)
    with drill_cols[0]:
        zone_pick = st.selectbox("🔍 Drill into zone", ["Select a zone..."] + sorted(complaints["zone"].unique().tolist()), key="comp_zone_drill")
    with drill_cols[1]:
        cat_pick = st.selectbox("🔍 Drill into category", ["Select a category..."] + sorted(complaints["category"].unique().tolist()), key="comp_cat_drill")
    
    if zone_pick != "Select a zone..." or cat_pick != "Select a category...":
        drill_df = complaints.copy()
        drill_title_parts = []
        if zone_pick != "Select a zone...":
            drill_df = drill_df[drill_df["zone"] == zone_pick]
            drill_title_parts.append(zone_pick)
        if cat_pick != "Select a category...":
            drill_df = drill_df[drill_df["category"] == cat_pick]
            drill_title_parts.append(cat_pick)
        
        with st.expander(f"📋 {' — '.join(drill_title_parts)} — {len(drill_df):,} complaints", expanded=True):
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Total", f"{len(drill_df):,}")
            dc2.metric("Open/In Progress", f"{len(drill_df[drill_df['status'].isin(['Open', 'In Progress'])]):,}")
            dc3.metric("Avg Resolution", f"{drill_df['resolution_hours'].mean():.0f} hrs")
            dc4.metric("Avg Satisfaction", f"{drill_df['satisfaction_rating'].mean():.1f}/5")
            
            sev_drill = drill_df["severity"].value_counts()
            fig_d = go.Figure(data=[go.Bar(
                x=sev_drill.index, y=sev_drill.values,
                marker_color=[SEVERITY_COLORS.get(k, COLORS["teal"]) for k in sev_drill.index],
                text=sev_drill.values, textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig_d.update_layout(**PLOT_LAYOUT, height=250, xaxis_title="Severity", yaxis_title="Count")
            st.plotly_chart(fig_d, use_container_width=True)
            
            # Show recent complaints
            show_cols = [c for c in ["complaint_id", "resident_id", "community", "severity", "status", "resolution_hours", "satisfaction_rating", "created_date"] if c in drill_df.columns]
            st.dataframe(drill_df[show_cols].sort_values("created_date", ascending=False).head(20), use_container_width=True, hide_index=True)
    
    # ---- Row 2 ----
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Top Complaint Categories", "📊")
        cat_counts = complaints["category"].value_counts().head(10)
        fig = go.Figure(data=[go.Bar(
            y=cat_counts.index[::-1], x=cat_counts.values[::-1],
            orientation="h",
            marker_color=COLORS["teal"],
            text=cat_counts.values[::-1],
            textposition="outside",
            textfont=dict(color=CHART_TEXT_COLOR)
        )])
        fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Severity Distribution", "⚡")
        sev_counts = complaints["severity"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=sev_counts.index, values=sev_counts.values,
            hole=0.55,
            marker_colors=[SEVERITY_COLORS.get(k, COLORS["teal"]) for k in sev_counts.index],
            textfont=dict(color=CHART_TEXT_COLOR),
            textinfo="label+percent"
        )])
        fig.update_layout(**PLOT_LAYOUT, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- Complaint Trends ----
    section_header("Monthly Complaint Trends by Severity", "📈")
    
    complaints["month"] = complaints["created_date"].dt.to_period("M").astype(str)
    trend = complaints.groupby(["month", "severity"]).size().reset_index(name="count")
    trend = trend[trend["month"] >= "2024-01"]
    
    fig = px.line(trend, x="month", y="count", color="severity",
                 color_discrete_map=SEVERITY_COLORS,
                 markers=True)
    fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title="", yaxis_title="Complaints",
                     legend_title_text="Severity")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- Resolution Analysis ----
    section_header("Resolution Time by Category", "⏱️")
    
    resolution_by_cat = complaints.groupby("category")["resolution_hours"].agg(["mean", "median"]).reset_index()
    resolution_by_cat.columns = ["Category", "Mean Hours", "Median Hours"]
    resolution_by_cat = resolution_by_cat.sort_values("Mean Hours", ascending=False).head(12)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Mean", y=resolution_by_cat["Category"][::-1], 
        x=resolution_by_cat["Mean Hours"][::-1],
        orientation="h", marker_color=COLORS["coral"]
    ))
    fig.add_trace(go.Bar(
        name="Median", y=resolution_by_cat["Category"][::-1],
        x=resolution_by_cat["Median Hours"][::-1],
        orientation="h", marker_color=COLORS["teal"]
    ))
    fig.update_layout(**PLOT_LAYOUT, height=450, barmode="group",
                     xaxis_title="Hours", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_complaints(complaints))


# ============================================================================
# PAGE: SENTIMENT & SATISFACTION
# ============================================================================
def page_sentiment_satisfaction(residents, interactions, complaints):
    """Sentiment Tracking and Satisfaction Analysis."""
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">Resident Experience</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Sentiment & Satisfaction</h1>
        <p style="color: var(--text-muted);">Resident sentiment trends, satisfaction scoring, and NPS analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPIs ----
    avg_sentiment = interactions["sentiment_score"].mean()
    positive_pct = (interactions["sentiment_score"] > 0.2).mean() * 100
    negative_pct = (interactions["sentiment_score"] < -0.2).mean() * 100
    avg_csat = interactions["csat_score"].mean()
    avg_satisfaction = residents["satisfaction_score"].mean() if "satisfaction_score" in residents.columns else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Avg Sentiment", f"{avg_sentiment:.2f}", "Scale: -1 to 1", "up", COLORS["teal"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Positive Interactions", f"{positive_pct:.1f}%", "", "up", COLORS["green"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Negative Interactions", f"{negative_pct:.1f}%", "", "down", COLORS["coral"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Avg CSAT Score", f"{avg_csat:.1f}/5", "", "up", COLORS["blue"]), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Satisfaction Index", f"{avg_satisfaction:.1f}", "Out of 100", "up", COLORS["purple"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Sentiment Trends ----
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Sentiment Trend Over Time", "📈")
        interactions["month"] = interactions["timestamp"].dt.to_period("M").astype(str)
        monthly = interactions.groupby("month").agg(
            avg_sentiment=("sentiment_score", "mean"),
            positive_pct=("sentiment_score", lambda x: (x > 0.2).mean() * 100),
            volume=("interaction_id", "count")
        ).reset_index()
        monthly = monthly.tail(18)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["avg_sentiment"],
            mode="lines+markers", name="Avg Sentiment",
            line=dict(color=COLORS["teal"], width=3),
            marker=dict(size=8)
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["volume"],
            name="Volume", marker_color=COLORS["navy"], opacity=0.4
        ), secondary_y=True)
        fig.update_layout(**{**PLOT_LAYOUT, "legend": dict(x=0, y=1.1, orientation="h", bgcolor="rgba(0,0,0,0)", font=dict(color=CHART_MUTED_COLOR))}, height=400)
        fig.update_yaxes(title_text="Sentiment", secondary_y=False, gridcolor=PLOT_LAYOUT["yaxis"]["gridcolor"] if IS_DARK else "#F0EBE3")
        fig.update_yaxes(title_text="Volume", secondary_y=True, gridcolor=PLOT_LAYOUT["yaxis"]["gridcolor"] if IS_DARK else "#F0EBE3")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Sentiment by Channel", "📡")
        channel_sent = interactions.groupby("channel")["sentiment_score"].agg(["mean", "count"]).reset_index()
        channel_sent.columns = ["Channel", "Avg Sentiment", "Volume"]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=channel_sent["Channel"], y=channel_sent["Avg Sentiment"],
            marker_color=[COLORS["teal"] if v > 0.2 else COLORS["yellow"] if v > 0 else COLORS["coral"] 
                         for v in channel_sent["Avg Sentiment"]],
            text=[f"{v:.2f}" for v in channel_sent["Avg Sentiment"]],
            textposition="outside",
            textfont=dict(color=CHART_TEXT_COLOR)
        ))
        fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="", yaxis_title="Avg Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- Satisfaction by Community ----
    section_header("Satisfaction Score by Community", "🏘️")
    
    comm_sat = residents.groupby("community").agg(
        avg_satisfaction=("satisfaction_score", "mean"),
        median_satisfaction=("satisfaction_score", "median"),
        count=("resident_id", "count")
    ).reset_index().sort_values("avg_satisfaction", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=comm_sat["community"], x=comm_sat["avg_satisfaction"],
        orientation="h",
        marker_color=[COLORS["green"] if v > 75 else COLORS["teal"] if v > 65 else COLORS["yellow"] if v > 55 else COLORS["coral"]
                     for v in comm_sat["avg_satisfaction"]],
        text=[f"{v:.1f}" for v in comm_sat["avg_satisfaction"]],
        textposition="outside",
        textfont=dict(color=CHART_TEXT_COLOR)
    ))
    fig.update_layout(**PLOT_LAYOUT, height=420, xaxis_title="Satisfaction Score (0-100)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    # DRILL-DOWN: Community satisfaction deep-dive
    comm_drill = st.selectbox("🔍 Drill into community sentiment", ["Select a community..."] + sorted(residents["community"].unique().tolist()), key="sent_comm_drill")
    if comm_drill != "Select a community...":
        comm_res = residents[residents["community"] == comm_drill]
        comm_int = interactions[interactions["resident_id"].isin(comm_res["resident_id"])] if "resident_id" in interactions.columns else interactions
        comm_comp = complaints[complaints["community"] == comm_drill] if "community" in complaints.columns else complaints
        
        with st.expander(f"🏘️ {comm_drill} — Sentiment Deep-Dive", expanded=True):
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Residents", f"{len(comm_res):,}")
            dc2.metric("Satisfaction", f"{comm_res['satisfaction_score'].mean():.1f}/100")
            dc3.metric("Avg Sentiment", f"{comm_int['sentiment_score'].mean():.2f}" if len(comm_int) > 0 else "N/A")
            dc4.metric("Open Complaints", f"{len(comm_comp[comm_comp['status'].isin(['Open', 'In Progress'])]):,}" if len(comm_comp) > 0 else "0")
            
            if len(comm_int) > 0:
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    # Sentiment by purpose for this community
                    purp_sent = comm_int.groupby("purpose")["sentiment_score"].mean().sort_values()
                    fig_p = go.Figure(data=[go.Bar(
                        y=purp_sent.index, x=purp_sent.values, orientation="h",
                        marker_color=[COLORS["green"] if v > 0.2 else COLORS["yellow"] if v > 0 else COLORS["coral"] for v in purp_sent.values],
                        text=[f"{v:.2f}" for v in purp_sent.values], textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                    )])
                    fig_p.update_layout({**PLOT_LAYOUT, "title": "Sentiment by Purpose"}, height=280, xaxis_title="Sentiment")
                    st.plotly_chart(fig_p, use_container_width=True)
                with dcol2:
                    # CSAT distribution for this community
                    csat_d = comm_int["csat_score"].value_counts().sort_index()
                    fig_c = go.Figure(data=[go.Bar(
                        x=csat_d.index, y=csat_d.values,
                        marker_color=[COLORS["red"], COLORS["coral"], COLORS["yellow"], COLORS["teal"], COLORS["green"]][:len(csat_d)],
                        text=csat_d.values, textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                    )])
                    fig_c.update_layout({**PLOT_LAYOUT, "title": "CSAT Distribution"}, height=280, xaxis_title="CSAT Score", yaxis_title="Count")
                    st.plotly_chart(fig_c, use_container_width=True)
    
    # ---- Sentiment by Purpose ----
    section_header("Sentiment by Interaction Purpose", "🎯")
    
    purpose_sent = interactions.groupby("purpose")["sentiment_score"].agg(["mean", "count"]).reset_index()
    purpose_sent.columns = ["Purpose", "Avg Sentiment", "Count"]
    purpose_sent = purpose_sent.sort_values("Avg Sentiment", ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=purpose_sent["Purpose"], x=purpose_sent["Avg Sentiment"],
        orientation="h",
        marker_color=[COLORS["green"] if v > 0.3 else COLORS["teal"] if v > 0.1 else COLORS["yellow"] if v > -0.1 else COLORS["coral"]
                     for v in purpose_sent["Avg Sentiment"]],
        text=[f"{v:.2f}" for v in purpose_sent["Avg Sentiment"]],
        textposition="outside",
        textfont=dict(color=CHART_TEXT_COLOR)
    ))
    fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Avg Sentiment", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_sentiment(residents, interactions))


# ============================================================================
# PAGE: LEAD INTELLIGENCE
# ============================================================================
def page_lead_intelligence(leads):
    """Lead Management AI Intelligence Dashboard."""
    
    if leads.empty:
        st.warning("Lead data not found. Please generate roshn_leads.csv first.")
        return
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">Lead Management AI</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Lead Intelligence & Conversion Analytics</h1>
        <p style="color: var(--text-muted);">AI-powered lead scoring, funnel analytics, source performance, and agent productivity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPIs ----
    total_leads = len(leads)
    won = len(leads[leads["stage"] == "Won"])
    lost = len(leads[leads["stage"] == "Lost"])
    active = total_leads - won - lost
    conversion_rate = won / total_leads * 100
    closed_value = leads[leads["stage"] == "Won"]["conversion_value_sar"].sum()
    avg_response = leads["response_time_hours"].mean()
    ai_pct = leads["ai_assisted"].mean() * 100
    avg_score = leads["lead_score"].mean()
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card("Total Leads", f"{total_leads:,}", "24-month pipeline", "up", COLORS["teal"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Conversion Rate", f"{conversion_rate:.1f}%", f"{won:,} won", "up", COLORS["green"]), unsafe_allow_html=True)
    with c3:
        closed_str = f"{closed_value/1e9:.1f}B SAR" if closed_value >= 1e9 else f"{closed_value/1e6:.0f}M SAR"
        st.markdown(kpi_card("Closed Revenue", closed_str, f"{won:,} deals closed", "up", COLORS["gold"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Active Leads", f"{active:,}", f"{lost:,} lost", "up", COLORS["blue"]), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Avg Response", f"{avg_response:.1f} hrs", "", "up", COLORS["orange"]), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("AI-Assisted", f"{ai_pct:.0f}%", f"Avg Score: {avg_score:.0f}", "up", COLORS["purple"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Tab Layout ----
    has_predictions = "conversion_probability" in leads.columns
    
    if has_predictions:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Funnel & Pipeline", "🎯 Lead Scoring & Sources", "👥 Agent Performance", "🔥 Hot Leads (ML)"])
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Funnel & Pipeline", "🎯 Lead Scoring & Sources", "👥 Agent Performance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Conversion Funnel", "📊")
            # Build a proper funnel: show cumulative leads that reached each stage
            stage_order = ["New", "Contacted", "Qualified", "Site Visit", "Negotiation", "Proposal Sent", "Won"]
            stage_map = {s: i for i, s in enumerate(stage_order)}
            
            # Each lead has reached its current stage AND all stages before it
            # So "Qualified" leads have also been through "New" and "Contacted"
            leads_copy = leads.copy()
            leads_copy["stage_idx"] = leads_copy["stage"].map(stage_map)
            # Lost leads — assign their last active stage index
            lost_mask = leads_copy["stage"] == "Lost"
            leads_copy.loc[lost_mask, "stage_idx"] = leads_copy.loc[lost_mask, "total_interactions"].clip(0, 5)
            
            funnel_counts = []
            for stage in stage_order:
                idx = stage_map[stage]
                count = (leads_copy["stage_idx"] >= idx).sum()
                funnel_counts.append(count)
            
            fig = go.Figure(go.Funnel(
                y=stage_order,
                x=funnel_counts,
                textinfo="value+percent initial",
                textfont=dict(color="#FFFFFF", size=13),
                marker=dict(color=[COLORS["teal"], COLORS["blue"], COLORS["gold"],
                                  COLORS["orange"], COLORS["purple"], COLORS["coral"], COLORS["green"]]),
                connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1))
            ))
            fig.update_layout(**PLOT_LAYOUT, height=420)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Monthly Lead Volume & Conversions", "📈")
            leads["month"] = leads["created_date_dt"].dt.to_period("M").astype(str)
            monthly = leads.groupby("month").agg(
                total=("lead_id", "count"),
                won=("stage", lambda x: (x == "Won").sum()),
            ).reset_index()
            monthly = monthly.tail(18)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly["month"], y=monthly["total"],
                name="Total Leads", marker_color=COLORS["blue"], opacity=0.5
            ))
            fig.add_trace(go.Bar(
                x=monthly["month"], y=monthly["won"],
                name="Won", marker_color=COLORS["green"]
            ))
            fig.update_layout(**PLOT_LAYOUT, height=420, barmode="overlay",
                            xaxis_title="", yaxis_title="Leads")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Pipeline by Community
        section_header("Pipeline by Community Interest", "🏘️")
        comm_pipeline = leads[~leads["stage"].isin(["Won", "Lost"])].groupby("community_interest").agg(
            active_leads=("lead_id", "count"),
            avg_score=("lead_score", "mean"),
        ).reset_index().sort_values("active_leads", ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=comm_pipeline["community_interest"], x=comm_pipeline["active_leads"],
            orientation="h",
            marker_color=COLORS["gold"],
            text=comm_pipeline["active_leads"].astype(int),
            textposition="outside",
            textfont=dict(color=CHART_TEXT_COLOR)
        ))
        fig.update_layout(**PLOT_LAYOUT, height=420, xaxis_title="Active Leads", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # DRILL-DOWN: Community pipeline detail
        comm_pick = st.selectbox("🔍 Drill into community pipeline", ["Select a community..."] + sorted(leads["community_interest"].unique().tolist()), key="lead_comm_drill")
        if comm_pick != "Select a community...":
            comm_leads = leads[leads["community_interest"] == comm_pick]
            active_cl = comm_leads[~comm_leads["stage"].isin(["Won", "Lost"])]
            won_cl = comm_leads[comm_leads["stage"] == "Won"]
            
            with st.expander(f"🏘️ {comm_pick} — {len(comm_leads):,} leads ({len(active_cl):,} active)", expanded=True):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Total Leads", f"{len(comm_leads):,}")
                dc2.metric("Won", f"{len(won_cl):,}")
                dc3.metric("Conv Rate", f"{len(won_cl)/max(len(comm_leads),1)*100:.1f}%")
                dc4.metric("Avg Score", f"{comm_leads['lead_score'].mean():.0f}")
                
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    stage_d = comm_leads["stage"].value_counts()
                    fig_s = go.Figure(data=[go.Bar(x=stage_d.index, y=stage_d.values,
                        marker_color=COLORS["gold"], text=stage_d.values,
                        textposition="outside", textfont=dict(color=CHART_TEXT_COLOR))])
                    fig_s.update_layout({**PLOT_LAYOUT, "title": "Stage Distribution"}, height=260)
                    st.plotly_chart(fig_s, use_container_width=True)
                with dcol2:
                    src_d = comm_leads["source"].value_counts()
                    fig_sr = go.Figure(data=[go.Bar(x=src_d.index, y=src_d.values,
                        marker_color=COLORS["blue"], text=src_d.values,
                        textposition="outside", textfont=dict(color=CHART_TEXT_COLOR))])
                    fig_sr.update_layout({**PLOT_LAYOUT, "title": "Lead Sources"}, height=260)
                    st.plotly_chart(fig_sr, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Lead Score Distribution", "🎯")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=leads["lead_score"], nbinsx=30,
                marker_color=COLORS["gold"], opacity=0.85
            ))
            fig.add_vline(x=70, line_dash="dash", line_color=COLORS["green"],
                         annotation_text="High Quality", annotation_font_color=CHART_TEXT_COLOR)
            fig.add_vline(x=40, line_dash="dash", line_color=COLORS["coral"],
                         annotation_text="Low Quality", annotation_font_color=CHART_TEXT_COLOR)
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Lead Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Conversion Rate by Source", "📡")
            source_conv = leads.groupby("source").agg(
                total=("lead_id", "count"),
                won=("stage", lambda x: (x == "Won").sum()),
            ).reset_index()
            source_conv["conv_rate"] = (source_conv["won"] / source_conv["total"] * 100).round(1)
            source_conv = source_conv.sort_values("conv_rate", ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=source_conv["source"], x=source_conv["conv_rate"],
                orientation="h",
                marker_color=[COLORS["green"] if r > 12 else COLORS["gold"] if r > 8 else COLORS["coral"]
                             for r in source_conv["conv_rate"]],
                text=[f"{v:.1f}%" for v in source_conv["conv_rate"]],
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            ))
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Conversion %", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Source Volume & Quality
        section_header("Lead Source: Volume vs Quality", "⚡")
        source_quality = leads.groupby("source").agg(
            count=("lead_id", "count"),
            avg_score=("lead_score", "mean"),
            avg_response=("response_time_hours", "mean"),
            conv_rate=("stage", lambda x: (x == "Won").mean() * 100),
            total_value=("conversion_value_sar", "sum"),
        ).reset_index()
        source_quality["avg_score"] = source_quality["avg_score"].round(1)
        source_quality["avg_response"] = source_quality["avg_response"].round(1)
        source_quality["conv_rate"] = source_quality["conv_rate"].round(1)
        source_quality["total_value"] = (source_quality["total_value"] / 1e6).round(1)
        source_quality.columns = ["Source", "Leads", "Avg Score", "Avg Response (hrs)", "Conv Rate %", "Revenue (M SAR)"]
        source_quality = source_quality.sort_values("Leads", ascending=False)
        
        st.dataframe(source_quality, use_container_width=True, hide_index=True)
        
        # Budget & Property Interest
        col1, col2 = st.columns(2)
        with col1:
            section_header("Demand by Property Type", "🏠")
            prop_demand = leads["property_type_interest"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=prop_demand.index, values=prop_demand.values,
                hole=0.5,
                marker_colors=[COLORS["gold"], COLORS["teal"], COLORS["blue"], COLORS["purple"], COLORS["orange"]],
                textinfo="label+percent",
                textfont=dict(color=CHART_TEXT_COLOR, size=11)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Budget Range Distribution", "💰")
            budget_order = ["500K-1M", "1M-2M", "2M-3M", "3M-5M", "5M-10M", "10M+"]
            budget_dist = leads["budget_range"].value_counts().reindex(budget_order).fillna(0)
            fig = go.Figure(data=[go.Bar(
                x=budget_dist.index, y=budget_dist.values,
                marker_color=[COLORS["teal"], COLORS["blue"], COLORS["gold"],
                             COLORS["orange"], COLORS["purple"], COLORS["coral"]],
                text=budget_dist.values.astype(int),
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Budget (SAR)", yaxis_title="Leads")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        section_header("Agent Performance Scorecard", "👥")
        
        agent_perf = leads.groupby("assigned_agent").agg(
            total_leads=("lead_id", "count"),
            won=("stage", lambda x: (x == "Won").sum()),
            lost=("stage", lambda x: (x == "Lost").sum()),
            avg_score=("lead_score", "mean"),
            avg_response=("response_time_hours", "mean"),
            revenue=("conversion_value_sar", "sum"),
        ).reset_index()
        agent_perf["conv_rate"] = (agent_perf["won"] / agent_perf["total_leads"] * 100).round(1)
        agent_perf["avg_score"] = agent_perf["avg_score"].round(0).astype(int)
        agent_perf["avg_response"] = agent_perf["avg_response"].round(1)
        agent_perf["revenue"] = (agent_perf["revenue"] / 1e6).round(1)
        agent_perf = agent_perf.sort_values("conv_rate", ascending=False)
        agent_perf.columns = ["Agent", "Total Leads", "Won", "Lost", "Avg Score", "Avg Response (hrs)", "Revenue (M SAR)", "Conv Rate %"]
        
        st.dataframe(agent_perf, use_container_width=True, hide_index=True)
        
        # Agent comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Conversion Rate by Agent", "📊")
            fig = go.Figure(data=[go.Bar(
                x=agent_perf["Agent"], y=agent_perf["Conv Rate %"],
                marker_color=[COLORS["green"] if r > 11 else COLORS["gold"] if r > 9 else COLORS["coral"]
                             for r in agent_perf["Conv Rate %"]],
                text=[f"{v}%" for v in agent_perf["Conv Rate %"]],
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="", yaxis_title="Conversion %")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Revenue by Agent (M SAR)", "💰")
            fig = go.Figure(data=[go.Bar(
                x=agent_perf["Agent"], y=agent_perf["Revenue (M SAR)"],
                marker_color=COLORS["gold"],
                text=[f"{v}M" for v in agent_perf["Revenue (M SAR)"]],
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="", yaxis_title="Revenue (M SAR)")
            st.plotly_chart(fig, use_container_width=True)
        
        # AI vs Non-AI comparison
        section_header("AI-Assisted vs Manual Lead Handling", "🤖")
        col1, col2, col3 = st.columns(3)
        
        ai_leads = leads[leads["ai_assisted"] == True]
        manual_leads = leads[leads["ai_assisted"] == False]
        
        with col1:
            ai_conv = (ai_leads["stage"] == "Won").mean() * 100
            man_conv = (manual_leads["stage"] == "Won").mean() * 100
            st.markdown(kpi_card("AI Conv. Rate", f"{ai_conv:.1f}%", f"Manual: {man_conv:.1f}%", "up", COLORS["green"]), unsafe_allow_html=True)
        with col2:
            ai_resp = ai_leads["response_time_hours"].mean()
            man_resp = manual_leads["response_time_hours"].mean()
            st.markdown(kpi_card("AI Avg Response", f"{ai_resp:.1f} hrs", f"Manual: {man_resp:.1f} hrs", "up", COLORS["blue"]), unsafe_allow_html=True)
        with col3:
            ai_score = ai_leads["lead_score"].mean()
            man_score = manual_leads["lead_score"].mean()
            st.markdown(kpi_card("AI Avg Lead Score", f"{ai_score:.0f}", f"Manual: {man_score:.0f}", "up", COLORS["purple"]), unsafe_allow_html=True)

    # ---- Tab 4: Hot Leads (ML Predictions) ----
    if has_predictions:
        with tab4:
            active_leads = leads[~leads["stage"].isin(["Won", "Lost"])].copy()
            
            if "lead_priority" in active_leads.columns:
                hot = active_leads[active_leads["lead_priority"] == "Hot"]
                warm = active_leads[active_leads["lead_priority"] == "Warm"]
                cool = active_leads[active_leads["lead_priority"] == "Cool"]
                cold = active_leads[active_leads["lead_priority"] == "Cold"]
            else:
                hot = active_leads[active_leads["conversion_probability"] >= 0.75]
                warm = active_leads[(active_leads["conversion_probability"] >= 0.50) & (active_leads["conversion_probability"] < 0.75)]
                cool = active_leads[(active_leads["conversion_probability"] >= 0.25) & (active_leads["conversion_probability"] < 0.50)]
                cold = active_leads[active_leads["conversion_probability"] < 0.25]
            
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(kpi_card("🔥 Hot Leads", f"{len(hot):,}", ">75% conversion prob", "up", "#C1666B"), unsafe_allow_html=True)
            with c2:
                st.markdown(kpi_card("🟡 Warm Leads", f"{len(warm):,}", "50-75% probability", "up", COLORS["yellow"]), unsafe_allow_html=True)
            with c3:
                st.markdown(kpi_card("🔵 Cool Leads", f"{len(cool):,}", "25-50% probability", "up", COLORS["blue"]), unsafe_allow_html=True)
            with c4:
                st.markdown(kpi_card("⚪ Cold Leads", f"{len(cold):,}", "<25% probability", "up", COLORS["taupe"]), unsafe_allow_html=True)
            
            st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
            
            # Alert - estimate pipeline value from budget range (active leads have 0 conversion_value)
            budget_midpoints = {"500K-1M": 750000, "1M-2M": 1500000, "2M-3M": 2500000, 
                               "3M-5M": 4000000, "5M-10M": 7500000, "10M+": 15000000}
            hot_est_value = hot["budget_range"].map(budget_midpoints).sum()
            hot_est_str = f"{hot_est_value/1e9:.1f}B" if hot_est_value >= 1e9 else f"{hot_est_value/1e6:.0f}M"
            
            st.markdown(f"""
            <div class="alert-critical" style="border-left-color: #C4515A;">
                <span style="color: #C4515A; font-weight: 700; font-size: 13px; letter-spacing: 0.5px;">🔥 ML PREDICTION</span><br>
                <span style="color: var(--alert-text); font-size: 14px; line-height: 1.6;">
                {len(hot)} hot leads identified with &gt;75% predicted conversion probability.
                Estimated pipeline value: {hot_est_str} SAR based on budget interest.
                Prioritize immediate follow-up with assigned agents.</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                section_header("Conversion Probability Distribution", "📊")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=active_leads["conversion_probability"], nbinsx=40,
                    marker_color=COLORS["gold"], opacity=0.85
                ))
                fig.add_vline(x=0.75, line_dash="dash", line_color="#C1666B",
                             annotation_text="Hot", annotation_font_color=CHART_TEXT_COLOR)
                fig.add_vline(x=0.50, line_dash="dash", line_color=COLORS["yellow"],
                             annotation_text="Warm", annotation_font_color=CHART_TEXT_COLOR)
                fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Conversion Probability", yaxis_title="Leads")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                section_header("Priority by Source", "📡")
                if "lead_priority" in active_leads.columns:
                    priority_source = active_leads.groupby(["source", "lead_priority"]).size().reset_index(name="count")
                    priority_colors = {"Hot": "#C1666B", "Warm": "#D4A843", "Cool": "#5E7F9A", "Cold": "#9A8C7A"}
                    fig = px.bar(priority_source, x="source", y="count", color="lead_priority",
                                color_discrete_map=priority_colors, barmode="stack")
                    fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="", yaxis_title="Leads")
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Hot Leads Table
            section_header("Top Priority Leads — Immediate Action Required", "🎯")
            
            hot_display = active_leads.sort_values("conversion_probability", ascending=False).head(30)
            display_cols = ["lead_id", "source", "community_interest", "property_type_interest",
                           "budget_range", "lead_score", "total_interactions", "response_time_hours",
                           "assigned_agent", "conversion_probability"]
            if "lead_priority" in hot_display.columns:
                display_cols.append("lead_priority")
            
            available = [c for c in display_cols if c in hot_display.columns]
            show_df = hot_display[available].copy()
            if "conversion_probability" in show_df.columns:
                show_df["conversion_probability"] = (show_df["conversion_probability"] * 100).round(1)
                show_df = show_df.rename(columns={"conversion_probability": "Conv Prob %"})
            
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            
            # Download full scored active leads
            download_leads = active_leads.sort_values("conversion_probability", ascending=False).copy()
            dl_cols = ["lead_id", "source", "community_interest", "zone", "property_type_interest",
                      "budget_range", "nationality", "lead_score", "stage", "total_interactions",
                      "response_time_hours", "ai_assisted", "assigned_agent",
                      "conversion_probability", "lead_priority", "follow_up_scheduled"]
            dl_available = [c for c in dl_cols if c in download_leads.columns]
            dl_df = download_leads[dl_available].copy()
            if "conversion_probability" in dl_df.columns:
                dl_df["conversion_probability"] = (dl_df["conversion_probability"] * 100).round(2)
                dl_df = dl_df.rename(columns={"conversion_probability": "Conversion Prob %"})
            dl_df.columns = [c.replace("_", " ").title() for c in dl_df.columns]
            csv_leads = dl_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download All {len(active_leads)} Scored Active Leads (CSV)",
                data=csv_leads,
                file_name="roshn_hot_leads_ml_predictions.csv",
                mime="text/csv",
                use_container_width=False,
            )
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_leads(leads))


# ============================================================================
# PAGE: DEMAND FORECASTING
# ============================================================================
def page_demand_forecasting(bookings, complaints, interactions):
    """Demand Forecasting and Resource Optimization."""
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">Operations Planning</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Community Amenity Demand & Resource Planning</h1>
        <p style="color: var(--text-muted);">Facility booking patterns by community, peak usage analysis, and resource optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- KPIs ----
    total_bookings = len(bookings)
    completed = len(bookings[bookings["status"] == "Completed"])
    no_show_rate = len(bookings[bookings["status"] == "No-Show"]) / total_bookings * 100
    top_facility = bookings["facility"].value_counts().index[0]
    unique_facilities = bookings["facility"].nunique()
    avg_daily = total_bookings / max(bookings["booking_date"].dt.date.nunique(), 1)
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card("Total Bookings", f"{total_bookings:,}", "", "up", COLORS["teal"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Completion Rate", f"{completed/total_bookings*100:.1f}%", "", "up", COLORS["green"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("No-Show Rate", f"{no_show_rate:.1f}%", "", "down", COLORS["coral"]), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Top Facility", top_facility, "", "up", COLORS["blue"]), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Unique Amenities", f"{unique_facilities}", "", "up", COLORS["gold"]), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("Avg Daily Bookings", f"{avg_daily:.0f}", "", "up", COLORS["purple"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- TABS ----
    tab1, tab2, tab3, tab4 = st.tabs(["🏘️ Community Demand", "🏋️ Facility Rankings", "⏰ Peak Patterns", "🔧 Maintenance Forecast"])
    
    # ---- TAB 1: Community-Level Demand ----
    with tab1:
        section_header("Bookings by Community", "🏘️")
        
        comm_demand = bookings.groupby("community").agg(
            bookings=("booking_id", "count") if "booking_id" in bookings.columns else ("facility", "count"),
            unique_facilities=("facility", "nunique"),
            completion_rate=("status", lambda x: (x == "Completed").mean() * 100),
            no_show_rate=("status", lambda x: (x == "No-Show").mean() * 100),
        ).reset_index().sort_values("bookings", ascending=True)
        
        fig = go.Figure(data=[go.Bar(
            y=comm_demand["community"], x=comm_demand["bookings"],
            orientation="h",
            marker_color=COLORS["teal"],
            text=comm_demand["bookings"].astype(int),
            textposition="outside",
            textfont=dict(color=CHART_TEXT_COLOR)
        )])
        fig.update_layout(**PLOT_LAYOUT, height=420, xaxis_title="Total Bookings", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # DRILL-DOWN: Community amenity breakdown
        comm_pick = st.selectbox("🔍 Drill into community", ["Select a community..."] + sorted(bookings["community"].unique().tolist()), key="demand_comm_drill")
        if comm_pick != "Select a community...":
            comm_bk = bookings[bookings["community"] == comm_pick]
            
            with st.expander(f"🏘️ {comm_pick} — {len(comm_bk):,} bookings", expanded=True):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Total Bookings", f"{len(comm_bk):,}")
                dc2.metric("Completion", f"{(comm_bk['status']=='Completed').mean()*100:.1f}%")
                dc3.metric("No-Show", f"{(comm_bk['status']=='No-Show').mean()*100:.1f}%")
                dc4.metric("Amenities Used", f"{comm_bk['facility'].nunique()}")
                
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    fac_d = comm_bk["facility"].value_counts().head(10)
                    fig_f = go.Figure(data=[go.Bar(
                        y=fac_d.index[::-1], x=fac_d.values[::-1], orientation="h",
                        marker_color=COLORS["teal"], text=fac_d.values[::-1],
                        textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                    )])
                    fig_f.update_layout(**{**PLOT_LAYOUT, "title": f"Top Amenities — {comm_pick}"}, height=320, xaxis_title="Bookings")
                    st.plotly_chart(fig_f, use_container_width=True)
                with dcol2:
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    day_d = comm_bk["day_of_week"].value_counts().reindex(day_order).fillna(0)
                    fig_dy = go.Figure(data=[go.Bar(
                        x=day_d.index, y=day_d.values,
                        marker_color=[COLORS["gold"] if d in ["Friday", "Saturday"] else COLORS["teal"] for d in day_d.index],
                        text=day_d.values.astype(int), textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                    )])
                    fig_dy.update_layout(**{**PLOT_LAYOUT, "title": f"Day-of-Week — {comm_pick}"}, height=320, yaxis_title="Bookings")
                    st.plotly_chart(fig_dy, use_container_width=True)
        
        # Community × Facility Heatmap
        section_header("Community × Facility Demand Heatmap", "🔥")
        comm_fac = bookings.groupby(["community", "facility"]).size().reset_index(name="count")
        comm_fac_pivot = comm_fac.pivot(index="community", columns="facility", values="count").fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=comm_fac_pivot.values,
            x=comm_fac_pivot.columns,
            y=comm_fac_pivot.index,
            colorscale=[[0, "#FAF8F5"], [0.25, "#E8E0D4"], [0.5, "#B8860B"], [0.75, "#C96B30"], [1, "#C4515A"]],
            text=comm_fac_pivot.values.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=9, color=CHART_TEXT_COLOR),
            colorbar=dict(title=dict(text="Bookings", font=dict(color=CHART_MUTED_COLOR)), tickfont=dict(color=CHART_MUTED_COLOR))
        ))
        fig.update_layout(**PLOT_LAYOUT, height=450, xaxis_title="", yaxis_title="")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- TAB 2: Facility Rankings ----
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            section_header("Facility Demand Ranking", "🏋️")
            fac_counts = bookings["facility"].value_counts().head(12)
            fig = go.Figure(data=[go.Bar(
                y=fac_counts.index[::-1], x=fac_counts.values[::-1],
                orientation="h",
                marker_color=COLORS["teal"],
                text=fac_counts.values[::-1],
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=450, xaxis_title="Bookings", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Facility Completion vs No-Show", "📊")
            fac_status = bookings.groupby("facility")["status"].value_counts(normalize=True).unstack(fill_value=0) * 100
            fac_status = fac_status.sort_values("Completed" if "Completed" in fac_status.columns else fac_status.columns[0], ascending=True).tail(12)
            
            fig = go.Figure()
            for status_col, color in [("Completed", COLORS["green"]), ("No-Show", COLORS["coral"]), ("Cancelled", COLORS["yellow"])]:
                if status_col in fac_status.columns:
                    fig.add_trace(go.Bar(
                        y=fac_status.index, x=fac_status[status_col],
                        name=status_col, orientation="h", marker_color=color
                    ))
            fig.update_layout(**{**PLOT_LAYOUT, "legend": dict(x=0, y=1.1, orientation="h", bgcolor="rgba(0,0,0,0)", font=dict(color=CHART_MUTED_COLOR))}, height=450, barmode="stack", xaxis_title="% of Bookings", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Facility drill-down
        fac_pick = st.selectbox("🔍 Drill into facility", ["Select a facility..."] + sorted(bookings["facility"].unique().tolist()), key="demand_fac_drill")
        if fac_pick != "Select a facility...":
            fac_bk = bookings[bookings["facility"] == fac_pick]
            with st.expander(f"🏋️ {fac_pick} — {len(fac_bk):,} bookings", expanded=True):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Total", f"{len(fac_bk):,}")
                dc2.metric("Completion", f"{(fac_bk['status']=='Completed').mean()*100:.1f}%")
                dc3.metric("No-Show", f"{(fac_bk['status']=='No-Show').mean()*100:.1f}%")
                dc4.metric("Communities", f"{fac_bk['community'].nunique()}")
                
                # Which communities use this facility most
                comm_usage = fac_bk["community"].value_counts()
                fig_cu = go.Figure(data=[go.Bar(
                    x=comm_usage.index, y=comm_usage.values,
                    marker_color=COLORS["gold"], text=comm_usage.values,
                    textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
                )])
                fig_cu.update_layout(**{**PLOT_LAYOUT, "title": f"{fac_pick} — Usage by Community"}, height=280, yaxis_title="Bookings")
                fig_cu.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_cu, use_container_width=True)
    
    # ---- TAB 3: Peak Patterns ----
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            section_header("Booking by Day of Week", "📅")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_counts = bookings["day_of_week"].value_counts().reindex(day_order)
            
            fig = go.Figure(data=[go.Bar(
                x=day_counts.index, y=day_counts.values,
                marker_color=[COLORS["gold"] if d in ["Friday", "Saturday"] else COLORS["teal"] for d in day_counts.index],
                text=day_counts.values,
                textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="", yaxis_title="Bookings")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("Hourly Demand Pattern", "⏰")
            bookings["hour"] = bookings["time_slot"].str[:2].astype(int)
            hourly = bookings.groupby("hour").size().reset_index(name="count")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["count"],
                mode="lines+markers+text",
                line=dict(color=COLORS["teal"], width=3, shape="spline"),
                marker=dict(size=10, color=COLORS["teal"]),
                fill="tozeroy",
                fillcolor="rgba(61, 139, 94, 0.12)",
                text=hourly["count"],
                textposition="top center",
                textfont=dict(color=CHART_TEXT_COLOR, size=10)
            ))
            fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(dtick=1, **{k: v for k, v in PLOT_LAYOUT.get("xaxis", {}).items() if k in ["gridcolor", "zerolinecolor", "linecolor"]})}, height=400,
                             xaxis_title="Hour of Day", yaxis_title="Bookings")
            peak_hour = hourly.loc[hourly["count"].idxmax(), "hour"]
            fig.add_annotation(x=peak_hour, y=hourly["count"].max(),
                              text="PEAK", showarrow=True, arrowhead=2,
                              font=dict(color=COLORS["coral"], size=14, family="Lato"),
                              arrowcolor=COLORS["coral"])
            st.plotly_chart(fig, use_container_width=True)
        
        # Day × Hour heatmap
        section_header("Usage Heatmap: Day × Hour", "🔥")
        day_order_h = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        usage_heatmap = bookings.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
        usage_pivot = usage_heatmap.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
        usage_pivot = usage_pivot.reindex(day_order_h)
        
        fig = go.Figure(data=go.Heatmap(
            z=usage_pivot.values,
            x=[f"{h}:00" for h in usage_pivot.columns],
            y=usage_pivot.index,
            colorscale=[[0, "#FAF8F5"], [0.3, "#E8E0D4"], [0.6, "#B8860B"], [0.8, "#C96B30"], [1, "#C4515A"]],
            colorbar=dict(title=dict(text="Bookings", font=dict(color=CHART_MUTED_COLOR)), tickfont=dict(color=CHART_MUTED_COLOR))
        ))
        fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title="Hour", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- TAB 4: Maintenance Forecast ----
    with tab4:
        section_header("Maintenance Request Volume Trend", "🔧")
        
        complaints["month"] = complaints["created_date"].dt.to_period("M").astype(str)
        maint_categories = ["Plumbing/Water Leakage", "Electrical Issues", "HVAC/Air Conditioning",
                           "Appliance Repair", "Paint/Finishing", "Door/Window Issues"]
        maint = complaints[complaints["category"].isin(maint_categories)]
        maint_monthly = maint.groupby(["month", "category"]).size().reset_index(name="count")
        maint_monthly = maint_monthly[maint_monthly["month"] >= "2024-01"]
        
        fig = px.line(maint_monthly, x="month", y="count", color="category", markers=True,
                     color_discrete_sequence=[COLORS["teal"], COLORS["coral"], COLORS["yellow"],
                                            COLORS["blue"], COLORS["purple"], COLORS["green"]])
        fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="", yaxis_title="Requests",
                         legend_title_text="Category")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance by community
        section_header("Maintenance Demand by Community", "🏘️")
        maint_comm = maint.groupby("community").size().reset_index(name="count").sort_values("count", ascending=True)
        fig = go.Figure(data=[go.Bar(
            y=maint_comm["community"], x=maint_comm["count"], orientation="h",
            marker_color=COLORS["coral"], text=maint_comm["count"],
            textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
        )])
        fig.update_layout(**PLOT_LAYOUT, height=400, xaxis_title="Maintenance Requests", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_demand(bookings, complaints))


# ============================================================================
# PAGE: AI PERFORMANCE — UNIFIED 3-AGENT COMMAND CENTER
# ============================================================================
def page_ai_performance(interactions, residents, leads):
    """Unified AI Operations Command Center — All 3 AI Agents."""
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">AI Operations Command Center</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Unified AI Agent Performance</h1>
        <p style="color: var(--text-muted);">Cross-agent monitoring: Debt Collection AI · Customer Care AI · Lead Management AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Compute metrics for all 3 agents ----
    
    # AGENT 1: Debt Collection AI
    has_pred = "predicted_default_prob" in residents.columns
    if has_pred:
        high_risk_count = len(residents[residents["predicted_risk_grade"].isin(["D - High", "E - Critical"])])
        avg_risk_accuracy = 0.997  # From training ROC-AUC
    else:
        high_risk_count = len(residents[residents["risk_category"].isin(["High", "Critical"])])
        avg_risk_accuracy = 0
    total_residents = len(residents)
    default_rate = residents["default_flag"].mean() * 100
    avg_dpd = residents["current_dpd"].mean()
    
    # AGENT 2: Customer Care AI
    total_interactions = len(interactions)
    ai_resolution = interactions["resolved_by_ai"].mean() * 100
    escalation_rate = interactions["escalated_to_human"].mean() * 100
    avg_csat = interactions["csat_score"].mean()
    avg_sentiment = interactions["sentiment_score"].mean()
    avg_duration = interactions["duration_seconds"].mean()
    
    # AGENT 3: Lead Management AI
    if not leads.empty:
        total_leads = len(leads)
        won_leads = len(leads[leads["stage"] == "Won"])
        active_leads = len(leads[~leads["stage"].isin(["Won", "Lost"])])
        lead_conv_rate = won_leads / total_leads * 100
        ai_assisted_pct = leads["ai_assisted"].mean() * 100
        avg_response_hrs = leads["response_time_hours"].mean()
        has_lead_ml = "conversion_probability" in leads.columns
        if has_lead_ml:
            hot_leads = len(leads[(~leads["stage"].isin(["Won", "Lost"])) & (leads.get("lead_priority", "") == "Hot")])
        else:
            hot_leads = 0
    else:
        total_leads = won_leads = active_leads = hot_leads = 0
        lead_conv_rate = ai_assisted_pct = avg_response_hrs = 0
        has_lead_ml = False
    
    # ---- Top-Level KPIs: One per Agent ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {COLORS['coral']};">
            <div class="kpi-label">🛡️ DEBT COLLECTION AI</div>
            <div class="kpi-value" style="color: {COLORS['coral']};">{high_risk_count:,}</div>
            <div class="kpi-delta delta-up">Flagged high-risk · {default_rate:.1f}% default rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {COLORS['teal']};">
            <div class="kpi-label">💬 CUSTOMER CARE AI</div>
            <div class="kpi-value" style="color: {COLORS['teal']};">{ai_resolution:.1f}%</div>
            <div class="kpi-delta delta-up">AI resolution · {total_interactions:,} interactions</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {COLORS['gold']};">
            <div class="kpi-label">🎯 LEAD MANAGEMENT AI</div>
            <div class="kpi-value" style="color: {COLORS['gold']};">{lead_conv_rate:.1f}%</div>
            <div class="kpi-delta delta-up">Conversion rate · {active_leads:,} active leads</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Tabs per Agent ----
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cross-Agent Summary", "🛡️ Debt Collection AI", "💬 Customer Care AI", "🎯 Lead Management AI"])
    
    # ---- TAB 1: Cross-Agent Summary ----
    with tab1:
        section_header("AI Agent Scorecard", "📊")
        
        scorecard_data = {
            "AI Agent": ["🛡️ Debt Collection", "💬 Customer Care", "🎯 Lead Management"],
            "Status": ["✅ Active", "✅ Active", "✅ Active"],
            "Total Processed": [f"{total_residents:,}", f"{total_interactions:,}", f"{total_leads:,}"],
            "Primary KPI": [
                f"ROC-AUC: {avg_risk_accuracy:.3f}" if has_pred else "Rule-based scoring",
                f"AI Resolution: {ai_resolution:.1f}%",
                f"Conv Rate: {lead_conv_rate:.1f}%"
            ],
            "AI Coverage": [
                f"{high_risk_count/max(total_residents,1)*100:.1f}% flagged",
                f"{ai_resolution:.1f}% auto-resolved",
                f"{ai_assisted_pct:.0f}% AI-assisted"
            ],
            "Satisfaction": [
                f"Avg DPD: {avg_dpd:.0f} days",
                f"CSAT: {avg_csat:.1f}/5",
                f"Avg Response: {avg_response_hrs:.1f} hrs"
            ],
        }
        st.dataframe(pd.DataFrame(scorecard_data), use_container_width=True, hide_index=True)
        
        st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
        
        # Combined impact metrics
        section_header("Combined AI Impact", "⚡")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            total_processed = total_residents + total_interactions + total_leads
            st.markdown(kpi_card("Total Records Processed", f"{total_processed:,}", "Across all agents", "up", COLORS["teal"]), unsafe_allow_html=True)
        with c2:
            at_risk_val = residents[residents["risk_category"].isin(["High", "Critical"])]["property_value_aed"].sum()
            at_risk_str = f"{at_risk_val/1e9:.1f}B" if at_risk_val >= 1e9 else f"{at_risk_val/1e6:.0f}M"
            st.markdown(kpi_card("Portfolio Protected", f"{at_risk_str} SAR", "Early risk detection", "up", COLORS["coral"]), unsafe_allow_html=True)
        with c3:
            cost_saved = int(total_interactions * (ai_resolution/100) * 15)  # ~15 SAR per auto-resolved interaction
            st.markdown(kpi_card("Est. Cost Savings", f"{cost_saved/1e6:.1f}M SAR", "Auto-resolved interactions", "up", COLORS["green"]), unsafe_allow_html=True)
        with c4:
            if not leads.empty:
                closed_val = leads[leads["stage"]=="Won"]["conversion_value_sar"].sum()
                closed_str = f"{closed_val/1e9:.1f}B" if closed_val >= 1e9 else f"{closed_val/1e6:.0f}M"
            else:
                closed_str = "N/A"
            st.markdown(kpi_card("Revenue Closed", f"{closed_str} SAR", f"{won_leads:,} deals", "up", COLORS["gold"]), unsafe_allow_html=True)
    
    # ---- TAB 2: Debt Collection AI ----
    with tab2:
        section_header("Debt Collection AI — Risk Prediction Performance", "🛡️")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Model Accuracy", f"{avg_risk_accuracy:.1%}" if has_pred else "N/A", "ROC-AUC score", "up", COLORS["green"]), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("High Risk Flagged", f"{high_risk_count:,}", f"{high_risk_count/max(total_residents,1)*100:.1f}% of portfolio", "up", COLORS["coral"]), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("Default Rate", f"{default_rate:.1f}%", f"{int(residents['default_flag'].sum()):,} defaulted", "down", COLORS["red"]), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Avg Days Past Due", f"{avg_dpd:.0f}", "days", "down", COLORS["yellow"]), unsafe_allow_html=True)
        
        st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            section_header("Risk Grade Distribution", "📊")
            if has_pred:
                risk_dist = residents["predicted_risk_grade"].value_counts().sort_index()
            else:
                risk_dist = residents["risk_category"].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Bar(
                x=risk_dist.index, y=risk_dist.values,
                marker_color=[RISK_COLORS.get(k, COLORS["teal"]) for k in risk_dist.index],
                text=risk_dist.values, textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Risk Grade", yaxis_title="Residents")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("DPD Distribution", "📈")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residents["current_dpd"], nbinsx=40,
                marker_color=COLORS["coral"], opacity=0.8
            ))
            fig.add_vline(x=30, line_dash="dash", line_color=COLORS["red"],
                         annotation_text="30-day threshold", annotation_font_color=CHART_TEXT_COLOR)
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Days Past Due", yaxis_title="Residents")
            st.plotly_chart(fig, use_container_width=True)
        
        # Intervention effectiveness
        section_header("Collection Efficiency by Zone", "🗺️")
        zone_risk = residents.groupby("zone").agg(
            total=("resident_id", "count"),
            high_risk=("risk_category", lambda x: x.isin(["High", "Critical"]).sum()),
            avg_dpd=("current_dpd", "mean"),
            avg_balance=("outstanding_balance_aed", "mean"),
            default_rate=("default_flag", "mean"),
        ).reset_index()
        zone_risk["high_risk_pct"] = (zone_risk["high_risk"] / zone_risk["total"] * 100).round(1)
        zone_risk["default_rate"] = (zone_risk["default_rate"] * 100).round(1)
        zone_risk["avg_dpd"] = zone_risk["avg_dpd"].round(0).astype(int)
        zone_risk["avg_balance"] = (zone_risk["avg_balance"] / 1e6).round(2)
        zone_risk.columns = ["Zone", "Residents", "High Risk", "Avg DPD", "Avg Balance (M SAR)", "Default Rate %", "High Risk %"]
        st.dataframe(zone_risk, use_container_width=True, hide_index=True)
    
    # ---- TAB 3: Customer Care AI ----
    with tab3:
        section_header("Customer Care AI — Interaction Analytics", "💬")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("AI Resolution", f"{ai_resolution:.1f}%", f"{total_interactions:,} total", "up", COLORS["green"]), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Escalation Rate", f"{escalation_rate:.1f}%", "", "down", COLORS["coral"]), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("Avg CSAT", f"{avg_csat:.1f}/5", "", "up", COLORS["blue"]), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Avg Duration", f"{avg_duration:.0f}s", "", "up", COLORS["yellow"]), unsafe_allow_html=True)
        
        st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            section_header("Resolution by Channel", "📡")
            ch_perf = interactions.groupby("channel").agg(
                ai_rate=("resolved_by_ai", "mean"),
                count=("interaction_id", "count"),
            ).reset_index()
            ch_perf["ai_rate"] *= 100
            
            fig = go.Figure(data=[go.Bar(
                x=ch_perf["channel"], y=ch_perf["ai_rate"],
                marker_color=COLORS["teal"],
                text=[f"{v:.1f}%" for v in ch_perf["ai_rate"]],
                textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="", yaxis_title="AI Resolution %", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("CSAT Distribution", "⭐")
            csat_dist = interactions["csat_score"].value_counts().sort_index()
            fig = go.Figure(data=[go.Bar(
                x=csat_dist.index, y=csat_dist.values,
                marker_color=[COLORS["red"], COLORS["coral"], COLORS["yellow"], COLORS["teal"], COLORS["green"]],
                text=csat_dist.values, textposition="outside",
                textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**{**PLOT_LAYOUT, "xaxis": dict(dtick=1, **{k: v for k, v in PLOT_LAYOUT.get("xaxis", {}).items() if k in ["gridcolor", "zerolinecolor", "linecolor"]})}, height=380, xaxis_title="CSAT Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trend
        section_header("Monthly AI Performance Trend", "📈")
        interactions["month"] = interactions["timestamp"].dt.to_period("M").astype(str)
        monthly_perf = interactions.groupby("month").agg(
            ai_resolution=("resolved_by_ai", "mean"),
            escalation=("escalated_to_human", "mean"),
            volume=("interaction_id", "count")
        ).reset_index().tail(18)
        monthly_perf["ai_resolution"] *= 100
        monthly_perf["escalation"] *= 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=monthly_perf["month"], y=monthly_perf["ai_resolution"],
            mode="lines+markers", name="AI Resolution %",
            line=dict(color=COLORS["green"], width=3), marker=dict(size=8)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=monthly_perf["month"], y=monthly_perf["escalation"],
            mode="lines+markers", name="Escalation %",
            line=dict(color=COLORS["coral"], width=3), marker=dict(size=8)
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=monthly_perf["month"], y=monthly_perf["volume"],
            name="Volume", marker_color=COLORS["navy"], opacity=0.3
        ), secondary_y=True)
        fig.update_layout(**{**PLOT_LAYOUT, "legend": dict(x=0, y=1.12, orientation="h", bgcolor="rgba(0,0,0,0)", font=dict(color=CHART_MUTED_COLOR))}, height=400)
        fig.update_yaxes(title_text="Percentage", secondary_y=False, gridcolor=PLOT_LAYOUT["yaxis"]["gridcolor"] if IS_DARK else "#F0EBE3")
        fig.update_yaxes(title_text="Volume", secondary_y=True, gridcolor=PLOT_LAYOUT["yaxis"]["gridcolor"] if IS_DARK else "#F0EBE3")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Purpose table
        section_header("Performance by Purpose", "🎯")
        purpose_perf = interactions.groupby("purpose").agg(
            count=("interaction_id", "count"),
            ai_rate=("resolved_by_ai", "mean"),
            avg_csat=("csat_score", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
        ).reset_index()
        purpose_perf["ai_rate"] = (purpose_perf["ai_rate"] * 100).round(1)
        purpose_perf["avg_csat"] = purpose_perf["avg_csat"].round(2)
        purpose_perf["avg_sentiment"] = purpose_perf["avg_sentiment"].round(3)
        purpose_perf = purpose_perf.sort_values("count", ascending=False)
        purpose_perf.columns = ["Purpose", "Volume", "AI Resolution %", "Avg CSAT", "Avg Sentiment"]
        st.dataframe(purpose_perf, use_container_width=True, hide_index=True)
    
    # ---- TAB 4: Lead Management AI ----
    with tab4:
        if leads.empty:
            st.warning("Lead data not available.")
            return
        
        section_header("Lead Management AI — Conversion Intelligence", "🎯")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card("Conversion Rate", f"{lead_conv_rate:.1f}%", f"{won_leads:,} won", "up", COLORS["green"]), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("AI-Assisted", f"{ai_assisted_pct:.0f}%", f"{total_leads:,} leads", "up", COLORS["blue"]), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("Avg Response Time", f"{avg_response_hrs:.1f} hrs", "", "up", COLORS["yellow"]), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Hot Leads (ML)", f"{hot_leads:,}", ">75% conv prob" if has_lead_ml else "No ML model", "up", COLORS["coral"]), unsafe_allow_html=True)
        
        st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            section_header("AI vs Manual: Conversion Rate", "🤖")
            ai_leads = leads[leads["ai_assisted"] == True]
            manual_leads = leads[leads["ai_assisted"] == False]
            ai_conv = (ai_leads["stage"] == "Won").mean() * 100
            man_conv = (manual_leads["stage"] == "Won").mean() * 100
            
            fig = go.Figure(data=[go.Bar(
                x=["AI-Assisted", "Manual"],
                y=[ai_conv, man_conv],
                marker_color=[COLORS["teal"], COLORS["taupe"]],
                text=[f"{ai_conv:.1f}%", f"{man_conv:.1f}%"],
                textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="", yaxis_title="Conversion Rate %")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_header("AI vs Manual: Response Time", "⏱️")
            ai_resp = ai_leads["response_time_hours"].mean()
            man_resp = manual_leads["response_time_hours"].mean()
            
            fig = go.Figure(data=[go.Bar(
                x=["AI-Assisted", "Manual"],
                y=[ai_resp, man_resp],
                marker_color=[COLORS["teal"], COLORS["taupe"]],
                text=[f"{ai_resp:.1f} hrs", f"{man_resp:.1f} hrs"],
                textposition="outside", textfont=dict(color=CHART_TEXT_COLOR)
            )])
            fig.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="", yaxis_title="Avg Response (hours)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Conversion by source
        section_header("Conversion by Lead Source", "📡")
        source_perf = leads.groupby("source").agg(
            total=("lead_id", "count"),
            won=("stage", lambda x: (x == "Won").sum()),
            ai_pct=("ai_assisted", "mean"),
            avg_score=("lead_score", "mean"),
        ).reset_index()
        source_perf["conv_rate"] = (source_perf["won"] / source_perf["total"] * 100).round(1)
        source_perf["ai_pct"] = (source_perf["ai_pct"] * 100).round(0)
        source_perf["avg_score"] = source_perf["avg_score"].round(0).astype(int)
        source_perf = source_perf.sort_values("conv_rate", ascending=False)
        source_perf.columns = ["Source", "Total Leads", "Won", "AI Assisted %", "Avg Score", "Conv Rate %"]
        st.dataframe(source_perf, use_container_width=True, hide_index=True)
    
    # ---- AI INSIGHTS ----
    render_ai_insights(insights_ai_performance(interactions, residents, leads))


# ============================================================================
# PAGE: RESIDENT DEEP DIVE
# ============================================================================
def page_resident_deep_dive(residents, payments, complaints, interactions):
    """Individual Resident Risk Profile."""
    
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="font-size: 10px; color: var(--accent-gold); letter-spacing: 3px; text-transform: uppercase; font-weight: 600;">Individual Profile</div>
        <h1 style="font-size: 28px; margin-top: 8px;">Resident Deep Dive</h1>
        <p style="color: var(--text-muted);">Individual resident risk profile, payment history, and engagement analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Search ----
    col1, col2 = st.columns([2, 3])
    
    with col1:
        search_method = st.selectbox("Search by", ["Resident ID", "Name", "Community (High Risk)"])
    
    with col2:
        if search_method == "Resident ID":
            selected_id = st.text_input("Enter Resident ID", value="RES-000001", placeholder="RES-XXXXXX")
        elif search_method == "Name":
            name_search = st.text_input("Search Name", placeholder="Enter name...")
            if name_search:
                matches = residents[
                    residents["first_name"].str.contains(name_search, case=False, na=False) |
                    residents["last_name"].str.contains(name_search, case=False, na=False)
                ]
                if len(matches) > 0:
                    options = [f"{r['resident_id']} — {r['first_name']} {r['last_name']} ({r['community']})" 
                              for _, r in matches.head(10).iterrows()]
                    selected = st.selectbox("Select Resident", options)
                    selected_id = selected.split(" — ")[0]
                else:
                    st.warning("No residents found.")
                    selected_id = None
            else:
                selected_id = None
        else:
            has_pred = "predicted_risk_grade" in residents.columns
            if has_pred:
                high_risk = residents[residents["predicted_risk_grade"].isin(["D - High", "E - Critical"])]
            else:
                high_risk = residents[residents["risk_category"].isin(["High", "Critical"])]
            
            community = st.selectbox("Select Community", sorted(residents["community"].unique()))
            comm_high = high_risk[high_risk["community"] == community].sort_values("risk_score", ascending=False)
            
            if len(comm_high) > 0:
                options = [f"{r['resident_id']} — {r['first_name']} {r['last_name']} (Risk: {r['risk_score']:.1f})" 
                          for _, r in comm_high.head(20).iterrows()]
                selected = st.selectbox(f"High Risk in {community}", options)
                selected_id = selected.split(" — ")[0]
            else:
                st.success(f"No high-risk residents in {community}")
                selected_id = None
    
    if selected_id is None:
        return
    
    # ---- Get Resident Data ----
    res = residents[residents["resident_id"] == selected_id]
    
    if len(res) == 0:
        st.error(f"Resident {selected_id} not found.")
        return
    
    res = res.iloc[0]
    
    # ---- Profile Card ----
    has_pred = "predicted_risk_grade" in residents.columns
    risk_grade = res.get("predicted_risk_grade", res.get("risk_category", "Unknown"))
    risk_prob = res.get("predicted_default_prob", res.get("risk_score", 0) / 100)
    
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border-main); border-radius: 12px; padding: 28px; margin: 16px 0; box-shadow: var(--shadow-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 26px; font-weight: 600; color: var(--text-primary); font-family: 'Playfair Display', serif;">{res['first_name']} {res['last_name']}</div>
                <div style="color: var(--text-muted); margin-top: 6px; font-size: 13px; letter-spacing: 0.5px;">{selected_id} · {res['community']} · {res['zone']} Zone · {res['property_type']}</div>
            </div>
            <div style="text-align: right;">
                {risk_badge(risk_grade) if has_pred else f'<span class="risk-badge risk-medium">{risk_grade}</span>'}
                <div style="color: var(--text-muted); font-size: 12px; margin-top: 8px;">Default Probability: <span style="color: #C4515A; font-weight: 700;">{risk_prob:.1%}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Detail Cards ----
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(kpi_card("Property Value", f"{res['property_value_aed']:,.0f} SAR", "", "up", COLORS["teal"]), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Outstanding Balance", f"{res['outstanding_balance_aed']:,.0f} SAR", "", "down", COLORS["coral"]), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Credit Score", f"{res['credit_score']}", "", "up", COLORS["blue"]), unsafe_allow_html=True)
    with c4:
        sat = res.get("satisfaction_score", 0)
        st.markdown(kpi_card("Satisfaction", f"{sat:.1f}/100", "", "up", COLORS["green"]), unsafe_allow_html=True)
    
    st.markdown('<div style="margin:8px 0"></div>', unsafe_allow_html=True)
    
    # ---- Risk Factors ----
    col1, col2 = st.columns(2)
    
    with col1:
        section_header("Risk Factor Analysis", "⚡")
        
        risk_factors = {
            "Days Past Due": {"value": res["current_dpd"], "max": 180, "threshold": 30},
            "Late Payments (12m)": {"value": res["late_payments_12m"], "max": 12, "threshold": 3},
            "Payment Consistency": {"value": 100 - res["payment_consistency_pct"], "max": 100, "threshold": 30},
            "Debt-to-Income %": {"value": res["debt_to_income_pct"], "max": 60, "threshold": 40},
            "Credit Score Gap": {"value": max(0, 750 - res["credit_score"]), "max": 450, "threshold": 100},
        }
        
        factors = list(risk_factors.keys())
        values = [min(v["value"] / v["max"] * 100, 100) for v in risk_factors.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=factors[::-1], x=values[::-1],
            orientation="h",
            marker_color=[COLORS["red"] if v > 60 else COLORS["yellow"] if v > 30 else COLORS["green"] for v in values[::-1]],
            text=[f"{v:.0f}%" for v in values[::-1]],
            textposition="outside",
            textfont=dict(color=CHART_TEXT_COLOR)
        ))
        fig.update_layout(**PLOT_LAYOUT, height=300, xaxis_title="Risk Contribution %", yaxis_title="",
                         xaxis_range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        section_header("Resident Details", "👤")
        
        details = {
            "Nationality": res["nationality"],
            "Age": res["age"],
            "Family Size": res["family_size"],
            "Occupation": res["occupation_category"],
            "Tenure": f"{res['tenure_months']} months",
            "Monthly Income": f"{res['monthly_income_aed']:,.0f} SAR",
            "Monthly Installment": f"{res['monthly_installment_aed']:,.0f} SAR",
            "Service Charge": f"{res['service_charge_annual_aed']:,.0f} SAR/yr",
            "Occupancy Type": res["occupancy_type"],
            "Preferred Language": res["preferred_language"],
        }
        
        for k, v in details.items():
            st.markdown(f"""
            <div class="detail-row">
                <span class="label">{k}</span>
                <span class="value">{v}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # ---- Payment History ----
    section_header("Payment History", "💳")
    
    res_payments = payments[payments["resident_id"] == selected_id].sort_values("due_date", ascending=False)
    
    if len(res_payments) > 0:
        display_pay = res_payments[["due_date", "payment_type", "amount_due_aed", "amount_paid_aed", 
                                   "payment_status", "delay_days", "payment_method"]].head(15)
        st.dataframe(display_pay, use_container_width=True, hide_index=True)
    else:
        st.info("No payment records found.")
    
    # ---- Complaint History ----
    section_header("Complaint History", "📋")
    
    res_complaints = complaints[complaints["resident_id"] == selected_id].sort_values("created_date", ascending=False)
    
    if len(res_complaints) > 0:
        display_comp = res_complaints[["created_date", "category", "severity", "status", 
                                      "resolution_hours", "satisfaction_rating"]].head(10)
        st.dataframe(display_comp, use_container_width=True, hide_index=True)
    else:
        st.success("No complaints filed.")


# ============================================================================
# MAIN APP
# ============================================================================
# Load Data
residents, payments, complaints, bookings, interactions, leads = load_all_data()
metadata = load_model_metadata()

# Top Navigation + Filters
page, filter_communities, filter_zones = render_top_nav()

# Apply filters
residents_f = apply_filters(residents, filter_communities, filter_zones)
payments_f = apply_filters(payments, filter_communities, filter_zones)
complaints_f = apply_filters(complaints, filter_communities, filter_zones)
bookings_f = apply_filters(bookings, filter_communities, filter_zones)
interactions_f = apply_filters(interactions, filter_communities, filter_zones)
leads_f = apply_filters(leads, filter_communities, filter_zones, comm_col="community_interest") if not leads.empty else leads

# Route to page
if page == "Overview":
    page_executive_summary(residents_f, payments_f, complaints_f, bookings_f, interactions_f)
elif page == "Payment Risk":
    page_payment_risk(residents_f, payments_f)
elif page == "Complaints":
    page_complaint_intelligence(residents_f, complaints_f)
elif page == "Sentiment":
    page_sentiment_satisfaction(residents_f, interactions_f, complaints_f)
elif page == "Leads":
    page_lead_intelligence(leads_f)
elif page == "Demand":
    page_demand_forecasting(bookings_f, complaints_f, interactions_f)
elif page == "AI Performance":
    page_ai_performance(interactions_f, residents_f, leads_f)
elif page == "Resident Dive":
    page_resident_deep_dive(residents_f, payments_f, complaints_f, interactions_f)
