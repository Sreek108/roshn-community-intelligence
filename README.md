# ROSHN Community Intelligence — ML-Powered Dashboard

Enterprise AI dashboard for real estate operations featuring three AI agents:

- **🛡️ Debt Collection AI** — ML-based default prediction (ROC-AUC: 0.997)
- **💬 Customer Care AI** — Sentiment analysis, AI resolution tracking
- **🎯 Lead Management AI** — Conversion prediction (ROC-AUC: 0.841)

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/02_dashboard.py
```

## Project Structure

```
├── data/                    # Synthetic datasets (30K residents, 683K records)
├── models/                  # Pre-trained ML models
├── outputs/                 # Scored datasets & evaluation charts
├── src/
│   ├── 01_train_default_model.py   # Default prediction training
│   ├── 02_dashboard.py             # Main dashboard (Streamlit)
│   └── 03_train_lead_model.py      # Lead conversion training
├── requirements.txt
└── .streamlit/config.toml
```

## Tech Stack

Python · Streamlit · Plotly · Scikit-learn · Pandas
