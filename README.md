# NASA Turbofan RUL (FD001) — Streamlit Dashboard

Predict Remaining Useful Life (RUL) for turbofan engines (CMAPSS FD001).  
Includes:
- EDA + label construction
- Baseline ML models (RF/GBR) with rolling-window features and capping
- Streamlit dashboard: per-engine “RUL over time” + last-cycle metrics

## Data
CMAPSS Turbofan Engine Degradation dataset (FD001 subset).  
**Source**: NASA Prognostics Center of Excellence (PCoE).  
Use for research/educational purposes only.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
