# NASA Turbofan Engine Remaining Useful Life (RUL) Prediction

## Overview
This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using the **NASA C-MAPSS dataset**.  
I built machine learning models to estimate how many cycles an engine has left before failure, and deploy an **interactive dashboard** for visual exploration.

---

## Dataset
- **Source**: [NASA C-MAPSS Turbofan Degradation Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)  
- **Files used**: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`  
- **Features**:
  - Operational settings (e.g., altitude, Mach number, throttle resolver angle)
  - 21 sensor measurements (temperatures, pressures, vibration signals)
- **Labels**:
  - RUL (Remaining Useful Life) in cycles

---
I approached the project in multiple **phases**:

### Data Preparation
- Parsed the raw `.txt` files into Pandas DataFrames.  
- Created labels for RUL (`train` by counting down from last cycle, `test` from provided RUL file).  
- Checked sanity: no negative RULs, consistent cycles per unit.

### Feature Engineering
- Rolling-window statistics for each sensor:
  - Mean & Standard Deviation over windows (5, 15, 30 cycles).  
- Removed low-variance or constant sensors.  
- Final feature set balanced between **predictive power** and **dimensionality control**.

### Baseline Models
I trained and evaluated:
- **Random Forest Regressor**
- **Gradient Boosting Regressor (GBR)**

Metrics used:
- **RMSE** (Root Mean Square Error)  
- **MAE** (Mean Absolute Error)  
- **NASA Scoring Function** (penalizes late vs. early predictions asymmetrically).  

Baseline Results (FD001 dataset):
| Model              | RMSE   | MAE   | NASA Score |
|--------------------|--------|-------|------------|
| Random Forest      | ~86.2  | ~75.5 | 559k       |
| Gradient Boosting  | ~86.2  | ~75.5 | 559k       |

These were poor → model underfit.

### Improvements
- **Capped very large RUL values** (engines far from failure aren’t as critical).  
- **Added rolling windows (5, 15, 30)** for smoother degradation signals.  

Improved Results:
| Model   | RMSE   | MAE   | NASA Score |
|---------|--------|-------|------------|
| RF (capped)  | ~18.3 | ~13.5 | 806      |
| GBR (capped) | ~16.9 | ~12.9 | 533      |

Gradient Boosting with capped labels performed best.  

### Deployment
- Exported the tuned GBR model as a `.skops` file for portability.  
- Hosted model on **Hugging Face Hub**.  
- Built a **Streamlit Dashboard** that:
  - Loads the model directly from Hugging Face.  
  - Allows interactive predictions.  
  - Visualizes predicted vs. true RUL (scatter plots, degradation curves).  

---

## Outputs
- **Train RUL Distribution**: Smooth degradation curves across engines.  
- **Test RUL Distribution**: Realistic spread across engine units.  
- **Predicted vs True RUL (GBR)**: Predictions closely follow the diagonal (ideal).  

---

## Dashboard
The final dashboard is deployed with [**Streamlit**](https://nasa-turbofan-rul-fd001-27tu77kqv9b3hwm7zxbqew.streamlit.app/)
Features:
- Displays engine health trajectories.  
- Predicts RUL for unseen test units.  
- Interactive plots for exploration.  

---

## Usage
### Local Setup
```bash
# Clone repo
git clone https://github.com/<your-username>/nasa-turbofan-rul.git
cd nasa-turbofan-rul

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
