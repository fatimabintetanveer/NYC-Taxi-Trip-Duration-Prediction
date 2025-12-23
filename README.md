# 🚕 NYC Taxi Trip Duration Prediction

This project predicts the duration of New York City taxi rides using machine learning and deep learning models.  
It includes a complete data science pipeline (EDA, feature engineering, modeling) and a **user-friendly UI** for predictions.

---

## 📌 Project Overview

The goal is to estimate taxi trip durations accurately based on trip and location features.  
This can help optimize ride planning, improve scheduling, and support real-time trip management.

---

## 🧠 Models Used

- **Random Forest** (Scikit-learn)  
- **XGBoost** (Gradient Boosting)  
- **LSTM** (PyTorch) for sequence modeling of trips  

### Key Techniques:
- Log-transforming trip duration for better predictions  
- Feature engineering: distance, pickup/dropoff times, day-of-week, and more  
- Evaluation: RMSE and MAE metrics on validation set  

---

## 📊 Dataset

- **Source:** [Kaggle – NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)  
- **Note:** Full dataset is not included due to size; only a **sample CSV** is provided for testing.  

---

## 🛠 Tech Stack

| Layer        | Technology                   |
|--------------|-----------------------------|
| Backend      | FastAPI                     |
| Frontend     | Streamlit                   |
| Data Science | Python, Pandas, NumPy       |
| ML Models    | Scikit-learn, XGBoost, PyTorch |
| Visualization| Matplotlib, Seaborn         |

---

## 🖥️ Deployment

### Backend API
- File: `app/main.py`  
- Built with **FastAPI**  

### Frontend UI
- File: `ui/streamlit_app.py`  
- Built with **Streamlit**  

The app allows users to **input trip details** and receive predicted trip durations in seconds.

---

## 📸 Screenshots

![Streamlit UI](assets/ui_screenshot.png)  
*Example of the user interface to input trip details and get predictions.*

---

## 🚀 How to Run

### 1️⃣ Optional: Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
