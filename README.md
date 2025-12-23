NYC Taxi Trip Duration Prediction 🚕🕒

This project predicts the duration of NYC taxi rides using three models:
1. Random Forest (Scikit-learn)
2. XGBoost
3. LSTM (PyTorch)

The dataset is from Kaggle: "NYC Taxi Trip Duration".

The Colab notebook includes:
- Full EDA
- Feature engineering
- Model training and evaluation
- Log-transforming trip duration for better results

Deployment:
- Backend: FastAPI (`app/main.py`)
- Frontend: Streamlit (`ui/streamlit_app.py`)

How to Run:

1. Unzip the folder and open a terminal.
2. (Optional) Create a virtual environment:
   - `python -m venv venv`
   - `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install dependencies:
   - `pip install -r app/requirements.txt`
4. Run backend API:
   - `cd app`
   - `uvicorn main:app --reload`
5. Run Streamlit UI (in a new terminal):
   - `cd ui`
   - `streamlit run streamlit_app.py`

The app allows users to input trip details and get predicted duration in seconds.

Author: Assad Bin Tanveer
