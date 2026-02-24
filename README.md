# Tourism_Experience_Analytics

An end-to-end Machine Learning application that analyzes tourist behavior, predicts user patterns, and recommends attractions through an interactive Streamlit web app.
The project integrates regression, classification, and recommendation systems into a single deployment-ready solution.


Project Overview

This system performs:
  - Rating Prediction (Regression)
    Predicts attraction ratings using contextual and engineered aggregate features.
    Model used: CatBoost Regressor.
  - Visit Mode Prediction (Classification)
    Predicts likely visit mode (Family, Friends, Solo, etc.) based on location, attraction, and rating features.
    Model used: CatBoost Classifier.
  - Attraction Recommendation
    Item-based collaborative filtering recommending attractions using similarity between user–attraction interactions.

The repository includes trained models and saved artifacts, allowing the application to run without retraining.


Tech Stack

Python
Pandas, NumPy
Matplotlib, Seaborn
CatBoost
Joblib
Streamlit
MLFlow

Project Structure

Tourism_Experience_Analytics/
│
├── code_files/app.py
├── code_files/utils.py
├── models/        # Trained .cbm files
├── artifacts/     # Saved feature lists & aggregates
├── clean_data/    # df_clean.csv
└── README.md


How to Run the Streamlit app

pip install -r requirements.txt
streamlit run app.py
