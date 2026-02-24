# Importing the necessary libraries
import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor, CatBoostClassifier

# Importing the functions from utils.py
from utils import (plot_tourism_trends_year,
                   plot_tourism_trends_month,
                   plot_rating_distribution,
                   plot_avg_rating_by_attr_type,
                   plot_count_visitmode,
                   plot_avg_type_tourist_rating,
                   plot_prfrd_attr_by_tourist_type,
                   plot_mode_by_month,
                   plot_top_ten_attr,
                   plot_user_interactions,
                   plot_attr_interactions,
                   recommend_similar_attractions)

# App configuration
st.set_page_config(
    page_title="Tourism Experience Analytics",
    layout="wide")

# Loading the models and artifacts
@st.cache_resource
def load_models_artifacts():
    # Loading the cleaned dataframe
    df_clean = pd.read_csv('../clean_data/df_clean.csv')

    # Loading the trained models
    reg_m = CatBoostRegressor().load_model('../models/cat_reg_final.cbm')
    clf_m = CatBoostClassifier().load_model('../models/cat_clf_final.cbm')

    # Loading Recommendation Artifacts
    attr_list = joblib.load('../artifacts/attr_list.pkl')
    attr_similarity_df = joblib.load('../artifacts/attr_similarity_df.pkl')

    # Loading Regression Artifacts
    attr_stats_reg = joblib.load('../artifacts/attr_stats_reg.pkl')
    season_type_stats_reg = joblib.load('../artifacts/season_type_stats_reg.pkl')
    type_stats_reg = joblib.load('../artifacts/type_stats_reg.pkl')
    reg_cat_cols = joblib.load('../artifacts/reg_cat_cols.pkl')
    reg_feature_cols = joblib.load('../artifacts/reg_feature_cols.pkl')

    # Loading Classification Artifacts
    clf_cat_cols = joblib.load('../artifacts/clf_cat_cols.pkl')
    clf_feature_cols = joblib.load('../artifacts/clf_feature_cols.pkl')

    return (df_clean,
            clf_m,
            reg_m,
            attr_list,
            attr_similarity_df,
            attr_stats_reg,
            season_type_stats_reg,
            type_stats_reg,
            reg_cat_cols,
            reg_feature_cols,
            clf_cat_cols,
            clf_feature_cols)


(df_clean,
 clf_m,
 reg_m,
 attr_list,
 attr_similarity_df,
 attr_stats_reg,
 season_type_stats_reg,
 type_stats_reg,
 reg_cat_cols,
 reg_feature_cols,
 clf_cat_cols,
 clf_feature_cols) = load_models_artifacts()

# Configuring the sidebar
st.sidebar.title("🗽Tourism Experience Analytics: Regression, Classification & Recommendations🎢")
st.sidebar.header("🔧 Controls")
section = st.sidebar.radio('Choose Section',
                           ['Analytics', 'Rating Prediction', 'Visit Mode Prediction', 'Attraction Recommendation'])

# Analytics Section
if section == 'Analytics':
    st.subheader("📊 Data Visualizations")

    analytics_option = st.selectbox(
        "Select Analysis",
            ["Tourism Trends by Year",
            "Tourism Trends by Month",
            "Rating Distribution",
            "Average Rating by Attraction Type",
            "Count of Each Visit Mode",
            "Average Rating by Tourist Type",
            "Preferred Attraction Type by Tourist Type",
            "Visit Mode Trends by Month",
            "Top Ten Attractions",
            "User Interactions with Attractions",
            "Attraction Interactions"]
    )

    if analytics_option == "Tourism Trends by Year":
        st.pyplot(plot_tourism_trends_year(df_clean))

    elif analytics_option == "Tourism Trends by Month":
        st.pyplot(plot_tourism_trends_month(df_clean))

    elif analytics_option == "Rating Distribution":
        st.pyplot(plot_rating_distribution(df_clean))

    elif analytics_option == "Average Rating by Attraction Type":
        st.pyplot(plot_avg_rating_by_attr_type(df_clean))

    elif analytics_option == "Count of Each Visit Mode":
        st.pyplot(plot_count_visitmode(df_clean))

    elif analytics_option == "Average Rating by Tourist Type":
        st.pyplot(plot_avg_type_tourist_rating(df_clean))

    elif analytics_option == "Preferred Attraction Type by Tourist Type":
        st.pyplot(plot_prfrd_attr_by_tourist_type(df_clean))

    elif analytics_option == "Visit Mode Trends by Month":
        st.pyplot(plot_mode_by_month(df_clean))

    elif analytics_option == "Top Ten Attractions":
        st.pyplot(plot_top_ten_attr(df_clean))

    elif analytics_option == "User Interactions with Attractions":
        st.pyplot(plot_user_interactions(df_clean))

    elif analytics_option == "Attraction Interactions":
        st.pyplot(plot_attr_interactions(df_clean))

# Rating Prediction Section
elif section == 'Rating Prediction':
    st.subheader("⭐ Rating Prediction")

    # User Inputs
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month = st.selectbox('Select Month', month_order)
    attraction = st.selectbox('Select Attraction', sorted(df_clean['Attraction'].unique()))

    # Automatically derive AttractionType from the selected Attraction
    attr_type = df_clean[df_clean['Attraction'] == attraction]['AttractionType'].iloc[0]

    visit_mode = st.selectbox('Select Visit Mode', sorted(df_clean['VisitMode'].unique()))

    if st.button("Predict Rating"):

        # Base input dataframe
        input_df = pd.DataFrame({
            'Month': [month],
            'Attraction': [attraction],
            'AttractionType': [attr_type],
            'VisitMode': [visit_mode]
        })

        # Merging with the global aggregates
        input_df = input_df.merge(attr_stats_reg, on='Attraction', how='left')
        input_df = input_df.merge(type_stats_reg, on='AttractionType', how='left')
        input_df = input_df.merge(season_type_stats_reg, on=['Month', 'AttractionType'], how='left')

        # Ensuring correct column order
        x_reg_input = input_df[reg_feature_cols].copy()

        # Predicting the rating
        pred_rating = reg_m.predict(x_reg_input)[0]

        # Displaying Result
        st.success(f"🎯 Predicted Rating: {round(pred_rating, 2)} / 5")

# Visit Mode Prediction Section
elif section == 'Visit Mode Prediction':
    st.subheader('🚗 Visit Mode Prediction')
    # User Inputs
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month = st.selectbox('Select Month', month_order)
    city = st.selectbox('Select your City', sorted(df_clean['CityName'].unique()))    
    country = df_clean[df_clean['CityName'] == city]['Country'].iloc[0]
    region = df_clean[df_clean['CityName'] == city]['Region'].iloc[0]
    continent = df_clean[df_clean['CityName'] == city]['Continent'].iloc[0]
    attraction = st.selectbox('Select Attraction', sorted(df_clean['Attraction'].unique()))
    attr_type = df_clean[df_clean['Attraction'] == attraction]['AttractionType'].iloc[0]
    rating = st.slider('Select your expected rating for the attraction', 1, 5, 4)

    if st.button("Predict Visit Mode"):

        # Creating input row
        input_df = pd.DataFrame({
            "Month": [month],
            "CityName": [city],
            "Country": [country],
            "Region": [region],
            "Continent": [continent],
            "Attraction": [attraction],
            "AttractionType": [attr_type],
            "Rating": [rating]
        })

        # Ensuring correct column order
        x_clf_input = input_df[clf_feature_cols].copy()

        # Predicting the visit mode
        pred_mode = clf_m.predict(x_clf_input)[0][0]

        # Displaying Result
        st.success(f"🎯 Predicted Visit Mode: {pred_mode}")

# Attraction Recommendation Section
elif section == 'Attraction Recommendation':
    st.subheader('🎡 Attraction Recommendation')
    attr = st.selectbox('Select an Attraction:', attr_list)
    if st.button('Get Recommendations'):
        
        # Getting the recommendations and displaying them
        recommendations = recommend_similar_attractions(attr, attr_similarity_df)
        st.success("Top 5 Similar Attractions:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")