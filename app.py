import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 8px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #2a5298;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #2a5298;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1e3c72;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Student Academic Performance Predictor")
st.markdown("""
Welcome! This app predicts student exam scores based on their habits and background. Upload your data or enter details below to get started!
""")

# Sidebar with fun facts and engagement
st.sidebar.header("Did You Know?")
st.sidebar.info("Students who sleep 7-8 hours tend to perform better academically!")
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.write("This app uses machine learning models trained on real student data.")
st.sidebar.write("Made with ❤️ using Streamlit.")
st.sidebar.write("This application used to predict student performance")

# Load models
def load_model(model_name):
    with open(model_name, "rb") as f:
        return pickle.load(f)

model_options = {
    "Linear Regression": "linear_model.pkl",
    "Ridge Regression": "ridge_model.pkl",
    "Lasso Regression": "lasso_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

model_choice = st.selectbox("Choose a model for prediction:", list(model_options.keys()))
model = load_model(model_options[model_choice])

# Reference: Feature columns from training (LabelEncoder on all object columns, drop student_id)
feature_columns = [
    'age',
    'gender',
    'study_hours_per_day',
    'social_media_hours',
    'netflix_hours',
    'part_time_job',
    'attendance_percentage',
    'sleep_hours',
    'diet_quality',
    'exercise_frequency',
    'parental_education_level',
    'internet_quality',
    'mental_health_rating',
    'extracurricular_participation'
]

# LabelEncoder mappings (must match training)
label_maps = {
    'gender': {'Female': 0, 'Male': 1},
    'part_time_job': {'No': 0, 'Yes': 1},
    'diet_quality': {'Poor': 0, 'Fair': 1, 'Good': 2},
    'parental_education_level': {'High School': 0, 'Bachelor': 1, 'Master': 2},
    'internet_quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'extracurricular_participation': {'No': 0, 'Yes': 1}
}

# User input form
with st.form("input_form"):
    st.subheader("Enter Student Details:")
    age = st.number_input("Age", min_value=10, max_value=30, value=18)
    gender = st.selectbox("Gender", ["Female", "Male"])
    study_hours_per_day = st.slider("Study Hours per Day", 0, 12, 4)
    social_media_hours = st.slider("Social Media Hours per Day", 0, 12, 2)
    netflix_hours = st.slider("Netflix Hours per Day", 0, 12, 1)
    part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])
    attendance_percentage = st.slider("Attendance (%)", 0, 100, 90)
    sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
    diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
    exercise_frequency = st.slider("Exercise Frequency (days/week)", 0, 7, 3)
    parental_education = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master"])
    internet_quality = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
    mental_health_rating = st.slider("Mental Health Rating (1-10)", 1, 10, 7)
    extracurricular = st.selectbox("Extracurricular Participation", ["No", "Yes"])
    submit = st.form_submit_button("Predict Score")

# Feature engineering for input (LabelEncoder style)
def encode_input():
    input_dict = {
        'age': age,
        'gender': label_maps['gender'][gender],
        'study_hours_per_day': study_hours_per_day,
        'social_media_hours': social_media_hours,
        'netflix_hours': netflix_hours,
        'part_time_job': label_maps['part_time_job'][part_time_job],
        'attendance_percentage': attendance_percentage,
        'sleep_hours': sleep_hours,
        'diet_quality': label_maps['diet_quality'][diet_quality],
        'exercise_frequency': exercise_frequency,
        'parental_education_level': label_maps['parental_education_level'][parental_education],
        'internet_quality': label_maps['internet_quality'][internet_quality],
        'mental_health_rating': mental_health_rating,
        'extracurricular_participation': label_maps['extracurricular_participation'][extracurricular]
    }
    return np.array([input_dict[col] for col in feature_columns]).reshape(1, -1)

if submit:
    input_data = encode_input()
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Exam Score: {prediction:.2f}")
    st.info("\U0001F4AA Stay consistent with your habits for the best results!")
    st.balloons()

    # Visualizations based on input data
    st.markdown("---")
    st.markdown("### 📈 Your Academic Profile Insights")
    # Pie chart for time allocation
    time_labels = ['Study', 'Social Media', 'Netflix', 'Sleep', 'Other']
    time_values = [study_hours_per_day, social_media_hours, netflix_hours, sleep_hours, 
                   24 - (study_hours_per_day + social_media_hours + netflix_hours + sleep_hours)]
    time_values = [max(0, v) for v in time_values]
    time_df = pd.DataFrame({'Activity': time_labels, 'Hours': time_values})
    st.subheader("Daily Time Allocation")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    pd.DataFrame({'Hours': time_values}, index=time_labels).plot.pie(y='Hours', autopct='%1.1f%%', legend=False, ylabel='', ax=ax)
    st.pyplot(fig)

    # Bar chart for wellness factors
    wellness_factors = {
        'Attendance (%)': attendance_percentage,
        'Exercise (days/wk)': exercise_frequency,
        'Mental Health (1-10)': mental_health_rating
    }
    st.subheader("Wellness & Engagement Factors")
    st.bar_chart(pd.DataFrame(list(wellness_factors.values()), index=wellness_factors.keys(), columns=['Value']))

    # Display categorical choices
    st.markdown("#### Your Choices:")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Part-time Job:** {part_time_job}")
    st.write(f"**Diet Quality:** {diet_quality}")
    st.write(f"**Parental Education:** {parental_education}")
    st.write(f"**Internet Quality:** {internet_quality}")
    st.write(f"**Extracurricular Participation:** {extracurricular}")
