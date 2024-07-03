import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import bcrypt
import datetime
from sklearn.linear_model import LinearRegression 

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_insurance"]
users_collection = db["users"]
predictions_collection = db["predictions"]
# Initialize session state attributes if they do not exist
if 'username' not in st.session_state:
    st.session_state.username = ""

# Function to handle login logic
def login():
    st.session_state.username = st.text_input("Enter your username:", key='username_input')
    if st.button("Submit"):
        if st.session_state.username:
            st.success(f"Logged in as {st.session_state.username}")
        else:
            st.error("Please enter a username")

# Main app logic
if st.session_state.username:
    st.write(f"Hello {st.session_state.username}!")
    # Add your main app code here
else:
    st.write("Please log in.")
    login()


# Define a function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Define a function to check if a username already exists
def username_exists(username):
    return bool(users_collection.find_one({"username": username}))

# Define a function to register users
def register(username, password):
    if username_exists(username):
        st.error("Username already exists. Please choose a different one.")
        return
    hashed_password = hash_password(password)
    user_data = {"username": username, "password": hashed_password}
    users_collection.insert_one(user_data)
    st.success("User {} created successfully".format(username))

# Define a function to authenticate users
def authenticate(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return True
    else:
        return False
    
# Define BMI calculation function
def calculate_bmi(weight, height):
    """
    Calculate BMI (Body Mass Index).
    BMI = weight (kg) / (height (m))^2
    """
    bmi = weight / (height ** 2)
    return bmi

# Define a function to fetch past predictions for the current user
def fetch_past_predictions(username):
    return list(predictions_collection.find({"username": username}))

# Load data
ins_df = pd.read_csv('insurance.csv')

# Data preprocessing
ins_df['smoker'] = ins_df['smoker'].apply(lambda x: 0 if x == 'no' else 1)
ins_df['sex'] = ins_df['sex'].apply(lambda x: 0 if x == 'female' else 1)
X = ins_df.drop(columns=['charges'])
y = ins_df['charges']
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')
y = y.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model training
regression_model_sklearn = LinearRegression()
regression_model_sklearn.fit(X_train, y_train)

# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Main application
st.title("Medical Insurance Cost Prediction")

nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "BMI Calculator", "Contribute", "Login"])

# Session state for storing username
session_state = st.session_state
if not hasattr(session_state, "username"):
    session_state.username = None

if nav == "Home":
    # Display welcome message if user is logged in
    if session_state.username:
        st.markdown(
            f'<div style="color: green; font-size: 24px;">Welcome, {session_state.username}</div>',
            unsafe_allow_html=True
        )
    if st.checkbox("Show table"):
        st.table(ins_df)
    
    graph = st.selectbox("What kind of graph?", ["Interactive", "Non-Interactive"])
    if graph == "Non-Interactive":
        plt.figure(figsize=(10, 5))
        plt.scatter(ins_df["age"], ins_df["charges"])
        plt.ylim(0)
        plt.xlabel("Age")
        plt.ylabel("Charges")
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout = go.Layout(
            xaxis=dict(range=[18, 70]),
            yaxis=dict(range=[0, 60000])
        )
        fig = go.Figure(data=go.Scatter(x=ins_df["age"], y=ins_df["charges"], mode='markers'), layout=layout)
        st.plotly_chart(fig)
    
    
if nav == "Prediction":
    st.title("Make a Prediction")
    age = st.slider("Enter your age", 0, 100, 25)
    sex = st.radio("Enter your sex", ["Male", "Female"])
    sex = 0 if sex == "Female" else 1
    bmi = st.number_input("Enter your BMI")
    children = st.slider("How many children do you have?", 0, 10, 2)
    smoker = st.radio("Do you smoke?", ("Yes", "No"))
    smoker = 1 if smoker == "Yes" else 0
    
    input_data = np.array([age, sex, bmi, children, smoker]).reshape(1, -1)
    prediction = regression_model_sklearn.predict(input_data)
    positive_prediction = np.abs(prediction) 
    # Ensure positive value
    prediction_str = f"{positive_prediction[0][0]:,.2f} Rupees"  # Format the prediction with commas and 2 decimal places
    st.write("Your estimated medical insurance cost is:", prediction_str)
    
    # Store prediction in database
    st.write("Are you satisfied with this prediction?")
    if st.button('yes'):
         prediction_data = {
         "age": age,
         "sex": sex,
         "bmi": bmi,
         "children": children,
         "smoker": smoker,
         "predicted_cost": float(positive_prediction[0][0]),
         "timestamp": datetime.datetime.now()
         }
         predictions_collection.insert_one(prediction_data)
         st.write("Prediction stored in database.")
    else:
        pass

if nav == "BMI Calculator":
    st.title("BMI Calculator")

    # Input fields for weight and height
    weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Enter your height (m)", min_value=0.0, step=0.01)

    # Calculate BMI when "Calculate" button is clicked
    if st.button("Calculate BMI"):
        bmi = calculate_bmi(weight, height)
        st.write(f"Your BMI: {bmi:.2f}")

        # Interpret BMI result
        if bmi < 18.5:
            st.write("You are underweight.")
        elif 18.5 <= bmi < 25:
            st.write("You have a normal weight.")
        elif 25 <= bmi < 30:
            st.write("You are overweight.")
        else:
            st.write("You are obese.")

if nav == "Contribute":
    st.header("Contribute to our dataset")

    age = st.slider("Enter your age", 0, 100, 25)
    sex = st.radio("Enter your sex", ["Male", "Female"])
    sex = 0 if sex == "Female" else 1  # Convert to numerical value

    bmi = st.number_input("Enter your BMI")
    children = st.slider("How many children do you have?", 0, 10, 2)
    smoker = st.radio("Do you smoke?", ("Yes", "No"))
    smoker = 1 if smoker == "Yes" else 0  # Convert to numerical value

    charges = st.number_input("Charges")

    if st.button("Submit"):
        try:
        # Store data in MongoDB
            new_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "charges": charges
        }
            predictions_collection.insert_one(new_data)
            st.success("Data submitted successfully")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    
    
if nav == "Login":
    st.title("Login and Registration")

    page = st.radio("Go to", ["Login", "Register"])

    if page == "Login":
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                session_state.username = username  # Store username in session state
                st.success("Logged in as {}".format(username))
                # Add your app logic here for when the user is authenticated
            else:
                st.error("Invalid username or password")

    elif page == "Register":
        st.header("Register")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            register(new_username, new_password)
