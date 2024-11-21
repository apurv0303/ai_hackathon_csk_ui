import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from validate_csv import validate_input_csv

# Load your model (replace with your model loading code)
def load_model():
    # Placeholder for model loading
    model = None  # Replace with actual model
    return model

# Function to make predictions
def predict(data, model):
    # Replace with your model's prediction logic
    predictions = ["Placeholder prediction"] * len(data)  # Dummy output
    return predictions

# Load the model
model = load_model()

# Sidebar for page navigation
st.sidebar.title("Jump to Section")
page = st.sidebar.radio("Go to", ["Model Predictions", "Documentation","CloudSEK"])

if page == "Model Predictions":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212; /* Dark background for sleek look */
            color: #EDEDED; /* Light text color for contrast */
            font-family: 'Roboto', sans-serif; /* Modern font */
        }
        h1, h2, h3 {
            color: #80C7E2; /* Soft Cyan for titles and headers */
        }
        .header-text {
            font-size: 18px;
            color: #A4B3B9; /* Lighter gray text for paragraphs */
        }
        .section-title {
            color: #4CAF50; /* Green for section titles for a refreshing pop */
            font-weight: bold;
        }
        .subheader-text {
            color: #B0BEC5;
            font-size: 16px;
        }
        .input-label {
            color: #90CAF9; /* Lighter cyan for the input labels */
            font-size: 14px;
        }
        .prediction-box {
            background-color: #2C2F36; /* Slightly lighter dark background for predictions */
            border: 1px solid #424242;
            padding: 15px;
            border-radius: 8px;
            color: #80C7E2;
            font-size: 16px;
        }
        .button {
            background-color: #00796B; /* Teal button for a unique look */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #004D40;
        }
        .file-upload-box {
            background-color: #333333; /* Dark box for file upload */
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #424242;
        }
        .download-btn {
            background-color: #4CAF50; /* Green button for download */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .download-btn:hover {
            background-color: #388E3C;
        }
        .title-text {
            color: #FFFFFF; /* White color for the title */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="title-text">IndiaAI CyberGuard AI Hackathon</h1>', unsafe_allow_html=True)

    # Real-time model check section
    st.header("Real-Time Model Check")
    input_text = st.text_input("Enter your input data:", label_visibility="collapsed")
    if st.button("Predict", key="real-time-predict", help="Click to get real-time predictions"):
        if input_text:
            # Add model prediction logic for real-time inputs
            prediction = predict([input_text], model)
            st.markdown(f"<div class='prediction-box'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some input data.")

    # CSV file input for batch processing
    st.header("Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="batch-upload")
    logger.info(f"{uploaded_file}")
    if uploaded_file:
        # Read the uploaded CSV
        data = pd.read_csv(uploaded_file)

        # Validate CSV input
        if not validate_input_csv(data):
            st.error("Invalid input CSV. Please check the format and content.")
        
        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data.head(3))

        # Dropdown to show the required columns for CSV
        required_columns = ['id','input_text']  # List required columns
        st.subheader("Required Columns in CSV:")
        st.markdown(f"<span class='subheader-text'>The uploaded CSV should contain the following columns:</span>", unsafe_allow_html=True)
        st.write(required_columns)

        # # Dropdown for users to select columns from the uploaded CSV
        # selected_column = st.selectbox("Select the input column from the uploaded CSV", options=data.columns.tolist())
        # st.write(f"Selected Column: {selected_column}")

        # Dropdown to choose between models
        selected_model = st.selectbox("Choose the model", ["Model 1", "Model 2"], key="model-selection")
        

        if st.button("Process CSV", key="process-csv", help="Click to process the uploaded CSV for predictions"):
            # Check if the selected column exists in the uploaded CSV
            predictions = predict(data['input_text'], model)
            data['Prediction'] = predictions
            st.write("Prediction Results:")
            st.write(data)

            # Option to download the output CSV with predictions
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                key="download-btn",
                help="Click to download the CSV file with predictions"
            )

            # Show Accuracy button
            accuracy=98
            if st.button("Show Accuracy", key="show-accuracy"):
                st.write(f"Model Accuracy: {accuracy:.2f}%")
                
                # # Add a button next to the accuracy to show the value
                # accuracy_button_label = f"Accuracy: {accuracy:.2f}%"
                # st.button(accuracy_button_label, key="accuracy-btn")
 

elif page == "Documentation":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212; /* Dark background for sleek look */
            color: #EDEDED; /* Light text color for contrast */
            font-family: 'Roboto', sans-serif; /* Modern font */
        }
        h1, h2, h3 {
            color: #80C7E2; /* Soft Cyan for titles and headers */
        }
        .header-text {
            font-size: 18px;
            color: #A4B3B9; /* Lighter gray text for paragraphs */
        }
        .section-title {
            color: #4CAF50; /* Green for section titles for a refreshing pop */
            font-weight: bold;
        }
        .subheader-text {
            color: #B0BEC5;
            font-size: 16px;
        }
        .input-label {
            color: #90CAF9; /* Lighter cyan for the input labels */
            font-size: 14px;
        }
        .prediction-box {
            background-color: #2C2F36; /* Slightly lighter dark background for predictions */
            border: 1px solid #424242;
            padding: 15px;
            border-radius: 8px;
            color: #80C7E2;
            font-size: 16px;
        }
        .button {
            background-color: #00796B; /* Teal button for a unique look */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #004D40;
        }
        .file-upload-box {
            background-color: #333333; /* Dark box for file upload */
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #424242;
        }
        .download-btn {
            background-color: #4CAF50; /* Green button for download */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .download-btn:hover {
            background-color: #388E3C;
        }
        .title-text {
            color: #FFFFFF; /* White color for the title */
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<h1 class="title-text">Model Documentation and Analysis</h1>', unsafe_allow_html=True)

    # About the model section
    st.header("About the Model")
    st.markdown("""
    <p class="header-text">This model is designed to solve a specific problem. It uses advanced machine learning techniques to process data and generate accurate predictions. Below are the key aspects of the model:</p>
    <ul>
        <li><span class="section-title">Architecture:</span> Describe the architecture (e.g., CNN, RNN, Transformer).</li>
        <li><span class="section-title">Input Format:</span> Specify the type and format of data the model takes.</li>
        <li><span class="section-title">Output:</span> Detail the output predictions and how they are generated.</li>
        <li><span class="section-title">Use Cases:</span> Mention real-world applications for the model.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Model performance graphs
    st.markdown('<h1 class="title-text">Model Performance Graphs</h1>', unsafe_allow_html=True)
    
    # Accuracy over epochs graph
    st.subheader("Accuracy Over Epochs")
    epochs = np.arange(1, 11)  # Example epoch numbers
    accuracy = np.random.uniform(0.7, 1.0, len(epochs))  # Simulated accuracy values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, marker='o', color='blue', linewidth=2, markersize=8)
    plt.title("Model Accuracy Over Epochs", fontsize=18, color='#000000', weight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(plt)

    # Loss over epochs graph
    st.subheader("Loss Over Epochs")
    loss = np.random.uniform(0.1, 0.5, len(epochs))  # Simulated loss values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, marker='o', color='red', linewidth=2, markersize=8)
    plt.title("Model Loss Over Epochs", fontsize=18, color='#000000', weight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(plt)

    # Additional Insights section
    st.header("Additional Insights")
    st.markdown("""
    <p class="content-text">Here you can add:</p>
    <ul>
        <li><span class="section-title">Detailed explanations of model components.</span></li>
        <li><span class="section-title">Examples of inputs and outputs.</span></li>
        <li><span class="section-title">Advanced visualizations, such as confusion matrices, feature importance plots, or clustering results.</span></li>
    </ul>
    """, unsafe_allow_html=True)

elif page == "CloudSEK":
    st.markdown(
        """
        <style>
        body {
            background-color: #1a1a1a; /* Dark background */
            color: #f0f0f0; /* Light text color */
            font-family: 'Arial', sans-serif; /* Font */
        }
        h1 {
            color: #FFFFFF; /* White color for the main title */
        }
        h2 {
            color: #80C7E2; /* Soft cyan color for subheadings */
        }
        p {
            line-height: 1.6; /* Increase line height for readability */
        }
        .title-text {
            color: #FFFFFF; /* White color for the title */
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Apply the custom class to the main title
    st.markdown('<h1 class="title-text">CloudSEK</h1>', unsafe_allow_html=True)
    
    # # About the company
    # st.header("About CloudSEK")
    st.markdown("""
    At CloudSEK, we combine the power of Cyber Intelligence, Brand Monitoring, Attack Surface Monitoring, Infrastructure Monitoring, and Supply Chain Intelligence to provide a comprehensive view of digital risks. Our offerings include:
    - **Cyber Intelligence**: Utilizing advanced machine learning techniques to analyze vast amounts of data to uncover patterns and trends in cyber threats.
    - **Comprehensive Assets Tracker**: Monitor all digital assets across various platforms to ensure thorough protection against external threats.
    - **Surface, Deep, and Dark Web Monitoring**: Continuously scan the internet, including the surface, deep, and dark web, for potential threats and mentions of your organization.
    - **Integrated Threat Intelligence**: Combine threat intelligence from multiple sources for a comprehensive understanding of the external threat landscape.
    """, unsafe_allow_html=True)

    # About Team Section
    st.header("About Our Team")
    st.markdown("""
    - **Lasya**: [LinkedIn Profile](https://www.linkedin.com)
    - **Lasya**: [LinkedIn Profile](https://www.linkedin.com)
    - **Lasya**: [LinkedIn Profile](https://www.linkedin.com)
    - **Lasya**: [LinkedIn Profile](https://www.linkedin.com)
    - **Lasya**: [LinkedIn Profile](https://www.linkedin.com)
    """, unsafe_allow_html=True)