# -*- coding: utf-8 -*-
"""
Disease Prediction System
A Streamlit web application for predicting diabetes and heart disease using machine learning models
"""

import numpy as np
import pickle
import streamlit as st
import time
import requests
import smtplib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from streamlit_option_menu import option_menu
from io import StringIO
from database import db

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded", 
)

# --- Initialize session state if needed ---
if 'selected' not in st.session_state:
    st.session_state.selected = "Home"

if 'prediction_tab' not in st.session_state:
    st.session_state.prediction_tab = "None"  # Default tab for prediction

if 'prediction_confirmed' not in st.session_state:
    st.session_state.prediction_confirmed = False  # Flag for confirmation


# --- Sidebar menu ---
with st.sidebar:
    sidebar_selection = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction", "Others", "About"],
        icons=["house", "clipboard-pulse", "three-dots", "info-circle"],
        default_index=0
    )

# --- VERY IMPORTANT: Update session state correctly ---
if sidebar_selection != st.session_state.selected:
    st.session_state.selected = sidebar_selection

    # Reset prediction flow when leaving Prediction page
    if sidebar_selection != "Prediction":
        st.session_state.prediction_tab = "None"
        st.session_state.prediction_confirmed = False


# --- Page Content ---

if st.session_state.selected == "Home":
    pass

elif st.session_state.selected == "Prediction":
    #st.title("Prediction Page")

    if not st.session_state.prediction_confirmed:
        # Show prediction type selection
        prediction_choice = st.selectbox(
            "Select Prediction Type",
            ["None", "Diabetes Prediction", "Heart Disease Prediction"],
            index=0  # Default to None
        )

        if st.button("Confirm Selection"):
            if prediction_choice == "None":
                st.warning("⚠️ Please select a valid prediction type before proceeding.")
            else:
                st.session_state.prediction_tab = prediction_choice
                st.session_state.prediction_confirmed = True
                st.rerun()

    else:
        # After confirmation
        if st.session_state.prediction_tab == "Diabetes Prediction":
            pass
            # Place your Diabetes Prediction form here

        elif st.session_state.prediction_tab == "Heart Disease Prediction":
            pass
            # Place your Heart Disease Prediction form here

        if st.button("← Back to Selection"):
            st.session_state.prediction_tab = "None" 
            st.session_state.prediction_confirmed = False
            st.rerun()


elif st.session_state.selected == "Others":
   pass

elif st.session_state.selected == "About":
    pass


    

# Load the models
@st.cache_resource
def load_models():
    try:
        diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
        heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
        return diabetes_model, heart_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        

# Try to load models
try:
    diabetes_model, heart_model = load_models()
    models_loaded = True
except Exception:
    models_loaded = False

# Define diabetes prediction function
def diabetes_prediction(input_data):
    """
    Function to predict diabetes based on input parameters
    """
    try:
        # Convert input to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = diabetes_model.predict(input_data_reshaped)
        
        # Check if the model has predict_proba attribute
        if hasattr(diabetes_model, 'predict_proba'):
            prediction_probability = diabetes_model.predict_proba(input_data_reshaped)
            prob_value = prediction_probability[0][1] if prediction[0] == 1 else prediction_probability[0][0]
        else:
            # If no predict_proba, use decision_function if available, otherwise set confidence to 0.8
            if hasattr(diabetes_model, 'decision_function'):
                decision_value = diabetes_model.decision_function(input_data_reshaped)[0]
                prob_value = 1 / (1 + np.exp(-decision_value))  # Sigmoid function to convert to probability
            else:
                prob_value = 0.8  # Default confidence
        
        # Return result
        if prediction[0] == 0:
            return 'Not Diabetic', prob_value if prediction[0] == 0 else 1 - prob_value
        else:
            return 'Diabetic', prob_value if prediction[0] == 1 else 1 - prob_value
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error in prediction", 0

# Define heart disease prediction function
def heart_disease_prediction(input_data):
    """
    Function to predict heart disease based on input parameters
    """
    try:
        # Convert all inputs to float
        input_data = [float(x) for x in input_data]
        
        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = heart_model.predict(input_data_as_numpy_array)
        
        # Check if the model has predict_proba attribute
        if hasattr(heart_model, 'predict_proba'):
            prediction_probability = heart_model.predict_proba(input_data_as_numpy_array)
            prob_value = prediction_probability[0][1] if prediction[0] == 1 else prediction_probability[0][0]
        else:
            # If no predict_proba, use decision_function if available, otherwise set confidence to 0.8
            if hasattr(heart_model, 'decision_function'):
                decision_value = heart_model.decision_function(input_data_as_numpy_array)[0]
                prob_value = 1 / (1 + np.exp(-decision_value))  # Sigmoid function to convert to probability
            else:
                prob_value = 0.8  # Default confidence
        
        # Return result
        if prediction[0] == 0:
            return 'No Heart Disease', prob_value if prediction[0] == 0 else 1 - prob_value
        else:
            return 'Heart Disease Detected', prob_value if prediction[0] == 1 else 1 - prob_value
    except ValueError as e:
        return f"Error: Invalid input! {str(e)}", 0
    except Exception as e:
        return f"Unexpected Error: {str(e)}", 0

# Function to generate PDF report
def generate_pdf_report(name, email, diagnosis, disease_type, user_data):
    """
    Function to generate PDF report with test results
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import io
        import uuid
        
        # Generate a unique ID for the report
        report_id = str(uuid.uuid4())[:8]
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        subtitle_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create custom styles
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Heading2'],
            textColor=colors.darkblue,
            spaceAfter=12
        )
        
        # Initialize story (content)
        story = []
        
        # Add header
        story.append(Paragraph(f"{disease_type} report", title_style))
        story.append(Paragraph(f"Patient Report - ID: {report_id}", subtitle_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Add patient info
        story.append(Paragraph("Patient Information", header_style))
        
        # Create patient info table
        patient_data = [
            ["Name:", name],
            ["Email:", email],
            ["Report Date:", time.strftime("%Y-%m-%d")],
            ["Report ID:", report_id]
        ]
        
        patient_table = Table(patient_data, colWidths=[1.5*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Add diagnosis section
        story.append(Paragraph("Diagnosis Result", header_style))
        
        # Format diagnosis
        diagnosis_color = colors.red if "Diabetic" in diagnosis or "Disease" in diagnosis else colors.green
        diagnosis_style = ParagraphStyle(
            'DiagnosisStyle',
            parent=normal_style,
            textColor=diagnosis_color,
            fontSize=16,
            fontName='Helvetica-Bold'
        )
        
        story.append(Paragraph(f"{diagnosis}", diagnosis_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Add parameters section
        story.append(Paragraph("Health Parameters", header_style))
        
        # Create parameters table based on disease type
        param_data = []
        if disease_type == "diabetes":
            param_data = [
                ["Parameter", "Value", "Normal Range"],
                ["Gender",user_data.get('sex', ''),"Male/Female"],
                ["Pregnancies", str(user_data.get('pregnancies', 'N/A')), "N/A"],
                ["Glucose", str(user_data.get('glucose', 'N/A')), "70-99 mg/dL (fasting)"],
                ["Blood Pressure", str(user_data.get('blood_pressure', 'N/A')), "< 120/80 mmHg"],
                ["Skin Thickness", str(user_data.get('skin_thickness', 'N/A')), "Variable"],
                ["Insulin", str(user_data.get('insulin', 'N/A')), "< 25 mIU/L (fasting)"],
                ["BMI", str(user_data.get('bmi', 'N/A')), "18.5-24.9"],
                ["Diabetes Pedigree", str(user_data.get('diabetes_pedigree', 'N/A')), "Variable"],
                ["Age", str(user_data.get('age', 'N/A')), "N/A"]
            ]
        else:  # heart disease
            param_data = [
                ["Parameter", "Value", "Normal Range"],
                ["Age", str(user_data.get('age', 'N/A')), "N/A"],
                ["Sex", str(user_data.get('sex', 'N/A')), "N/A"],
                ["Chest Pain Type", str(user_data.get('chest_pain_type', 'N/A')), "N/A"],
                ["Resting BP", str(user_data.get('resting_bp', 'N/A')), "< 120/80 mmHg"],
                ["Cholesterol", str(user_data.get('cholesterol', 'N/A')), "< 200 mg/dL"],
                ["Fasting Blood Sugar", str(user_data.get('fasting_bs', 'N/A')), "< 100 mg/dL"],
                ["Resting ECG", str(user_data.get('resting_ecg', 'N/A')), "Normal"],
                ["Max Heart Rate", str(user_data.get('max_heart_rate', 'N/A')), "220 - age"],
                ["Exercise Angina", str(user_data.get('exercise_angina', 'N/A')), "No"],
                ["ST Depression", str(user_data.get('oldpeak', 'N/A')), "N/A"],
                ["ST Slope", str(user_data.get('st_slope', 'N/A')), "N/A"]
            ]
        
        param_table = Table(param_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        param_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Add recommendations section
        story.append(Paragraph("Recommendations", header_style))
        
        if disease_type == "diabetes":
            if "Diabetic" in diagnosis:
                recommendations = [
                    "Monitor blood glucose levels regularly",
                    "Follow a balanced diet low in refined carbohydrates",
                    "Engage in regular physical activity",
                    "Take prescribed medications as directed",
                    "Schedule regular check-ups with your healthcare provider"
                ]
            else:
                recommendations = [
                    "Maintain a healthy weight",
                    "Eat a balanced diet rich in fruits, vegetables, and whole grains",
                    "Exercise regularly (at least 150 minutes per week)",
                    "Limit sugary drinks and processed foods",
                    "Have your blood glucose checked annually"
                ]
        else:  # heart disease
            if "Detected" in diagnosis:
                recommendations = [
                    "Consult with a cardiologist promptly",
                    "Take prescribed medications as directed",
                    "Follow a heart-healthy diet low in sodium and saturated fats",
                    "Engage in cardiac rehabilitation if recommended",
                    "Monitor blood pressure and cholesterol regularly"
                ]
            else:
                recommendations = [
                    "Maintain a heart-healthy diet rich in fruits, vegetables, and lean proteins",
                    "Exercise regularly (at least 150 minutes per week)",
                    "Avoid smoking and limit alcohol consumption",
                    "Manage stress through relaxation techniques",
                    "Schedule regular heart health check-ups"
                ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", normal_style))
        
        story.append(Spacer(1, 0.25*inch))
        
        # Add disclaimer
        disclaimer_style = ParagraphStyle(
            'DisclaimerStyle',
            parent=normal_style,
            textColor=colors.grey,
            fontSize=8
        )
        
        disclaimer_text = """Disclaimer: This report is generated based on machine learning predictions and should not be considered a substitute for professional medical advice. Please consult with a qualified healthcare provider for proper diagnosis and treatment."""
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF value from the buffer
        pdf_value = buffer.getvalue()
        buffer.close()
        
        return pdf_value, report_id
    
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None, None

# Function to send email with test results
def send_email(name, email, diagnosis, disease_type, user_data):
    """
    Function to send email with test results to the user
    """
    try:
        sender_name = "Disease Prediction System"
        sender_email = os.getenv("EMAIL_ADDRESS", "uzerkanza05@gmail.com")
        sender_password = os.getenv("EMAIL_PASSWORD", "snsf tlna hihm yhsn")
        
        webapp_url = "https://dps-web-app-by-uzair.streamlit.app/"
        
        # Select appropriate subject based on disease type
        if disease_type == "diabetes":
            subject = "Your Diabetes Prediction Results"

            banner = """<!-- Insert the banner image -->
            <img src="{}" alt="Banner Image" style="max-width: 100%; height: auto; margin-top: 20px;">
    """.format('https://d2jx2rerrg6sh3.cloudfront.net/images/Article_Images/ImageForArticle_22744_16565132428524067.jpg')
        
        # Additional tips for diabetic patients and prevention
            additional_tips = """ 
            <p><strong><u>Tips for Diabetic Patients:</u></strong></p>
            <ol>
                <li><strong>Monitor Blood Sugar Levels:</strong><br>
        - Regularly check your blood glucose levels as advised by your healthcare provider.</li>
                <li><strong>Medication Adherence:</strong><br>
        - Take medications as prescribed by your healthcare provider.</li>
                <li><strong>Balanced Nutrition:</strong><br>
        - Adopt a diet rich in whole grains, lean proteins, fruits, and vegetables.</li>
                <li><strong>Regular Exercise:</strong><br>
        - Engage in physical activity like brisk walking, swimming, or cycling.</li>
                <li><strong>Mindful Stress Management:</strong><br>
        - Practice stress-reducing techniques, such as mindfulness, meditation, or yoga.</li>
            </ol>
            <p><strong><u>Tips for Diabetes Prevention:</u></strong></p>
            <ol>
                <li><strong>Healthy Dietary Choices:</strong><br>
        - Consume a well-balanced diet with a focus on fruits, vegetables, whole grains, and lean proteins.<br>
        - Limit the intake of processed foods, sugary drinks, and high-fat items.</li>
                <li><strong>Regular Physical Activity:</strong><br>
        - Engage in regular physical activity to maintain a healthy weight and improve insulin sensitivity.</li>
                <li><strong>Weight Management:</strong><br>
        - Aim for a body mass index (BMI) within the normal range.<br>
        - Even a small reduction in weight can significantly lower the risk of diabetes.</li>
                <li><strong>Reduce Sedentary Time:</strong><br>
        - Minimize sitting time and incorporate more movement into your daily routine.</li>
                <li><strong>Routine Health Check-ups:</strong><br>
        - Schedule regular check-ups to monitor overall health and detect any potential issues early on.</li>
            </ol>
            <p>It's important to note that these tips should be personalized based on individual health conditions and preferences. Consultation with healthcare professionals is crucial for tailored advice and management.</p>
    """
        else:  # heart disease
            subject = "Your Heart Disease Prediction Results"

            banner = """<!-- Insert the banner image -->
<img src="{}" alt="Banner Image" style="max-width: 100%; height: auto; margin-top: 20px;">
""".format('https://www.labiotech.eu/wp-content/uploads/2023/05/Cure-for-cardiovascular-diseases.jpg')  # Replace with a valid image URL
    
    # Additional tips for heart patients and prevention
            additional_tips = """ 
            <p><strong><u>Tips for heart disease Patients:</u></strong></p>
            <ol>
                <li><strong>Heart-Healthy Diet:</strong><br>
        - Choose a diet rich in fruits, vegetables, whole grains, and lean proteins.<br>
        - Limit saturated and trans fats, cholesterol, and sodium.</li>
                <li><strong>Regular Exercise:</strong><br>
        - Engage in aerobic exercises like walking, jogging, or swimming for at least 150 minutes per week.<br>
        - Include strength training exercises to improve overall cardiovascular health.</li>
                <li><strong>Manage Blood Pressure:</strong><br>
        - Monitor blood pressure regularly and follow your healthcare provider's recommendations.<br>
        - Maintain a healthy weight and limit alcohol intake.</li>
                <li><strong>Quit Smoking:</strong><br>
        - If you smoke, quit. Smoking is a major risk factor for heart disease.</li>
                <li><strong>Manage Stress:</strong><br>
        - Practice stress-reducing techniques, such as mindfulness, meditation, or yoga.</li>
            </ol>
            <p><strong><u>Tips for heart Prevention:</u></strong></p>
            <ol>
                <li><strong>Regular Health Check-ups:</strong><br>
        - Monitor cholesterol levels, blood pressure, and other cardiovascular risk factors.<br>
        - Follow your healthcare provider's advice for preventive screenings.</li>
                <li><strong>Limit Alcohol Intake:</strong><br>
        -  If you drink alcohol, do so in moderation.</li>
                <li><strong>Maintain Optimal Blood Sugar Levels:</strong><br>
        - Keep blood sugar levels within the recommended range, as diabetes can contribute to heart disease.</li>
                <li><strong>Stay Hydrated:</strong><br>
        - Maintain proper hydration for overall health and heart function.</li>
        
            </ol>
            <p>It's important to note that these tips should be personalized based on individual health conditions and preferences. Consultation with healthcare professionals is crucial for tailored advice and management.</p>
    """
        
        # Set message color based on diagnosis
        if "Not" in diagnosis or "No" in diagnosis:
            color = "green"
        else:
            color = "red"
        
        # Create email body
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ margin: 0 auto; max-width: 600px; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; }}
                .result {{ font-size: 18px; margin: 20px 0; }}
                .footer {{ font-size: 12px; color: #6c757d; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <p>Dear {name},</p>
                <p>Thank you for using our Disease Prediction System. Below is your test result:</p>
                <div class="result">
                    <p><span style="color:{color}; font-weight:bold; font-size:20px;">{diagnosis}</span></p>
                </div>
                <p>{banner}</p>
                <p>{additional_tips}</p>
                <p>Please find attached a detailed PDF report of your results.</p>
                <p>Visit our web application for more information: <a href="{webapp_url}">{webapp_url}</a></p>
                
                <div class="footer">
                    <p>This is an automated message. Please do not reply.</p>
                    <p>Note: This prediction is based on machine learning models and should not replace professional medical advice.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create MIME message
        message = MIMEMultipart()
        message["From"] = f"{sender_name} <{sender_email}>"
        message["To"] = email
        message["Subject"] = subject
        message.attach(MIMEText(body, "html"))
        
        # Generate PDF report
        pdf_data, report_id = generate_pdf_report(name, email, diagnosis, disease_type, user_data)
        
        if pdf_data:
            # Attach PDF report
            pdf_attachment = MIMEApplication(pdf_data, _subtype="pdf")
            pdf_attachment.add_header(
                "Content-Disposition", 
                "attachment", 
                filename=f"Disease_Prediction_Report_{report_id}.pdf"
            )
            message.attach(pdf_attachment)
        
        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# Function to get user's prediction history based on email
def get_user_history(email, disease_type):
    try:
        if disease_type == "diabetes":
            history = db.get_diabetes_predictions_by_email(email)
        else:  # heart disease
            history = db.get_heart_disease_predictions_by_email(email)
        
        return history
    except Exception as e:
        st.error(f"Error retrieving history: {e}")
        return pd.DataFrame()

# Main content area
def main():             
    # Use the selection from session state
    selected = st.session_state.selected
    
    # HOME PAGE
    if selected == "Home":
        st.image("home image.png")
        
        # Introduction
        st.markdown("""
        ## Welcome to the Disease Prediction System
        
        This web application helps you predict the likelihood of diseases like **Diabetes** and **Heart Disease** using  machine learning algorithms.
        
        ### How it works
        1. Go to prediction tab             
        2. Select either Diabetes Prediction or Heart Disease Prediction from the sidebar
        3. Enter your health parameters
        4. Get an instant prediction of your risk level
        5. Get instant result on your enterd email id with  prevenation and health tips
        
        ### Features
        - **Disease Prediction**: Get predictions for diabetes and heart disease
        - **Educational Content**: Learn about disease risk factors and prevention
        - **Email Results**: Get your results sent to your email as a detailed PDF report
        - **History Tracking**: View your past predictions by entering your email
        
        ### Why use this system?
        - Early detection of potential health risks
        - Better understanding of your health status
        - Educational resources about disease prevention
        - Quick and easy to use interface
        
        
        """)
        
        # Disclaimer
        st.subheader("⚠️ Disclaimer")
        st.markdown("""
        - This web application provides predictions based on machine learning models and should not be considered a substitute for professional medical advice, diagnosis, or treatment.
        - Always consult with a qualified healthcare provider for medical concerns.
        - Your personal information and test results are kept confidential.
        """)

        # Display sample visualizations
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.subheader("Diabetes Information")
        #     st.image("diabetes home image.jpg", use_container_width=True)
        #     st.write("Diabetes is a chronic health condition that affects how your body processes glucose (blood sugar).")
        #     if st.button("Go to Diabetes Prediction", key="diabetes_btn",type="primary"):
        #         # Update selected value 
        #         st.session_state.selected == "Prediction"
        #         st.session_state.prediction_tab = "Diabetes Prediction"
        #         st.session_state.prediction_confirmed = True
        #         st.rerun()
        
        # with col2:
        #     st.subheader("Heart Disease Information")
        #     st.image("heart home image.png", use_container_width=True)
        #     st.write("Heart disease refers to various conditions that affect the heart's structure and function.")
        #     if st.button("Go to Heart Disease Prediction", key="heart_btn",type="primary"):
        #         # Update selected value 
        #         st.session_state.selected == "Prediction"
        #         st.session_state.prediction_tab = "Heart Disease Prediction"
        #         st.session_state.prediction_confirmed = True
        #         st.rerun()
        
        
    
    # DIABETES PREDICTION PAGE
    elif st.session_state.prediction_tab == "Diabetes Prediction":
        st.title("Diabetes Prediction")
        
        tab1, tab2, tab3 = st.tabs(["Make Prediction", "About Diabetes", "Your History"])
        
        with tab1:
            name = st.text_input("Name", key="diabetes_name", autocomplete="off")
            email = st.text_input("Email Address", key="diabetes_email", autocomplete="off")

        
            if not name or not email:
                st.warning('Please enter both Name and Email to proceed!', icon="⚠️")
            else:
                if not name.replace(" ", "").isalpha():
                    st.error("Invalid name ❌ Please enter letters only.")
                    st.stop()
                email_type = ('@gmail.com', '@yahoo.com', '@outlook.com')
                if not email.endswith(email_type):
                    st.error("Invalid email address!", icon="❌")
                else:
                    st.caption("**Having confusion in giving inputs? Navigate to the Options menu in the top-left corner and click on **Others** for more information.")    
            
            # Create columns for the form
                    col1, col2 = st.columns(2)
            
            # Get user input
                    with col1:
                        sex = st.selectbox("Gender", ["Female", "Male"])
                        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0,disabled=sex=="Male")
                        glucose = st.slider("Glucose Level (mg/dL)", 0, 199, 100)
                        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
                        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
            
                    with col2:
                        insulin = st.slider("Insulin (µU/ml)", 0, 846, 80)
                        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
                        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0,    value=0.5, step=0.01)
                        age = st.slider("Age", 0, 100, 30)
            
            # Create a button for prediction
                    if st.button("Predict Diabetes"):
                # Ensure all fields are filled
                    # Make prediction
                        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
                    
                        if models_loaded:
                            diagnosis, _ = diabetes_prediction(input_data)

                        # Display result with progress bar and spinner
                            with st.spinner("Processing..."):
                                progress_bar = st.progress(0)
                                for i in range(100):
                                    time.sleep(0.01)
                                    progress_bar.progress(i + 1)
                            
                            # Show result
                                st.success("Prediction Complete!")
                            
                                if "Not" in diagnosis:
                                    st.markdown(f"### Result: <span style='color:green'>{diagnosis}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"### Result: <span style='color:red'>{diagnosis}</span>", unsafe_allow_html=True)

                                with st.expander("Click here to see Test Report"):
                                        st.markdown("### Patient Details")
                                        st.markdown(f"**Patient Name:** {name}")
                                        st.markdown(f"**Gender:** {sex}")
                                        st.markdown(f"**Age:** {age}")

                                    # Dynamic Data for Table
                                        data = {
                                        "Parameter Name": ["name","gender","Age", "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function"],
                                        "Patient Values": [(f"{name}"),sex,age, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree],
                                        "Normal Range": ["character","male/female","0-100", "0-10", "70-125", "120/80", "8-25", "25-250", "18.5-24.9", "< 1"],
                                        "Unit": ["string","string","years", "Number", "mg/dL", "mmHg", "mm", "mIU/L", "kg/m²", "No units"]
                                    }

                                    # Display Table
                                        import pandas as pd
                                        df = pd.DataFrame(data)
                                        st.dataframe(df, use_container_width=True)
                                        st.info('Do check your email for more details, Thank You.', icon="ℹ️")    
                            
                            # Save user data for report and database
                                user_data = {
                                'name': name,
                                'sex' : sex,
                                'email': email,
                                'pregnancies': pregnancies,
                                'glucose': glucose,
                                'blood_pressure': blood_pressure,
                                'skin_thickness': skin_thickness,
                                'insulin': insulin,
                                'bmi': bmi,
                                'diabetes_pedigree': diabetes_pedigree,
                                'age': age
                            }
                            
                            # Save prediction to database
                                diagnosis, _ = diabetes_prediction(input_data) 
                                db.save_diabetes_prediction(user_data, diagnosis)
                                st.success("Prediction saved to database!")
                            
                            # Attempt to send email with result
                                st.info("Sending results to your email...")
                                if send_email(name, email, diagnosis, "diabetes", user_data):
                                    st.success("Email with PDF report sent successfully!")
                                else:
                                    st.warning("Could not send email. Please check your email address.")
                        else:
                            st.error("Model loading error. Unable to make prediction.")
                
        st.markdown(
            """<div style="position: fixed; bottom: 7.6px; left: 10px; right: 10px; text-align: left; color: grey; font-size: 14px;">
            Made by <span style="font-weight: bold; color: grey;">uzair</span>🎈
            </div>""",
            unsafe_allow_html=True
        )               
        
        with tab2:
            st.subheader('What is Diabetes?')
            st.write(
        """
        Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. 
        It occurs when your blood glucose (sugar) levels are too high. Over time, having too much sugar in your 
        blood can cause serious health problems, such as heart disease, vision loss, and kidney disease.
        """)

            st.subheader('Types of Diabetes:')
            st.write(
        """
        1. **Type 1 Diabetes:** 
           - A condition where the body produces little or no insulin. 
           - Often diagnosed in children and young adults.
           - Requires daily insulin injections.
        
        2. **Type 2 Diabetes:** 
           - A condition where the body doesn’t use insulin properly (insulin resistance).
           - Most common type of diabetes.
           - Often associated with obesity and lifestyle factors.
        
        3. **Gestational Diabetes:** 
           - Diabetes that occurs during pregnancy.
           - Usually resolves after childbirth but increases the risk of Type 2 diabetes later in life.
        """)

            st.subheader('Symptoms of Diabetes:')
            st.markdown(
        """
        - **Frequent urination:** Excess glucose in the blood causes the kidneys to work harder to filter it out.
        - **Increased thirst:** Frequent urination leads to dehydration, causing excessive thirst.
        - **Unexplained weight loss:** The body starts burning fat and muscle for energy when it can't use glucose properly.
        - **Fatigue:** Lack of glucose in the cells leads to low energy levels.
        - **Blurred vision:** High blood sugar levels can cause swelling in the lenses of the eyes.
        - **Slow-healing sores:** Diabetes affects blood circulation and the body's ability to heal.
        """)

            st.subheader('Risk Factors for Diabetes:')
            st.markdown(
        """
        - **Family history:** Having a parent or sibling with diabetes increases your risk.
        - **Obesity:** Excess body fat, especially around the abdomen, is a major risk factor.
        - **Sedentary lifestyle:** Lack of physical activity contributes to insulin resistance.
        - **Age:** Risk increases with age, especially after 45.
        - **Ethnicity:** Certain ethnic groups (e.g., African American, Hispanic, Native American) are at higher risk.
        - **Gestational diabetes:** Women who had diabetes during pregnancy are at higher risk of developing Type 2 diabetes later.
        """ )

            st.subheader('Prevention and Management:')
            st.write(
        """
        - **Maintain a healthy diet:** Focus on whole grains, fruits, vegetables, lean proteins, and healthy fats.
        - **Exercise regularly:** Aim for at least 30 minutes of moderate exercise most days of the week.
        - **Monitor blood sugar levels:** Regular monitoring helps you understand how food, activity, and medication affect your blood sugar.
        - **Take prescribed medications:** Follow your doctor's advice on medications or insulin therapy.
        - **Avoid smoking and limit alcohol:** Smoking and excessive alcohol consumption can worsen diabetes complications.
        """)

            st.subheader('Complications of Diabetes:')
            st.markdown(
        """
        - **Cardiovascular disease:** Diabetes increases the risk of heart attack, stroke, and high blood pressure.
        - **Nerve damage (neuropathy):** High blood sugar can damage nerves, leading to pain, tingling, or numbness.
        - **Kidney damage (nephropathy):** Diabetes can damage the kidneys, potentially leading to kidney failure.
        - **Eye damage (retinopathy):** Diabetes can damage the blood vessels in the retina, leading to blindness.
        - **Foot problems:** Poor circulation and nerve damage can lead to foot ulcers and infections.
        """)

            st.subheader('Myths and Facts About Diabetes:')
            st.markdown(
        """
        - **Myth:** Eating too much sugar causes diabetes.
          **Fact:** While sugar intake is a factor, diabetes is caused by a combination of genetic and lifestyle factors.
        
        - **Myth:** People with diabetes can't eat sweets.
          **Fact:** Sweets can be eaten in moderation as part of a balanced diet.
        
        - **Myth:** Only overweight people get diabetes.
          **Fact:** While obesity is a risk factor, people of any weight can develop diabetes.
        """)

            st.subheader('Resources and Further Reading:')
            st.markdown(
        """
        - [indian Diabetes Association](https://idf.org/our-network/regions-and-members/south-east-asia/members/india/diabetic-association-of-india/)
        - [World Health Organization (WHO) - Diabetes](https://www.who.int/health-topics/diabetes)
        - [National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)](https://www.niddk.nih.gov/)
        """)

            # Load the image and provide download option
            
            st.subheader('Diabetes Disease Infographic')
            with open("Diabetes_Infographics.png", "rb") as file:
                btn = st.download_button(
                label="⬇️ Download Infographic",
                data=file,
                file_name="Diabetes_Infographic.jpg",
                mime="image/jpeg")


            
            st.info(
        """
        If you suspect you have diabetes or are at risk, consult a healthcare professional for proper diagnosis and treatment. 
        """)    
            
        with tab3:
            st.subheader("View Your Diabetes Prediction History")

    # Input for email
            history_email = st.text_input("Enter your email address to view your diabetes prediction history", autocomplete="off")

            if history_email and "@" in history_email and "." in history_email.split("@")[1]:
                
                if st.button("Get Diabetes History"):
                    
        # Fetch history from database
                     history_data = db.get_diabetes_predictions_by_email(history_email)

                     if not history_data.empty:
                         
            # Format datetime properly
                        history_data['prediction_date'] = pd.to_datetime(history_data['prediction_date'])
                        history_data['prediction_date'] = history_data['prediction_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Count predictions
                       prediction_counts = history_data['prediction'].value_counts()
                       diabetes_count = prediction_counts.get(1, 0)
                       no_diabetes_count = prediction_counts.get(0, 0)
                       total_user_records = diabetes_count + no_diabetes_count

                # Centered metric display
                        col1, = st.columns(1)
                        with col1:
                            st.metric(
                                label="Your Diabetes Disease Predictions",
                                value=total_user_records,
                                delta=f"{(diabetes_count / total_user_records * 100):.1f}% Positive"
                                if total_user_records > 0 else "0% Positive"
                            )

                        st.success(f" Found {len(history_data)} diabetes records for {history_email}")
                        st.dataframe(history_data)

                # CSV download
                        csv = history_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Your Diabetes History as CSV",
                            data=csv,
                            file_name="diabetes_history.csv",
                            mime="text/csv",
                        )

                
                # Donut pie chart
                        # Update the labels and values for the pie chart
                        if total_user_records > 0:
                            labels = ['Diabetes Disease Detected', 'No Diabetes Disease']
                            values = [diabetes_count, no_diabetes_count]
    
    # Only create chart if we have at least one non-zero value
                            if diabetes_count > 0 or no_diabetes_count > 0:
                                fig = px.pie(
                                    names=labels,
                                    values=values,
                                    title="Your Diabetes Prediction Distribution",
                                    hole=0.3,
                                    color=labels,
                                    color_discrete_map={
                                        'Diabetes Disease Detected': 'tomato',
                                        'No Diabetes Disease': 'mediumseagreen'
                                    }
                                )
                                fig.update_traces(textfont_size=16, pull=[0, 0])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No prediction data available to generate chart.")

                    else:
                        st.warning(f"⚠️ No diabetes prediction history found for {history_email}")
            else:
                st.info("ℹ️ Please enter a valid email address to retrieve your prediction history.")
    
    # HEART DISEASE PREDICTION PAGE
    elif st.session_state.prediction_tab  == "Heart Disease Prediction":
        st.title("Heart Disease Prediction")
        
        tab1, tab2, tab3 = st.tabs(["Make Prediction", "About Heart Disease", "Your History"])
        
        with tab1:
            name = st.text_input("Name", key="heart_name", autocomplete="off")
            email = st.text_input("Email Address", key="heart_email", autocomplete="off")

            if not name or not email:
                st.warning('Please enter both Name and Email to proceed!', icon="⚠️")
            else:
                if not name.replace(" ", "").isalpha():
                    st.error("Invalid name ❌ Please enter letters only.")
                    st.stop()
                email_type = ('@gmail.com', '@yahoo.com', '@outlook.com')
                if not email.endswith(email_type):
                    st.error("Invalid email address!", icon="❌")
                else:
                    st.caption("**Having confusion in giving inputs? Navigate to the Options menu in the top-left corner and click on **Others** for more information.")
            
            # Create columns for the form
                    col1, col2 = st.columns(2)
            
            # Get user input
                    with col1:
                        age = st.slider("Age", 20, 100, 50, key="heart_age")
                        sex = st.selectbox("Gender", ["Female", "Male"])
                        cp_options = {
                        "Typical Angina": 0,
                        "Atypical Angina": 1,
                        "Non-Anginal Pain": 2,
                        "Asymptomatic": 3}
                        cp = st.selectbox("Chest Pain Type", list(cp_options.keys()))
                        cp_value = cp_options[cp]
                        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
                        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)
                        fbs_options = ["Less than 120 mg/dl", "Greater than 120 mg/dl"]
                        fbs = st.selectbox("Fasting Blood Sugar", fbs_options)
                        fbs_value = 1 if fbs == fbs_options[1] else 0
                        restecg_options = {
                        "Normal": 0,
                        "ST-T Wave Abnormality": 1,
                        "Left Ventricular Hypertrophy": 2}
                        restecg = st.selectbox("Resting ECG", list(restecg_options.keys()))
                        restecg_value = restecg_options[restecg]
            
                    with col2:
                        
                        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
                        exang_options = ["No", "Yes"]
                        exang = st.selectbox("Exercise Induced Angina", exang_options)
                        exang_value = 1 if exang == "Yes" else 0
                        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)
                        slope_options = {
                        "Upsloping": 0,    
                        "Flat": 1,
                        "Downsloping": 2}
                        slope = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_options.keys()))
                        slope_value = slope_options[slope]
                        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
                        thal_options = {
                            "Normal": 1,
                            "Fixed Defect": 2,
                            "Reversible Defect": 3}
                        thal = st.selectbox("Thalassemia", list(thal_options.keys()))
                        thal_value = thal_options[thal]
            
            # Create a button for prediction
                    if st.button("Predict Heart Disease"):
                # Ensure all fields are filled
                    # Make prediction
                        input_data = [age, 1 if sex == "Male" else 0, cp_value, trestbps, chol, fbs_value, 
                                 restecg_value, thalach, exang_value, oldpeak, 
                                 slope_value, ca, thal_value]
                    
                        if models_loaded:
                            diagnosis, _ = heart_disease_prediction(input_data)
                        
                        # Display result with progress bar and spinner
                            with st.spinner("Processing..."):
                                progress_bar = st.progress(0)
                                for i in range(100):
                                    time.sleep(0.01)
                                    progress_bar.progress(i + 1)
                            
                            # Show result
                                st.success("Prediction Complete!")
                            
                                if "No" in diagnosis:
                                    st.markdown(f"### Result: <span style='color:green'>{diagnosis}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"### Result: <span style='color:red'>{diagnosis}</span>", unsafe_allow_html=True)

                    # Expandable Test Report Section
                                with st.expander("Click here to see Test Report"):
                                    st.markdown("### Patient Details")
                                    st.markdown(f"**Patient Name:** {name}")
                                    st.markdown(f"**Gender:** {sex}")
                                    st.markdown(f"**Age:** {age}")

                                # Dynamic Data for Table
                                    data = {
                                    "Parameter Name": ["name","Age", "Gender", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol", "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina", "ST Depression", "Slope", "Major Vessels", "Thalassemia"],
                                    "Patient Values": [(f"{name}"),age, sex, cp, trestbps, chol, "True" if fbs == 1 else "False", restecg, thalach, "Yes" if exang == 1 else "No", oldpeak, slope, ca, thal],
                                    "Normal Range": ["character","0-100", "0 (Female), 1 (Male)", "0-3", "90-120", "<200", "0 (False), 1 (True)", "0-2", "60-200", "0 (No), 1 (Yes)", "0.0-6.2", "0-2", "0-3", "0-3"],
                                    "Unit": ["string","Years", "Binary", "Category", "mmHg", "mg/dL", "Binary", "Category", "bpm", "Binary", "mm", "Category", "Count", "Category"]
                                }
                        # Display Table
                                    import pandas as pd
                                    df = pd.DataFrame(data)
                                    st.dataframe(df, use_container_width=True)
                                    st.info('Do check your email for more details, Thank You.', icon="ℹ️")    
                            
                            # Save user data for report and database
                                user_data = {
                                'name': name,
                                'email': email,
                                'age': age,
                                'sex': sex,
                                'chest_pain_type': cp,
                                'resting_bp': trestbps,
                                'cholesterol': chol,
                                'fasting_bs': fbs,
                                'resting_ecg': restecg,
                                'max_heart_rate': thalach,
                                'exercise_angina': exang,
                                'oldpeak': oldpeak,
                                'st_slope': slope,
                                'major_vessels': ca,
                                'thalassemia': thal
                            }
                            
                            # Save prediction to database
                                diagnosis, _ = heart_disease_prediction(input_data)
                                db.save_heart_disease_prediction(user_data, diagnosis)
                                st.success("Prediction saved to database!")
                            
                            # Attempt to send email with result
                                st.info("Sending results to your email...")
                                if send_email(name, email, diagnosis, "heart", user_data):
                                    st.success("Email with PDF report sent successfully!")
                                else:
                                    st.warning("Could not send email. Please check your email address.")
                        else:
                            st.error("Model loading error. Unable to make prediction.")
        st.markdown(
            """<div style="position: fixed; bottom: 7.6px; left: 10px; right: 10px; text-align: left; color: grey; font-size: 14px;">
            Made by <span style="font-weight: bold; color: grey;">uzair</span>🎈
            </div>""",
            unsafe_allow_html=True
        )       
        
        with tab2:
            st.subheader('What is Heart Disease?')
            st.write(
            """
            Heart disease refers to various types of conditions that affect the heart's structure and function.
            It is one of the leading causes of death worldwide.
            """
        )

            st.subheader('Types of Heart Disease:')
            st.write("""
            1. **Coronary Artery Disease (CAD):** Blockage of the heart’s major blood vessels.  
            2. **Heart Arrhythmias:** Irregular heartbeats.  
            3. **Heart Valve Disease:** Malfunctioning of heart valves.  
            4. **Congenital Heart Defects:** Heart problems present at birth.
            """)

            st.subheader('Symptoms of Heart Disease:')
            st.markdown("""
            - **Chest pain (Angina):** Discomfort or pressure in the chest.
            - **Shortness of breath:** Difficulty breathing, especially after physical activity.
            - **Fatigue:** Unusual tiredness due to reduced blood flow.
            - **Dizziness or fainting:** May indicate poor circulation or irregular heartbeats.
            - **Swelling in legs or feet:** Due to fluid retention from heart failure.
            - **Heart palpitations:** Rapid or irregular heartbeat.
            """)


            st.subheader('Prevention:')
            st.write("""
            - Eat a healthy diet  
            - Exercise regularly  
            - Manage stress  
            - Avoid smoking and alcohol
            """)
            
            st.subheader('Causes of Heart Disease:')
            st.write("""
            - **High Blood Pressure (Hypertension):** Can damage arteries over time.
            - **High Cholesterol:** Leads to plaque buildup in arteries.
            - **Smoking:** Increases the risk of heart disease.
            - **Diabetes:** Can contribute to artery damage.
            - **Obesity:** Extra weight puts strain on the heart.
            - **Genetics:** Family history can play a role in heart disease.
            """)
            
            st.subheader('Risk Factors:')
            st.write("""
            - **Unhealthy Diet:** Diets high in salt, sugar, and unhealthy fats contribute to heart disease.
            - **Lack of Physical Activity:** Sedentary lifestyles increase risk.
            - **Chronic Stress:** Long-term stress can elevate blood pressure.
            - **Excess Alcohol Consumption:** Weakens heart muscles over time.
            """)
            
            st.subheader('Complications of Heart Disease:')
            st.markdown("""
            - **Heart Attack:** Caused by blocked arteries.  
            - **Stroke:** Reduced blood supply to the brain.  
            - **Heart Failure:** When the heart can't pump blood effectively.  
            - **Kidney Damage:** Poor blood circulation affects kidney function.  
            - **Aneurysm:** Artery walls weaken and bulge, which can be fatal.  
            """)
            
            st.markdown("""
🔗             **Learn More:**  
            - [indian Heart Association](https://indianheartassociation.org/)  
            - [World Health Organization - Heart Disease](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))  
            """)

            
            

            st.subheader('Heart Disease Infographic')
                        
            # Path to the uploaded image file

            image_path="Heart_Disease_Infographic.jpg"

            with open("Heart_Disease_Info.png", "rb") as file:
                st.download_button(
                label="⬇️ Download Infographic",
                data=file,
                file_name="Heart_Disease_Infographic.jpg",
                mime="image/jpeg") 

            st.info("For more information, consult a cardiologist.")    
    
        
        with tab3:
            st.subheader("View Your Heart Disease Prediction History")

    # Input for email
            history_email = st.text_input("Enter your email address to view your heart disease prediction history", autocomplete="off")

            if history_email and "@" in history_email and "." in history_email.split("@")[1]:
                if st.button("Get Heart Disease History"):
            # Fetch user-specific history from database
                    history_data = get_user_history(history_email, "heart_disease")

                    

                    if not history_data.empty:
                        # Count personal prediction stats
                        prediction_counts = history_data['prediction'].value_counts()
                        heart_disease_count = prediction_counts.get('Heart Disease Detected', 0)
                        no_heart_disease_count = prediction_counts.get('No Heart Disease', 0)
                        total_user_records = heart_disease_count + no_heart_disease_count

                        col1, = st.columns(1)
                        with col1: 
                            st.metric(
                                label="Your Heart Disease Predictions",
                                value=total_user_records,
                                delta=f"{(heart_disease_count / total_user_records * 100):.1f}% Positive"
                                if total_user_records > 0 else "0% Positive"
                            )
                        st.success(f" Found {len(history_data)} heart disease records for {history_email}")
                        st.dataframe(history_data)

                        # CSV download
                        csv = history_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Your Heart Disease History as CSV",
                            data=csv,
                            file_name="heart_disease_history.csv",
                            mime="text/csv",
                        )

    # Show pie chart for this user only
                        if total_user_records > 0:
                            labels = ['Heart Disease Detected', 'No Heart Disease']
                            values = [heart_disease_count, no_heart_disease_count]
                            hole=0.3,
                            fig = px.pie(
                                names=labels,
                                values=values,
                                title="Your Prediction Distribution",
                                color=labels,
                                color_discrete_map={
                                    'Heart Disease Detected': 'tomato',
                                    'No Heart Disease': 'mediumseagreen'
                                },
                                # color_discrete_sequence=px.colors.sequential.Plasma
                                
                            )
                            fig.update_traces(textfont_size=16, pull=[0, 0])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No prediction data available for pie chart.")

                    else:
                        st.warning(f"⚠️ No diabetes prediction history found for {history_email}")
            else:
                st.info("ℹ️ Please enter a valid email address to retrieve your prediction history.")

    
    # ABOUT PAGE
    elif selected == "About":
        st.title("About the Disease Prediction System")
        
        st.markdown("""
        ## Project Overview
        
        The Disease Prediction System is a machine learning-based web application designed to predict the likelihood of diabetes and heart disease based on personal health metrics. By providing accessible screening tools, this system aims to increase awareness about these conditions and encourage early detection and preventive measures.
        
        ## How It Works
        
        The application uses trained machine learning models to analyze user-provided health data and generate risk assessments. The prediction process involves:
        
        1. **Data Collection**: Users input their health parameters through an intuitive interface.
        2. **Data Processing**: The system preprocesses and normalizes the input data.
        3. **Model Prediction**: The trained models analyze the data to generate predictions.
        4. **Results Presentation**: Users receive clear results in form of pdf.
        
        ## Machine Learning Models
        
        The system employs the following machine learning models:
        
        - **Diabetes Prediction**: Uses a Support Vector Machine (SVM) classifier trained on the Pima Indians Diabetes Dataset.
        - **Heart Disease Prediction**: Uses a Logistic Regression model trained on the UCI Heart Disease dataset.
        
        ## Technologies Used
        
        - **Backend**: Python, scikit-learn for machine learning models
        - **Frontend**: Streamlit for the web interface
        - **Database**: SQLite for data storage
        - **Reporting**: ReportLab for PDF generation
        
        ## Privacy and Data Security
        
        User data is stored securely and used solely for the purpose of generating predictions and improving the system's accuracy. Email addresses are only used for sending prediction results and are not shared with any third parties.
        
        ## Credits
        
        This project was developed by Uzair Kanza.
        
        ## Disclaimer
        
        The predictions provided by this system are for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for any health concerns.
        """)
    
    # OTHER PAGE
    elif(selected == 'Others'):

        tab1, tab2, tab3 = st.tabs(["❓Help", "💬 Feedback", "📩 Contact"])
        
        with tab1:
            placeholder = st.empty()
            st.header("Welcome to the Help Page!",divider='rainbow')
            st.subheader("Diabetes help section:")
            st.write("This application is designed to predict whether a person is diabetic or not based on input data such as the number of pregnancies, glucose level, blood pressure, and other relevant factors.")
            st.write("It works in real-time with 90% accuracy, since it is built using a trained and tested machine learning model.")
            st.write("If you possess true values for pregnancies, BMI, insulin, etc., enter them for precise predictions.")
            st.write("To experience how the application functions, you can use the diabetes data values from kaggle provided below.")
            st.write("In the table Outcome, the Diabetes Status is represented as follows:")
            st.markdown("""
            - **1 indicates that the person is Diabetic.**
            - **0 indicates that the person is Non-Diabetic.**
            """
            )

            # diabetes data table
            Diabetes_data = {
                "Pregnancies": [6, 1, 8, 1, 0, 5, 3, 10, 2],
                "Glucose": [148, 85, 183, 89, 137, 116, 78, 115, 197],
                "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 0, 70],
                "SkinThickness": [35, 29, 0, 23, 35, 0, 32, 0, 45],
                "Insulin": [0, 0, 0, 94, 168, 0, 88, 0, 543],
                "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31, 35.3, 30.5],
                "PedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158],
                "Age": [50, 31, 32, 21, 33, 30, 26, 29, 53],
                "Outcome": [1,0,1,0,1,0,1,0,1]
            }

            # Convert sample data to a Pandas DataFrame for better tabular display
            import pandas as pd
            diab_df = pd.DataFrame(Diabetes_data)

            st.caption("Diabetes Data:")
            # st.dataframe(sample_df)
            st.table(diab_df)


            st.subheader("Diabetes Disease Data Info")

            st.write("Download the **Diabetes Disease Data** PDF to learn more about key health indicators, symptoms, and prevention methods.")

            # Path to the uploaded PDF file
            pdf_path = "Diab_Data_info.pdf"

            # Display Download Button
            with open("Diab_Data_info.pdf", "rb") as file:
                st.download_button(
                    label="⬇️ Download Diabetes Disease Data info PDF",
                    data=file,
                    file_name="Diab_Data_info.pdf",
                    mime="application/pdf")

            #for heart 
            placeholder = st.empty()
            with placeholder.container():
                st.header("",divider='rainbow')
                st.subheader("Heart Disease help section:")
                st.write("This application is designed to predict whether a person is heart disease or not based on input data such as the age,  gender, chest pain type , resting blood pressure, and other relevant factors.")
                st.write("It works in real-time with 90% accuracy, since it is built using a trained and tested machine learning model.")
                st.write("If you possess true values for age, gender, chest pain type, etc., enter them for precise predictions.")
                st.write("To experience how the application functions, you can use the heart data values from kaggle provided below.")
                st.write("In the table Outcome, the Diabetes Status is represented as follows:")
                st.markdown("""
            - **1 indicates that the person is Heart Disease.**
            - **0 indicates that the person is not Heart Disease.**
            """
            )


            
            #heart data table   
            import pandas as pd         
            heart_data ={
                "age": [63, 37, 41, 56,57,45,68,57,57],
                "Gender": [1, 1, 0, 1, 0,1,1,1,0],
                "cp": [3, 2, 1, 1, 0,3,0,0,1],
                "trestbps": [145, 130, 130, 120, 140,110,144,130,130],
                "chol": [233, 250, 204, 236, 241,264,193,131,236],
                "fbs": [1, 0, 0, 0, 0,0,1,0,0],
                "restecg": [0, 1, 0, 1, 1,1,1,1,0],
                "maximum heart rate": [150, 187, 172, 178, 123,132,141,115,174],
                "exang":[0,0,0,0,1,0,0,1,0],
                "oldpeak":[2.3,3.5,1.4,0.8,0.2,1.2,3.4,1.2,0],
                "slope":[0,0,2,2,1,1,1,1,1],
                "ca":[0,0,0,0,0,0,2,1,1],
                "thal":[1,1,2,2,3,3,3,3,2],
                "Outcome":[1,1,1,1,0,0,0,0,0]
            }

            heart_df = pd.DataFrame(heart_data)

# --------------------------
# Replace numeric codes with human-readable labels
# --------------------------
            gender_map = {0: "Female", 1: "Male"}
            cp_map = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic"
}
            fbs_map = {0: "<120 mg/dl", 1: ">120 mg/dl"}
            restecg_map = {             
    0: "Normal",
    1: "ST-T Abnormality",
    2: "Left Ventricular Hypertrophy"
}
            exang_map = {0: "No", 1: "Yes"}
            slope_map = {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }
            thal_map = {
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect"
}
            outcome_map = {0: "No Heart Disease", 1: "Heart Disease"}

# Apply mappings
            heart_df["Gender"] = heart_df["Gender"].replace(gender_map)
            heart_df["cp"] = heart_df["cp"].replace(cp_map)
            heart_df["fbs"] = heart_df["fbs"].replace(fbs_map)
            heart_df["restecg"] = heart_df["restecg"].replace(restecg_map)
            heart_df["exang"] = heart_df["exang"].replace(exang_map)
            heart_df["slope"] = heart_df["slope"].replace(slope_map)
            heart_df["thal"] = heart_df["thal"].replace(thal_map)
            heart_df["Outcome"] = heart_df["Outcome"].replace(outcome_map)

            # --------------------------
            # Display Table
            # --------------------------
            st.caption("Heart Data:")
            st.table(heart_df)
            
                

            # Convert sample data to a Pandas DataFrame for better tabular display
            # import pandas as pd
            # heart_df = pd.DataFrame(heart_data)

            # st.caption("Heart Data:")
            # # st.dataframe(heart_df)
            # st.table(heart_df)


            st.subheader("Heart Disease Data Info")

            st.write("Download the **Heart Disease Data** PDF to learn more about key health indicators, symptoms, and prevention methods.")

            # Path to the uploaded PDF file
            pdf_path = "Heart_Data_info.pdf"

            # Display Download Button
            with open("Heart_Data_info.pdf", "rb") as file:
                st.download_button(
                    label="⬇️ Download Heart Disease Data info PDF",
                    data=file,
                    file_name="Heart_Data_info.pdf",
                    mime="application/pdf")


            st.subheader("Note:")

            
            st.info(
                
                "This webpage requests your name and email to send you details about your test results.\n\n"
                "Rest assured, your information is safe and will be kept confidential."
                )
            
        with tab3:
            st.write("Email: [uzerkanza05@gmail.com](mailto:uzerkanza05@gmail.com)")
            st.write(" ")
            
            st.markdown(
            """<div style="position: fixed; bottom: 7.6px; left: 10px; right: 10px; text-align: left; color: grey; font-size: 14px;">
            Made by <span style="font-weight: bold; color: grey;">uzair</span>🎈
            </div>""",
            unsafe_allow_html=True
            ) 
            
        with tab2:


# --- Feedback Page without st.form ---

            st.subheader("Your Feedback is Valuable!", divider='rainbow')

# --- User Information ---
            feedback_name = st.text_input("Your Name*", placeholder="Enter your name",autocomplete="off")
            feedback_email = st.text_input("Your Email*", placeholder="Enter your email",autocomplete="off")

# --- Feedback Category ---
            feedback_category = st.selectbox(
    "Feedback Category",
    ["General Feedback", "Bug Report", "Feature Request", "User Experience", "Other"]
)

# --- Rating Section ---
            st.write("Please rate your overall experience in using our Web App:")
            rating = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 3, 5)
            stars = "⭐" * rating
            st.write(f"Your Rating: {stars}")
# --- Message ---
            user_message = st.text_area(
    "Have questions or suggestions? I'd love to hear from you.*",
    height=100,
    placeholder="Type your detailed feedback here..."
)

# --- Submit Button ---
            if st.button("Send Feedback"):
    # Form submission handling
                if not all([feedback_name, feedback_email, user_message]):
                    st.warning("⚠️ Please fill in all required fields (marked with *)")
                else:
        # Prepare sending payload
                    formspree_endpoint = "https://formspree.io/f/xyzkwvel"
                    payload = {
            "subject": f"{feedback_category} : {stars}",
            "name": feedback_name,
            "email": feedback_email,
            "category": feedback_category,
            "rating": f"{rating}/5 ({stars})",
            "message": user_message,
            "_replyto": feedback_email
        }
                    try:
                        response = requests.post(
                            formspree_endpoint,
                            data=payload,
                            headers={"Accept": "application/json"}
            )
                        if response.status_code == 200:
                            st.success(f"✅ Thank you for your feedback! Your Rating: {stars}")
                            st.balloons()
                        else:
                            st.error(f"❌ Oops! Something went wrong. Status code: {response.status_code}")
        
                    except Exception as e:
                        st.error(f"❌ Failed to submit: {str(e)}")
                        st.info("ℹ️ You can also email us directly at uzerkanza05@gmail.com")




# Run the main function
if __name__ == "__main__":
    # Install required packages
    try:
        import reportlab
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "reportlab", "uuid"])
    main()
