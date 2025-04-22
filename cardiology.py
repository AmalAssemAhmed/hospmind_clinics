from pandas import notnull


def cardiology_content(ecg_model,heartattack_model,heartattack_scaler):
    import streamlit as st
    import sqlite3
    import re
    from reportlab.lib.utils import ImageReader
    import pandas as pd
    import plotly.express as px
    from fpdf import FPDF
    import pdfkit
    import shap
    import joblib
    import datetime
    import plotly.graph_objects as go
    import numpy as np
    from reportlab.lib.utils import simpleSplit
    import markdown2
    from PIL import Image
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import os
    import cv2
    import tensorflow as tf
    from tensorflow.keras.utils import img_to_array
    import matplotlib.pyplot as plt
    import gdown
    
    ecg_report_markdown = None
    heartattack_report_markdown = None
    heartattack_report_file = None
    ecg_report_file = None
    report_type =None
    patient_name =None
    result =None
    image =None
    heartattack_path=""
    ecg_path =""
   
    table = "cardiology_patients" 

   
    # Connect to the database
    conn = sqlite3.connect("healthcare.db", check_same_thread=False)
    cursor = conn.cursor()

    # Get feature names from the training data
    feature_names = ['Age', 'Heart rate', 'Systolic blood pressure',
                     'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

    # Function to predict heart attack risk using the trained model
    def predict_heart_attack_risk(data):
        data_reshaped = np.array(data).reshape(1, -1)
        scaled_data = heartattack_scaler.transform(data_reshaped)
        probability = heartattack_model.predict_proba(scaled_data)[0][1]
        explainer = shap.TreeExplainer(heartattack_model)
        risk_level = "High Risk" if probability >= 0.7 else "Medium Risk" if probability >= 0.3 else "Low Risk"
        shap_values = explainer.shap_values(scaled_data)
        return probability, shap_values

    def heartattack_report(first_name, last_name, national_id, mobile, gender, patient_data, prediction_prob):
        report_text = f"###  Heart Attack Medical Report\n\n"

        # Patient Information
        report_text += f"####  Patient Information\n"
        report_text += f"- *Name:* {first_name} {last_name if last_name else ''}\n"
        report_text += f"- *National ID:* {national_id}\n"
        report_text += f"- *Mobile:* {mobile if mobile else 'Not Provided'}\n"
        report_text += f"- *Gender:* {gender if gender else 'Not Specified'}\n\n"

        # Clinical Evaluation
        report_text += f"### Clinical Evaluation\n\n"

        age = patient_data.get("Age", None)
        systolic = patient_data.get("Systolic blood pressure", None)
        diastolic = patient_data.get("Diastolic blood pressure", None)
        heart_rate = patient_data.get("Heart rate", None)
        blood_sugar = patient_data.get("Blood sugar", None)
        ck_mb = patient_data.get("CK-MB", None)
        troponin = patient_data.get("Troponin", None)

        # Age Analysis
        if age is not None:
            report_text += "####  Age and Cardiovascular Risk\n"
            if age < 40:
                report_text += "- The patient is young, indicating a lower cardiovascular risk.\n"
            elif 40 <= age <= 55:
                report_text += "- The patient is middle-aged, requiring regular monitoring of vital signs.\n"
            else:
                report_text += "- Advanced age is a significant cardiovascular risk factor, requiring routine check-ups.\n\n"

        # Blood Pressure Analysis (Systolic & Diastolic Combined)
        if systolic is not None and diastolic is not None:
            report_text += "####  Blood Pressure Analysis\n"
            report_text += f"- *Recorded BP:* {systolic}/{diastolic} mmHg\n"
            if systolic < 120 and diastolic < 80:
                report_text += "- 🟢 Blood pressure is within the normal range.\n"
            elif 120 <= systolic <= 139 or 80 <= diastolic <= 89:
                report_text += "- 🟡 Slightly elevated blood pressure, requires regular monitoring.\n"
            else:
                report_text += "-  High blood pressure detected, medical intervention is recommended.\n\n"

        # Heart Rate Analysis
        if heart_rate is not None:
            report_text += "####  Heart Rate\n"
            report_text += f"- *Recorded Value:* {heart_rate} bpm\n"
            if 60 <= heart_rate <= 100:
                report_text += "- 🟢 Heart rate is within the normal range.\n"
            elif heart_rate > 100:
                report_text += "- 🟡 Elevated heart rate, possibly due to stress or other factors.\n"
            else:
                report_text += "-  Low heart rate detected, further evaluation may be required.\n\n"

        # Blood Sugar Analysis
        if blood_sugar is not None:
            report_text += "####  Blood Sugar Levels\n"
            report_text += f"- *Recorded Value:* {blood_sugar} mg/dL\n"
            if 70 <= blood_sugar <= 99:
                report_text += "- 🟢 Normal blood sugar levels.\n"
            elif 100 <= blood_sugar <= 125:
                report_text += "- 🟡 Slightly high blood sugar, lifestyle changes recommended.\n"
            else:
                report_text += "-  High blood glucose detected, potential diabetes risk requiring further assessment.\n\n"

        # CK-MB & Troponin Analysis
        if ck_mb is not None and troponin is not None:
            report_text += "####  Cardiac Enzymes (CK-MB & Troponin)\n"
            report_text += f"- *CK-MB:* {ck_mb}\n"
            report_text += f"- *Troponin:* {troponin}\n"
            if ck_mb < 5 and troponin < 0.04:
                report_text += "- 🟢 Normal levels, no signs of myocardial damage.\n"
            elif 5 <= ck_mb < 10 or 0.04 <= troponin < 0.1:
                report_text += "- 🟡 Slightly elevated levels, further monitoring required.\n"
            else:
                report_text += "-  Elevated cardiac markers detected, urgent medical evaluation is necessary.\n\n"
        #  Symptoms
        report_text += "### Symptoms\n\n"
        if symptoms:
            report_text += ",".join(symptoms)
            report_text += "\n\n"
        else:
            report_text += "No significant symptoms\n\n"

        # Final Assessment & Recommendations
        report_text += "###  Final Assessment & Medical Recommendations\n"
        if prediction_prob < 40:
            report_text += "✅ No major risk factors detected. Routine check-ups and a healthy lifestyle are advised.\n"
        elif 40 <= prediction_prob < 70:
            report_text += "⚠ Some risk factors are present. Regular monitoring and medical consultation are recommended.\n"
        else:
            report_text += " High cardiovascular risk detected. Immediate medical attention is advised.\n\n"

        # Report Date
        report_text += f" *Report Date:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Disclaimer
        report_text += "---\n"
        report_text += "### 🔍 Additional Notes\n"
        report_text += "This report provides an *initial assessment* based on the patient's input data and does not constitute a final diagnosis. A specialist consultation is recommended for a thorough medical evaluation.\n"

        return report_text

    def ecg_report(first_name, last_name,national_id, mobile, gender, prediction):
        diagnosis_dict = {
            "N": "Indicates a normal heart rhythm. The electrical activity of the heart is functioning within the expected parameters.",
            "F": "Normal heart rhythm, though it may include minor variations not considered pathological.",
            "Q": "Abnormal electrical activity, possibly arrhythmias like premature beats or heart block.",
            "S": "Irregular rhythms, potentially flutter or blocks needing examination.",
            "V": "Ventricular arrhythmias like tachycardia, requiring urgent attention.",
            "M": "Severe arrhythmias like ventricular fibrillation. Immediate medical action required."
        }
        report_text = "### Comprehensive Medical Report\n\n"

        # Patient Information
        report_text += f"####  Patient Information\n"
        report_text += f"- *Name:* {first_name} {last_name if last_name else ''}\n"
        report_text += f"- *National ID:* {national_id}\n"
        report_text += f"- *Mobile:* {mobile if mobile else 'Not Provided'}\n"
        report_text += f"- *Gender:* {gender if gender else 'Not Specified'}\n\n"
        report_text += "---\n"
        report_text += "### 🔍 Classification\n"
        report_text += f"The ECG image was classified into the *{prediction}* category."

        report_text += "---\n"
        report_text += "### Diagnosis :\n"

        report_text += f"*{prediction}* → {diagnosis_dict[prediction]}"

        report_text += "---\n"

        report_text += "### Recommendation:\n"
        report_text += "Further clinical investigation may be needed based on patient history and additional diagnostics."

         # Disclaimer
        report_text += "---\n"
        report_text += "### 🔍 Additional Notes\n"
        report_text += "This report provides an *initial assessment* based on the patient's input data and does not constitute a final diagnosis. A specialist consultation is recommended for a thorough medical evaluation.\n"

        return report_text

    def convert_markdown_to_pdf(report_markdown, national_id, report_type):
        os.makedirs("reports", exist_ok=True)  # Ensure the reports directory exists
        output_path = os.path.join("reports", f"{report_type}_report_{national_id}.pdf")

        # Create a new PDF file with A4 size
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        y_position = 750

        # Regex patterns for markdown formatting
        image_pattern = re.compile(r"!(.*?)(.*?)")
        bold_pattern = re.compile(r"\*\*(.*?)\*\*")
        italic_pattern = re.compile(r"\*(.*?)\*")
        bold_italic_pattern = re.compile(r"\*\*\*(.*?)\*\*\*")

        for line in report_markdown.split("\n"):
            image_match = image_pattern.search(line)
            if image_match:
                image_path = image_match.group(2)
                try:
                    img = ImageReader(image_path)
                    c.drawImage(img, 100, y_position - 100, width=100, height=100)
                    y_position -= 120
                except Exception as e:
                    print(f"⚠ Error loading image: {e}")

            elif line.startswith("# "):
                c.setFont("Helvetica-Bold", 16)
                c.drawString(100, y_position, line.replace("# ", ""))
            elif line.startswith("## "):
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, y_position, line.replace("## ", ""))
            elif line.startswith("### "):
                c.setFont("Helvetica-Bold", 12)
                c.drawString(100, y_position, line.replace("### ", ""))
            elif line.startswith("#### "):
                c.setFont("Helvetica-Bold", 10)
                c.drawString(100, y_position, line.replace("#### ", ""))
            else:
                # Apply markdown styles
                line = bold_italic_pattern.sub(r"\1", line)  # Apply bold italic
                line = bold_pattern.sub(r"\1", line)  # Apply bold
                line = italic_pattern.sub(r"\1", line)  # Apply italic

                c.setFont("Helvetica", 12)
                c.drawString(100, y_position, line)

            y_position -= 20

        c.save()
        return output_path

    def plot_patient_waterfall(feature_names, shap_values, total_risk):
        """
        Generates a Waterfall Chart to explain heart attack risk for a single patient.

        Parameters:
        - feature_names (list): List of medical factors.
        - shap_values (list): Corresponding SHAP values for each factor.
        - total_risk (float): Predicted heart attack risk (in percentage).

        Returns:
        - Displays an easy-to-understand waterfall chart.
        """

        # Sort factors by absolute impact (most influential first)
        sorted_indices = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=True)
        feature_names = [feature_names[i] for i in sorted_indices]
        shap_values = [shap_values[i] for i in sorted_indices]

        # Define colors (🔴 red = increases risk, 🟢 green = decreases risk)
        colors = ["red" if val > 0 else "green" for val in shap_values]

        # Create the Waterfall Chart
        fig = go.Figure(go.Waterfall(
            name="Risk Contribution",
            orientation="v",
            measure=["relative"] * len(feature_names),
            x=feature_names,
            y=shap_values,
            text=[f"{val:.2f}%" for val in shap_values],  # Show percentage impact
            textposition="outside",
            connector={"line": {"color": "gray"}},
            decreasing={"marker": {"color": "green"}},  # Green for risk-reducing factors
            increasing={"marker": {"color": "red"}},  # Red for risk-increasing factors
        ))
        # Add total risk as a separate annotation
        fig.add_annotation(
            text=f"🔹 *Total Risk: {total_risk:.1f}%*",
            x=0.5, y=1.1,  # Adjust Y to place it right under the title
            showarrow=False,
            font=dict(size=16, color="black"),
            xref="paper", yref="paper",  # Relative positioning
            align="center"
        )

        # Update layout for better readability
        fig.update_layout(
            title="📊 Heart Attack Risk Breakdown (Single Patient)",
            yaxis_title="Impact on Risk (%)",
            xaxis_title="Medical Factors",
            showlegend=False
        )
        return fig

    # Function to predict ECG classification
    def predict_ecg(image):
        # Define class labels and their corresponding indices
        classes = {'F': 0, 'M': 1, 'N': 2, 'Q': 3, 'S': 4, 'V': 5}
        inverse_classes = {v: k for k, v in classes.items()}

        # Dictionary mapping each class to its diagnosis description

        # Load and preprocess the ECG image

        img_array = img_to_array(image) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # Make prediction
        predictions = ecg_model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = inverse_classes[predicted_class_index]
        # Get actual class (e.g., from folder name)
        # actual_class = os.path.basename(os.path.dirname(image_path))

        return predicted_class,predictions[0]

   
    # Custom CSS for Dark Mode and Text Color
    st.markdown("""
     <style>
     body, .stApp {background-image :url("/content/edited_hostimind.jpg");
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #161a24;
    }

    /* Tabs and widgets */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #ffffff;
        border: none !important;
        box-shadow: none !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #ffffff;
    }

    /* Input and select boxes */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stDateInput input,
    .stSelectbox > div > div,
    .stButton button {
        background-color: #21262d;
        color: #ffffff;
        border: 1px solid #30363d;
    }

    /* Buttons */
    .stButton button {
        background-color: #238636;
        color: white;
        
        border-radius: 30%;
    }
     .stButton button:hover {
        background-color: #2ea043;
        color: white;
    }
   
    /* Markdown links */
    a {
        color: #58a6ff;
    }

    .stSuccess {
        background-color: #1d2b1f;
        color: #a3f7bf;
    }
    h1 {
        font-size: 36px;
        margin-top: 50px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .labels {
        font-size: 20px;
        margin-top: 30px;
    }
    
    </style>
    """, unsafe_allow_html=True)
    department = "Cardiology"
    logo,header =st.columns([1,9]) 
    with logo:
      st.image("hos.png",use_container_width=False)     
    with header :
      st.markdown('<h1 >Cardiology Department</h1>', unsafe_allow_html=True)
    # ---- Patient Information ----
    st.markdown(
        '<h2 >🆔 Personal Information</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:

        first_name = st.text_input("First Name (required)", key="first_name")
        last_name = st.text_input("Last Name (required)", key="last_name")
        national_id = st.text_input("National ID (required)", key="national_id")
    with col2:
        mobile = st.text_input("Mobile Number (optional)", key='mobile')
        gender = st.selectbox("Gender (optional)", ["Male", "Female"], key='gender')
        age = st.number_input("Age (optional)", min_value=0, step=1, key='age')
    patient_name = first_name + " " + last_name
    save_patient = st.button("Save Patient", key="save_patient")
    if save_patient:
        if national_id and patient_name :
            cursor.execute("""
            INSERT OR REPLACE INTO cardiology_patients 
            (national_id, name, mobile, gender, age,department, report_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (national_id, patient_name, mobile, gender, age, "cardiology",
              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            st.success(f"Saved {patient_name} information Successfully!")
        else:
            st.warning("Please complete patient personal information")
    st.markdown(
        '<h2 >🏥 Medical Information</h2>', unsafe_allow_html=True)        
   
    col3, col4 = st.columns(2)
    with col3:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, step=1)
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, step=1)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=150, step=1)
    with col4:
        blood_sugar = st.number_input("Blood Sugar Level (mg/dL)", min_value=50, max_value=400, step=1)
        ck_mb = st.number_input("CK-MB Level (ng/mL)", min_value=0.0, max_value=10.0, step=0.1)
        troponin = st.number_input("Troponin Level (ng/mL)", min_value=0.0, max_value=5.0, step=0.01)

    # Symptoms selection
    symptoms = st.multiselect("Select Symptoms",
                              ["Chest Pain", "Shortness of Breath", "Dizziness", "Fatigue", "Palpitations"])
    patient_data = [age, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin] 
                             
     
    if patient_data:
         probability, shap_values = predict_heart_attack_risk(patient_data)

         # Convert patient_data and shap_values to dictionaries
         patient_data_dict = dict(zip(feature_names, patient_data))  # Convert patient data to a dictionary
         shap_values_dict = dict(zip(feature_names, shap_values[0]))  # Convert SHAP values to a dictionary

         heartattack_report_markdown = heartattack_report(first_name, last_name,national_id, mobile,gender,
                                                         patient_data_dict, probability * 100)

                                                        
    col7,col8 =st.columns(2)
    col5, coll6 = st.columns(2)                                                     
    with col7 :                        
      heartattack_prediction = st.button("Heart attack Prediction", key="predict")
      if heartattack_prediction:
        if patient_data:
         
            
            # 📌 display report
            heartattack_report_markdown =f"<div style = 'color : white;'>{heartattack_report_markdown}</div>"
            st.markdown(heartattack_report_markdown, unsafe_allow_html=True)
            # Visualization
            st.markdown("## 📊 Risk Analysis")
            
            with col5:
              fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Heart Attack Risk"},
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "black"},
                       'steps': [
                           {'range': [0, 40], 'color': "green"},
                           {'range': [40, 70], 'color': "orange"},
                           {'range': [70, 100], 'color': "red"}]}
                       
                        ))
              st.plotly_chart(fig_gauge)
            with coll6:
              fig = plot_patient_waterfall(feature_names, shap_values[0], probability * 100)
              st.plotly_chart(fig)
        else:
          st.warning("Please,Enter all medical Data")      
      with col8:
         # Button to save ECG report
          save_report_heartattack = st.button("Save Heart Attack Report", key="save_report_heartattack")
          if save_report_heartattack:
              if national_id and patient_name and heartattack_report_markdown :
                    #report_type = "heart_attack"
                    
                    heartattack_report_file = convert_markdown_to_pdf(heartattack_report_markdown, national_id, "heart_attack")
                    if ecg_report_markdown:
                        ecg_report_file = convert_markdown_to_pdf(ecg_report_markdown, national_id, "ecg")
       
                    # insert patient information and Heart Attack REport in Cardiology table
                    cursor.execute("""
                           INSERT OR REPLACE INTO cardiology_patients 
                           (national_id, name, mobile, gender, age,department,heartattack_report_pdf,ecg_report_pdf, report_date) 
                            VALUES (?, ?, ?, ?, ?, ?, ?,?,?)
                            """, (national_id, patient_name, mobile, gender, age, "cardiology", heartattack_report_file,ecg_report_file,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    st.success(f"Heart Attack Report saved successfuly at :{heartattack_report_file}")
           
              else:
                st.warning("please complete required data")

    # ECGG image upload
    st.markdown(
        '<h2 >ECG image uploader</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", 'png', "jpeg"])
    col9,col10 = st.columns(2)

    if uploaded_file is not None:
       image = Image.open(uploaded_file)
       image =image.convert('RGB')
       image = image.resize((224,224))
       prediction,prob= predict_ecg(image)
       ecg_report_markdown = ecg_report(first_name, last_name, national_id, mobile, gender, prediction)
       
       col11,col12 =st.columns(2) 

    with col9 :
         
      ecg_prediction = st.button("ECG Prediction", key="ecg_predict")
      if ecg_prediction:
          if uploaded_file is not None:
            with col11:
              st.image(image, caption="Uploaded Image", width=450) 
            with col12:
                # Display bar chart
                #st.subheader("Class Probabilities")
                class_labels = ['F', 'M', 'N', 'Q', 'S', 'V']
                fig, ax = plt.subplots()
                bars = ax.bar(class_labels, prob, color='lightblue')
                ax.set_ylim([0, 1])
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Probabilities")
                st.pyplot(fig)
            ecg_report_markdown =f"<div style = 'color : white;'>{ecg_report_markdown}</div>"
            st.markdown(ecg_report_markdown, unsafe_allow_html=True) 
          else: 
              st.warning("Pleaswe upload an ECG Image")
    with col10 :        
          # Button to save ECG report
          save_report = st.button("Save ECG Report", key="save_report")
          if save_report:
              if national_id and patient_name and ecg_report_markdown :
                if uploaded_file is not None:
                    report_type = "ecg"
                    ecg_report_file = convert_markdown_to_pdf(ecg_report_markdown, national_id, report_type)
                    if ecg_report_markdown:
                        ecg_report_file = convert_markdown_to_pdf(ecg_report_markdown, national_id, "heart_attack")
       

                         # insert patient information and ECG REport in Cardiology table
                        cursor.execute("""
                           INSERT OR REPLACE INTO cardiology_patients 
                           (national_id, name, mobile, gender, age,department,heartattack_report_pdf,ecg_report_pdf, report_date) 
                            VALUES (?, ?, ?, ?, ?, ?, ?,?,?)
                            """, (national_id, patient_name, mobile, gender, age, "cardiology",heartattack_report_file, ecg_report_file,
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
              
                        conn.commit()
                        st.success(f"ECG Report saved successfuly at :{ecg_report_file}")
                else: 
                    st.warning("Pleaswe upload an ECG Image") 
           
              else:
                st.warning("please complete required data")
    st.markdown(
        '<h2 >Search for a Patien</h2>', unsafe_allow_html=True)            
    
    search_id = st.text_input("Enter Patient ID to Retrieve Data",key ="search_id")
    col13,col14,col15 = st.columns(3)
    with col13:
     ecg_search = st.button("ECG Report", key="ecg_search")
  
     
     if ecg_search :
      if search_id:
        '''
        file_path = f"reports/ecg_report_{search_id}.pdf"
        if  os.path.exists(file_path) :
            with open(file_path, "rb") as file:
                st.download_button(label=" Download Report", data=file,file_name=f"ecg_report_{search_id}.pdf", mime="application/pdf")
        else:
          st.warning("No ECG Reportfound for this PatientID")
       
        '''
       
        cursor.execute(f"SELECT ecg_report_pdf FROM {table} WHERE national_id=?", (search_id,))
        result = cursor.fetchone()
        #st.write("Debug: result =", result)
        
        if result is not None  and result[0] is not None:
            with open(result[0], "rb") as file:
                st.download_button(label="📄 Download Report", data=file,file_name=f"ecg_report_{search_id}.pdf", mime="application/pdf")
        else :
          st.warning("No ECG Report found for this PatientID")
      else:
          st.warning("Please enter PatientID")
      


    with col14:  
     heartattack_search = st.button("Heart Attack Report", key="heartattack_search")
     if heartattack_search :
      if search_id: 
        cursor.execute(f"SELECT heartattack_report_pdf FROM {table} WHERE national_id=?", (search_id,))
        result = cursor.fetchone()
        #st.write("Debug: result =", result)
        if result is not None and result[0] is not None:
            with open(result[0], "rb") as file:
                st.download_button(label="📄 Download Report", data=file,file_name=f"heart_attack_report_{search_id}.pdf", mime="application/pdf")
          
        else:
          st.warning("No Heart Attack Reportfound for this PatientID")
      else:
          st.warning("Please enter PatientID")


       
      # Button for deleting patient
   with col15:
    delete_patient = st.button("🗑 Delete Patient", key="delete_patient")  
    if delete_patient:
        if search_id:    
            cursor.execute(f"SELECT * FROM {table} WHERE national_id=?", (search_id,))
            result = cursor.fetchone()

            if result is not None:
                heartattack_path = result[6]
                ecg_path = result[7]

                # Check and delete heartattack report
                if heartattack_path and os.path.exists(heartattack_path):
                    st.write("Deleting heart attack report at path:", heartattack_path)
                    os.remove(heartattack_path)

                # Check and delete ecg report
                if ecg_path and os.path.exists(ecg_path):
                    st.write("Deleting ECG report at path:", ecg_path)
                    os.remove(ecg_path)

                # Delete patient from database
                cursor.execute(f"DELETE FROM {table} WHERE national_id=?", (search_id,))
                conn.commit()
                st.success("✅ Patient record deleted successfully!")
            else:
                st.error("❌ No patient found!")
        else:
            st.warning("Please enter PatientID")

    
# Back to Home 
    col16,col17,col18 = st.columns(3)
    with col18: 
      if st.button(" 🏠 Go to home"):
        st.session_state.page = "main"
        st.rerun()
   
    conn.close()
