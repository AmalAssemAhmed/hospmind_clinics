
def neurology_content(tumor_model): 

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
  import pandas as pd 

  # Connect to the database
  conn = sqlite3.connect("healthcare.db", check_same_thread=False)
  cursor = conn.cursor()
  
   #define variables
  class_names =["glioma" ,"meningioma","notumor","pituitary"]      
  table = "neurology_patients" 
  image_path =""
  result = None

  def preprocess_and_predict(image_path):
   # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        st.warning("Error: couldn't read image:", image_path)
        return None,None,None

    img =cv2.resize(img,(240,240))
    img_normalized = img / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Model prediction
    prediction = tumor_model.predict(img_input)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100
    predicted_label = class_names[predicted_index]
    return predicted_label,img,prediction[0]

  
  def MRI_report(first_name, last_name, national_id, mobile, gender, prediction):
        diagnosis_dict = {
            "glioma": "-Glioma- Tumor-The MRI scan demonstrates a mass lesion consistent with a glioma tumor. The lesion appears infiltrative with irregular borders and heterogeneous signal intensity. The tumor may involve both gray and white matter and shows mild mass effect. Edema is also noted surrounding the lesion.",
            "meningioma": " -Meningioma Tumor-MRI reveals a well-defined, extra-axial mass with broad dural attachment. The lesion demonstrates homogeneous enhancement post-contrast and shows a classic “dural tail” sign, which is characteristic of meningioma. There is mild mass effect, but no evidence of parenchymal invasion.",
            "notumor": " -No Tumor- MRI scan appears within normal limits. No evidence of space-occupying lesions, abnormal signal intensity, or midline shift. Ventricular system and sulci are preserved, and no signs of edema or hemorrhage are identified. .",
            "pituitary": " -Pituitary Tumor- There is an enhancing lesion within the sella turcica, involving the pituitary gland. The mass mildly expands the sella and abuts the optic chiasm superiorly. The imaging features are consistent with a pituitary adenoma. No evidence of hemorrhage or invasion is noted."
        }
        report_text = "## Comprehensive AI Medical Report\n\n"

        # Patient Information
        report_text += f"####  Patient Information\n"
        report_text += f"- *Name:* {first_name} {last_name if last_name else ''}\n"
        report_text += f"- *National ID:* national_{national_id}\n"
        report_text += f"- *Mobile:* {mobile if mobile else 'Not Provided'}\n"
        report_text += f"- *Gender:* {gender if gender else 'Not Specified'}\n\n"
        report_text += "---\n"
          #  Symptoms
        report_text += "### Symptoms\n\n"
        if symptoms:
            report_text += ",".join(symptoms)
            report_text += "\n\n"
        else:
            report_text += "No significant symptoms\n\n"
        report_text += "### 🔍 Classification\n"
        report_text += f"The Chest X ray was classified into the *{prediction}* category."
        

        report_text += "---\n"
        report_text += "### Diagnosis :\n"

        report_text += f"*{prediction}* → {diagnosis_dict[prediction]}"

         # Disclaimer
        report_text += "---\n"
        report_text += "### 🔍 Additional Notes\n"
        report_text += "This report provides an *initial assessment* based on the patient's input data and does not constitute a final diagnosis. A specialist consultation is recommended for a thorough medical evaluation.\n"

    
        return report_text 
  
  def convert_markdown_to_pdf(report_markdown, national_id):
        os.makedirs("reports", exist_ok=True)  # Ensure the reports directory exists
        output_path = os.path.join("reports", f"brain_tumor_report_{national_id}.pdf")

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
   


    # Custom CSS for Dark Mode and Text Color
  st.markdown("""
        <style>
             body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
-
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
70.    .stSuccess {
        background-color: #1d2b1f;
        color: #a3f7bf;
    }
    
      
        </style>
        """, unsafe_allow_html=True)
  col1,col2 =st.columns([1,9]) 
  with col1:
    st.image("hos.png",use_container_width=False)     
  with col2 :
    st.title("Neurology Diagnosis")
  col3, col4 = st.columns(2)

  with col3:

        first_name = st.text_input("First Name (required)", key="first_name")
        last_name = st.text_input("Last Name (required)", key="last_name")
        national_id = st.text_input("National ID (required)", key="national_id")
  with col4:
        mobile = st.text_input("Mobile Number (optional)", key='mobile')
        gender = st.selectbox("Gender (optional)", ["Male", "Female"], key='gender')
        age = st.number_input("Age (optional)", min_value=0, step=1, key='age')
  patient_name = first_name + " " + last_name
  save_patient = st.button("Save Patient", key="save_patient")
  if save_patient:
        if national_id and patient_name :
            cursor.execute("""
            INSERT OR REPLACE INTO neurology_patients 
            (national_id, name, mobile, gender, age,department, report_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (national_id, patient_name, mobile, gender, age, "neurology",
              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            st.success(f"Saved {patient_name} information Successfully!")
        else:
            st.warning("Please complete patient personal information")
  st.markdown(
        '<h2 >🏥 Medical Information</h2>', unsafe_allow_html=True)
       
  # Symptoms selection
  symptoms = st.multiselect("Select Symptoms",
                                ["Seizures",
                                 "Loss of consciousness",
                                "Paralysis (one side or limbs)",
                                "Muscle weakness",
                                "Speech difficulty",
                                "Memory problems",
                                "Vision problems (double/blurry vision)",
                                "Loss of balance",
                                "Confusion or disorientation",
                                "Tremors"
                                   ])        
  st.title("Upload MRI for diagnosis.")
  uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
  col5,col6 = st.columns(2)
  col7,col8 =st.columns(2)

  if uploaded_file is not None: 
        image_path = f"temp_{uploaded_file.name}"

        with open(image_path, "wb") as f:


           f.write(uploaded_file.getbuffer())
           predicted_label,image,prob = preprocess_and_predict(image_path)
          
           tumor_report_markdown = MRI_report(first_name, last_name, national_id, mobile, gender, predicted_label)


        with col5 :
         
              tumor_prediction = st.button("MRI Prediction", key="tumor_predict")
              if tumor_prediction:

               

                with col7:
                   st.image(image, caption="Uploaded Image", use_container_width = True) 
           
                with col8:
                   # Display bar chart
                   #st.subheader("Class Probabilities")
                   class_labels = ["glioma" ,"meningioma","notumor","pituitary"]
                   fig, ax = plt.subplots()
                   bars = ax.bar(class_labels, prob, color='lightblue')
                   ax.set_ylim([0, 1])
                   ax.set_ylabel("Probability")
                   ax.set_title("Prediction Probabilities")
                   st.pyplot(fig)

                tumor_report_markdown =f"<div style = 'color : white;'>{tumor_report_markdown}</div>"
                st.markdown(tumor_report_markdown, unsafe_allow_html=True)
           
        with col6 :        
             # Button to save Chest X ray report
             save_report = st.button("Save AI Report", key="save_report")
             if save_report:
               if national_id and patient_name and tumor_report_markdown and image_path :
                   
                    tumor_report_file = convert_markdown_to_pdf(tumor_report_markdown, national_id)
          
                    # insert patient information and ECG REport in Cardiology table
                    cursor.execute("""
                           INSERT OR REPLACE INTO neurology_patients 
                           (national_id, name, mobile, gender, age,department,MRI_image,report_pdf, report_date) 
                            VALUES (?, ?, ?, ?, ?, ?, ?,?,?)
                            """, (national_id, patient_name, mobile, gender, age, "Neurology",image_path,tumor_report_file, 
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
              
                    conn.commit()
                    st.success(f"MRI Report saved successfuly at :{tumor_report_file}")
           
               else:
                  st.warning("please complete required data")

  st.markdown('<h2 >🔍 Search for a Patien</h2>', unsafe_allow_html=True)            
    
  search_id = st.text_input("Enter Patient ID to Retrieve Data",key ="search_id")

  MRI_search = st.button("Download Chest Report", key="MRI_search")

  if chest_search:
    if search_id:
        try:
            cursor.execute(f"SELECT report_pdf FROM {table} WHERE national_id = ?", (search_id,))
            result = cursor.fetchone()
            
            st.write("Debug: result =", result)
            
            if result and result[0] and os.path.exists(result[0]):
                with open(result[0], "rb") as file:
                    st.download_button(
                        label="📄 Download Report",
                        data=file,
                        file_name=f"chest_xray_report_{search_id}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No report file found for this patient ID.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid Patient ID.")

        # Button for deleting patient
  elif st.button("🗑 Delete Patient"):
    if search_id: 
     
      cursor.execute(f"SELECT report_pdf FROM {table} WHERE national_id=?", (search_id,))
      result = cursor.fetchone()

      if result[0]:
          tumor_path = result[0]
          
          if os.path.exists(tumor_path):
                os.remove(tumor_path)
         
                # Delete patient from database
                cursor.execute(f"DELETE FROM {table} WHERE national_id=?", (search_id,))
                conn.commit()
                st.success("✅ Patient record deleted successfully!")
      else:
         st.error("❌ No patient found!")
    else:
            st.warning("Please enter PatientID")     
           

# Add buttons to return to Clinics or Main
  st.markdown("---")
  col9, col10 ,col11= st.columns(3)
  with col11:
    if st.button(" 🏠 Go to home"):
        st.session_state.page = "main"
        st.rerun()

