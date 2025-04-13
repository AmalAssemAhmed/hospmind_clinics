import streamlit as st
import sqlite3
import pandas as pd
from hepatology import hapatology_content
from cardiology import cardiology_content
#from pulmonology import pulmonology_content
from neurology import neurology_content





# Connect to the database
conn = sqlite3.connect("healthcare.db")
cursor = conn.cursor()
#cursor.execute("DROP TABLE IF EXISTS pulmonology_patients")
#cursor.execute("DROP TABLE IF EXISTS neurology_patients")

# Ensure tables exist before executing queries
cursor.execute("""
    CREATE TABLE IF NOT EXISTS cardiology_patients (
        national_id TEXT PRIMARY KEY,
        name TEXT,
        mobile TEXT,
        gender TEXT,
        age INTEGER,
        department TEXT,
        heartattack_report_pdf TEXT,
        ecg_report_pdf TEXT,
        report_date TEXT,

    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS neurology_patients (
        national_id TEXT PRIMARY KEY,
        name TEXT,
        mobile TEXT,
        gender TEXT,
        age INTEGER,
        department TEXT,
        MRI_image TEXT,
        report_pdf TEXT,
        report_date TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS pulmonology_patients (
        national_id TEXT PRIMARY KEY,
        name TEXT,
        mobile TEXT,
        gender TEXT,
        age INTEGER,
        department TEXT,
        xray_image TEXT,
        report_pdf TEXT,
        report_date TEXT
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS hepatology_patients (
        national_id TEXT PRIMARY KEY,
        name TEXT,
        mobile TEXT,
        gender TEXT,
        age INTEGER,
        department TEXT,

        report_pdf TEXT,
        report_date TEXT
    )
""")







conn.commit()
# Fetch total patients and pending appointments
cursor.execute("SELECT COUNT(*) FROM cardiology_patients")
total_cardiology_patients = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM pulmonology_patients")
total_pulmonology_patients = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM neurology_patients")
total_neurology_patients = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM hepatology_patients")
total_hepatology_patients = cursor.fetchone()[0]

total_patients =total_hepatology_patients+ total_cardiology_patients + total_pulmonology_patients+total_neurology_patients


# Fetch  No completed tests (Assumed to be total patients with reports)
cursor.execute("SELECT COUNT(*) FROM cardiology_patients WHERE heartattack_report_pdf is Null and ecg_report_pdf IS  Null")
no_completed_cardiology_tests = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM pulmonology_patients WHERE report_pdf IS  NULL")
no_completed_pulmonology_tests = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM neurology_patients WHERE report_pdf IS  NULL")
no_completed_neurology_tests = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM hepatology_patients WHERE report_pdf IS  NULL")
no_completed_hepatology_tests = cursor.fetchone()[0]


no_completed_tests = no_completed_hepatology_tests +no_completed_cardiology_tests + no_completed_pulmonology_tests+no_completed_neurology_tests

# Function to get patients with no reports in a selected department
def get_patients_without_reports(department):
    # Query to get patients without reports in the selected department
    if department == "cardiology":
      query = f"""
        SELECT name, national_id, department
        FROM {department}_patients
        WHERE heartattack_report_pdf IS NULL and ecg_report_pdf IS NULL
          """
    else:
      query = f"""
      SELECT name, national_id, department
      FROM {department}_patients
      WHERE report_pdf IS NULL
      """
    cursor.execute(query)
    patients = cursor.fetchall()

    # Create a dataframe for the patients
    df = pd.DataFrame(patients, columns=["Name", "National ID", "Department"])
    return df
conn.close()
#st.set_page_config(page_title="Healthcare System", layout="wide")
st.set_page_config(page_title="Hospmind Clinics", page_icon="🏥", layout="wide")
# Sidebar contents
with st.sidebar:
    st.image("/content/drive/MyDrive/hospmind.jpg", width=150)
    st.title("HOSPMIND")

    st.markdown("### Departments")
    if st.button("Home"):
        st.session_state.page = "main"
        st.rerun()
    if st.button("Cardiology "):
        st.session_state.page = "cardiology"
        st.rerun()
    if st.button("Neurology "):
        st.session_state.page = "neurology"
        st.rerun()
    if st.button("Pulmonology"):
        st.session_state.page = "pulmonology"
        st.rerun()
    if st.button("Hepatology "):
        st.session_state.page = "hepatology"
        st.rerun()

    st.markdown("---")
    st.markdown(f"*Total Patients:* {total_patients}")
    st.markdown(f"*Patients w/o Reports:* {no_completed_tests}")



# Set up page layout
if "page" not in st.session_state:
  st.session_state.page = "main"
if st.session_state.page == "main":


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
      img {
        border-radius: 30% !important;
        width: 100px !important;
        height: 100px !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

  col1,col2 =st.columns([1,9])
  with col1:
    st.image("hospmind.jpg",use_container_width=False)
  with col2 :
    # Main title of the page
    st.markdown('<h1>Welcome in HOSPMIND Clinics</h1>', unsafe_allow_html=True)
  st.write("")
  st.write("")

  col3, col4 ,col9,col10= st.columns([0.2,0.4,0.6,1])
  col4.markdown(f"""
<div style="text-align:center;
 background-color: #dcdcdc;
        color: #238636  ;
         border-radius: 30%;">
  <p>Total Patients</p>
  <p class="sub-title">{total_patients}</p>
</div>
""", unsafe_allow_html=True)

  col9.markdown(f"""
<div style="text-align :center;
 background-color: #dcdcdc;
        color: #238636;
         border-radius: 30%;">
  <p>Patient without AI report</p>
  <p class="sub-title">{no_completed_tests}</p>
</div>
""", unsafe_allow_html=True)


st.write("")
  st.write("")
  st.markdown("### Quick Navigation")
  st.write("")
  st.write("")


# Create buttons for departments
  col5, col6, col7, col8 = st.columns([0.3,0.3,0.3,0.9])
  with col5:
    if st.button("Cardiology", key="cardiology"):
        st.success("You opened the Cardiology Department!")
        st.session_state.page = "cardiology"
        st.rerun()
  with col6:
    if st.button("Neurology", key="neurology"):
        st.success("You opened the Neurology Department!")
        st.session_state.page = "neurology"
        st.rerun()
  with col7:
    if st.button("Pulmonology", key="pulmonology"):
        st.success("You opened the Pulmonology Department!")
        #st.session_state.page = "pulmonology"
        #st.rerun()
  with col8:
    if st.button("Hepatology", key="hepatology"):
        st.success("You opened the Hepatology Department!")
        st.session_state.page = "hepatology"
        st.rerun()

  st.markdown("</div>", unsafe_allow_html=True)

  st.markdown("### Patients With No AI Reports")
  # Dropdown to select the department
  department = st.selectbox(
    "Select Department",
    ["cardiology", "neurology", "pulmonology", "hepatology"]
     )
   # Button to show patients without reports in the selected department
  no_report_patient =st.button("Show Patients Without Reports",key="no_erport")
  if no_report_patient :
    conn = sqlite3.connect("healthcare.db")
    cursor = conn.cursor()
    # Get the patients without reports for the selected department
    patients_df = get_patients_without_reports(department)

    if patients_df.empty:
        st.write(f"No patients found in the {department} department without reports.")
    else:
        st.write(f"Patients in {department.capitalize()} department without reports:")
        st.dataframe(patients_df)

    # Close the connection
    conn.close()



elif st.session_state.page =="cardiology":
  cardiology_content()

#elif st.session_state.page =="pulmonology":
  #pulmonology_content()
elif st.session_state.page =="neurology":
  neurology_content()
elif st.session_state.page =="hepatology":
  hapatology_content()





