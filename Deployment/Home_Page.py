import streamlit as st
from datetime import datetime
import pandas as pd
import glob

st.set_page_config(
    page_title="Attendance System", page_icon="📊", layout="wide"
)
# Details of the project
st.title("📊 Attendance System")
# Project Details
st.markdown("Welcome to the Attendance System application. This system allows you to efficiently manage and track attendance for various events, classes, or meetings.")
st.markdown("With its intuitive interface and powerful features, you can easily capture attendance using camera or manual input, view attendance records, and generate insightful visualizations.")

# Additional project details
st.markdown("Project Features:")
st.markdown("- View Attendance: Select and view attendance data for different dates and attendance modes.")
st.markdown("- Camera Attendance: Capture attendance using facial recognition from live camera feed.")
st.markdown("- Manual Attendance: Record attendance manually for situations where camera attendance is not available.")
st.markdown("- Attendance Visualization: Generate visualizations and insights from attendance data.")

# Add any other relevant information about the project, such as its purpose, benefits, or key functionality.

# End the project details section with a separator
st.write("---")

# functionalities for selecting the available sheets and viewing the attendance data as well as providing visualizations
st.title("🔎 View Attendance")
st.write("---")
path = r"./Attendance/"
# Select options for manual and camera attendance
tab1, tab2 = st.tabs(["Camera Attendance", "Manual Attendance"])
with tab1:
    d_l = glob.glob(path + "With_Camera/*.csv")
    for i in range(len(d_l)):
        d_l[i] = d_l[i].split("/")[-1].split(".")[0]
    sheet = st.selectbox("Select Date", d_l)
    df = pd.read_csv(path+"With_Camera/"+str(sheet)+".csv")
    st.table(df)
    st.write("---")
    st.write("###")
    

with tab2:
    d_l = glob.glob(path + "Manual/*.csv")
    for i in range(len(d_l)):
        d_l[i] = d_l[i].split("/")[-1].split(".")[0]
    sheet = st.selectbox("Select Date", d_l)
    df = pd.read_csv(path+"Manual/"+str(sheet)+".csv")
    st.table(df)
    st.write("---")
    st.write("###")