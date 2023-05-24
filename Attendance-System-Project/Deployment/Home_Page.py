import streamlit as st
from datetime import datetime
import pandas as pd
import glob

st.set_page_config(
    page_title="Attendance System", page_icon="📊", layout="wide"
)
# st.image("logo.png")

# functionalities for selecting the available sheets and viewing the attendance data as well as providing visualizations
st.title("🔎 View Attendance")
st.write("---")
path = r"./Attendence/"
# Select options for manual and camera attendance
tab1, tab2 = st.tabs(["Manual Attendance", "Camera Attendance"])
with tab1:
    d_l = glob.glob(path + "Manual/*.csv")
    for i in range(len(d_l)):
        d_l[i] = d_l[i].split("/")[-1].split(".")[0]
    sheet = st.selectbox("Select Date", d_l)
    df = pd.read_csv(path+"Manual/"+str(sheet)+".csv")
    st.table(df)
    st.write("---")
    st.write("###")

with tab2:
    d_l = glob.glob(path + "With_Camera/*.csv")
    for i in range(len(d_l)):
        d_l[i] = d_l[i].split("/")[-1].split(".")[0]
    sheet = st.selectbox("Select Date", d_l)
    df = pd.read_csv(path+"With_Camera/"+str(sheet)+".csv")
    st.table(df)
    st.write("---")
    st.write("###")