import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
import glob

st.set_page_config(
    page_title="Attendance System", page_icon="📊", layout="wide"
)

st.title(":camera: Take Attendance with Camera")
st.write("---")

# Load student encodings
df = pd.read_csv("./data/encodings/encodings.csv")
en = df["Encodings"]
n = df["Persons"]

e = []
for i in en:
    e.append(np.fromstring(i[1:-1], dtype=float, sep=' '))

def detect_known_faces(img, image_encodings=e, persons=n):
    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (int(width/4), int(height/4)))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    fc = []
    fn = []
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations, model="small")
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(image_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = persons[first_match_index]
        fc.append(face_location)
        fn.append(name)
    return fc, fn


path = "./Attendance/With_Camera/Camera "
d_l = glob.glob(path+"*.csv")
for i in range(len(d_l)):
    d_l[i] = d_l[i].split("/")[-1].split(".")[0]


with st.expander("Create New Attendance Sheet"):
    name_of_attendance_sheet = st.text_input("Enter Name of Attendance Sheet")
    if st.button("Create New Attendance Sheet"):
        now = datetime.now()
        date = now.strftime("%d-%m-%Y")
        df = pd.DataFrame(columns=['Date', 'Time', 'Name', 'Status'])
        name_of_attendance_sheet = name_of_attendance_sheet+" "+str(date)
        df.to_csv("./Attendance/With_Camera/Camera " + 
                  name_of_attendance_sheet+".csv", index=False)

with st.expander("Select Attendance Sheets"):
    date = st.selectbox("Select Sheet", d_l)
    name_of_attendance_sheet = date



with st.container():
    c = st.columns(1)
    with c[0]:
        start = st.button("Start Taking Attendance")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# list of column names
field_names = ['Date', 'Time', 'Name', 'Status']

now = datetime.now()
date = now.strftime("%d-%m-%Y")
time = now.strftime("%H:%M:%S")

df = pd.read_csv("./Attendance/With_Camera/Camera " +
                 name_of_attendance_sheet+".csv")

if start:
    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations, face_names = detect_known_faces(frame)
        if face_names:
            name = face_names[0]
            y1, x2, y2, x1 = face_locations[0]
            cv2.putText(frame, name, (x1*4, y1*4 - 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1*4, y1*4), (x2*4, y2*4), (0, 0, 200), 4)
            if name not in df['Name'].values:
                new_row = {'Date': date, 'Time': time, 'Name': name, 'Status': 'Present'}
                df = df.append(new_row, ignore_index=True)
                st.success(f"Attendance taken for: {name}")
                break
            else:
                st.info(f"Attendance already taken for: {name}")
                break
        FRAME_WINDOW.image(frame)

camera.release()
cv2.destroyAllWindows()

df.to_csv("./Attendance/With_Camera/Camera " + name_of_attendance_sheet+".csv", index=False)