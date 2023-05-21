from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing_extensions import Annotated
from starlette.middleware.cors import CORSMiddleware
import cv2
import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from requests import request
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img != []:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def get_attendance(camera, registered_users):
    """
    Starts the camera and upon recognizing a face from the registered user, instantly adds the attendance of that user and stops the webcam.

    Args:
    camera: A camera object.
    registered_users: A list of registered users.

    Returns:
    A list of users who have attended.
    """

    # Start the camera.
    camera.start()

    # Capture a frame.
    frame = camera.read()

    # Detect faces in the frame.
    faces = cv2.face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face that is detected, compare it to the faces of the registered users.
    for face in faces:
        (x, y, w, h) = face
        face_image = frame[y:y + h, x:x + w]
        face_encoding = cv2.face.faceHaarcascade.detectMultiScale(face_image)

        # If the face matches a registered user, add the user's attendance and stop the camera.
        for registered_user in registered_users:
            if registered_user.face_encoding == face_encoding:
                registered_user.attendance = True
                camera.stop()
                break

    return registered_users

#############################################


@app.get('/')
def home(request: Request):
    names, rolls, times, l = extract_attendance()
    return templates.TemplateResponse('index.html', {
        'request': request,
        'names': names,
        'rolls': rolls,
        'times': times,
        'l': l,
        'totalreg': totalreg(),
        'datetoday2': datetoday2
    })


@app.get('/start')
def start(request: Request):
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return templates.TemplateResponse('home.html', {
            'request': request,
            'totalreg': totalreg(),
            'datetoday2': datetoday2,
            'mess': 'There is no trained model in the static folder. Please add a new face to continue.'
        })

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return templates.TemplateResponse('index.html', {
        'request': request,
        'names': names,
        'rolls': rolls,
        'times': times,
        'l': l,
        'totalreg': totalreg(),
        'datetoday2': datetoday2
    })


@app.post('/add')
async def add(request: Request, newusername: str = Form(...), newuserid: str = Form(...)):
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        cv2.imshow('Adding new User', frame)
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return templates.TemplateResponse('index.html', {
        'request': request,
        'names': names,
        'rolls': rolls,
        'times': times,
        'l': l,
        'totalreg': totalreg(),
        'datetoday2': datetoday2
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1", port=5000')