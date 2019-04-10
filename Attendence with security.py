import cv2 as cv
import cv2.face
import pickle
import sqlite3

label = {1: "Unknown"}
conn = sqlite3.connect('pythonDB.db')
c = conn.cursor()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainner.yml")
with open("labels.pickel", "rb") as f:
    org_label = pickle.load(f)
    label = {v:k for k,v in org_label.items()}

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS Attendence_REC (Roll_Number REAL, Name TEXT, Attendence TEXT)')

def data_entry(value):
    number = 1649210042
    name = value
    at = "Present"
    c.execute("INSERT INTO Attendence_REC (Roll_Number, Name, Attendence) VALUES(?, ?, ?)", (number, name, at))
    conn.commit()

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        #cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        id_, confidance = recognizer.predict(roi_gray)
        if confidance>=90:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv.FONT_HERSHEY_SIMPLEX
            name = label[id_]
            color = (255, 255, 255)
            stroke = 2
            cv.putText(img, name, (x, y), font, 0.8, color, stroke, cv.LINE_AA)
    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    if len(faces)==2:
        create_table()
        data_entry(name)
        break

cap.release()
cv.destroyAllWindows()
c.close()
conn.close()

print(label)
