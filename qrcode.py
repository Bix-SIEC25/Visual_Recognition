import cv2 as cv
import cvzone
from pyzbar import pyzbar as bar

#Open webcam
cap = cv.VideoCapture(0)
OUTPUT = None
while 1:
    ret, frame = cap.read()

    result = bar.decode(frame)

    for data in result:
        OUTPUT = data.data
        print(data.data)
    cvzone.putTextRect(frame, 'QrCode Scanner', (190,30), scale=2, thickness=2, border =2)
    cvzone.putTextRect(frame, str(OUTPUT), (40,300), scale=2, thickness=2, border = 2)
    OUTPUT = None        
    cv.imshow('frame', frame)
    cv.waitKey(1)