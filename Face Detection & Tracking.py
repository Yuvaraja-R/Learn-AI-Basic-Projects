import cv2

alg = "haarcascade_frontalface_default.xml"  # Initializing Algorithm
haar_cascade = cv2.CascadeClassifier(alg)  # Loading Algorithm
cam = cv2.Videocapture(1)  # Initializing camara id online
'''
video_path = "Videofile.mp4"  # Video file in folder
cam.cv2.imread(video_path)
'''
'''
video_path = "yuvi_temple.jpg"  # picture file  in folder
img = cv2.imread(video_path)
'''
while True:  # Infinite loop
    _, img = cam.read()  # Reading Frame from camara
    grayImg = cv2.cvtColor(img, cv2.color_BGR2GRAY)  # Converting color image into  to grey scale image
    face = haar_cascade.detectMultiscale(grayImg, 1.3, 4)  # Getting coordinates
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)   #Drawing a rectangle

    cv2.imshow("FaceDetection", img)
    key = cv2.waitkey(10)
    print(key)
    if key == 27:
        break
cam.release()
cv2.destroyallwindows()
