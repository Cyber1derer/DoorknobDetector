import cv2
import time
last_time=0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

cvNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

while True:
    ret, frame = cap.read()
    rows = frame.shape[0]
    cols = frame.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            print("--- %s seconds ---" % (time.time() - last_time))
            last_time=time.time()
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    cv2.imshow('door_detect', frame)
    c = cv2.waitKey(1)
    if c ==27:
        break
    