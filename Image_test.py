import cv2
for i in range(37):
    cvNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
    img = cv2.imread("./Input/%d.jpg" %(i), cv2.IMREAD_COLOR)
    img=cv2.resize(img, (500, 800))
    #scale_percent = 20 # Процент от изначального размера
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        
            #winname = "Test"
            #cv2.namedWindow(winname)        # Create a named window
            #cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
            #cv2.imshow(winname, img)

            cv2.imwrite("./Output/%d.jpg" %(i), img)

    print(detection)

    #cv2.waitKey(0)
