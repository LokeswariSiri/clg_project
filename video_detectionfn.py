import numpy as np
from imutils.object_detection import non_max_suppression
import time
import cv2
import pytesseract
from tkinter import filedialog

net = cv2.dnn.readNet("frozen_east_text_detection.pb")
def createnotepad():
    filename = filedialog.askopenfilename(title='open')
    f=str(filename)
    lis=f.split("/")
    print(lis)
    #from tkinter import *
    # Python code to create a file
    file = open(filename,'a')
    return file


def text_detector(image):
	orig = image
	(H, W) = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	#file=createnotepad()
	

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 2
		

		text = orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
		text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		textRecongized = pytesseract.image_to_string(text)
		s=""
		for i in textRecongized:
                        if(i.isalnum()):
                                s+=i
                #textRecongized=s
        
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		orig = cv2.putText(orig, s, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		
	return orig
    



font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture("video22.mp4")
#cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open video")

cntr =0
while True:
    ret,frame = cap.read()
    img1=frame
    cntr= cntr+1;
    if((cntr%20)==0):
        image0 = cv2.resize(img1, (960,640), interpolation = cv2.INTER_CUBIC)
        orig = cv2.resize(img1, (640,320), interpolation = cv2.INTER_LINEAR)
                        #cv2.imshow("test",image2)
        textDetected = text_detector(image0)
                        #cv2.imshow("Orig Image",orig)
        cv2.imshow("Text Detection", textDetected)
                       
                        
        time.sleep(2)
        k = cv2.waitKey(30)
        if k == 27:
            break
        #imgH, imgW,_ = frame.shape
        #x1,y1,w1,h1=0,0,imgH,imgW
        #imgchar = pytesseract.image_to_string(frame)
        
        #s=""
        #for i in imgchar:
         #   if(i.isalnum()):
          #      s+=i
          #  elif(i.isspace()):
           #     s+=" "
            #else:
             #   s=s
        #imgboxes = pytesseract.image_to_boxes(frame)
        #print(s)
        #for boxes in imgboxes.splitlines():
         #   boxes= boxes.split(' ')
          #  x,y,w,h= int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
           # cv2.rectangle(frame, (x, imgH-y),(w,imgH-h),(0,0,255),3)

        #cv2.putText(frame, imgchar, (w1,h1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        #cv2.putText(frame, s, (x1 + int(w1/50),y1 + int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #scv2.imshow('Text detection tutorial', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
cap. release( )
cv2.destroyAllWindows( )
