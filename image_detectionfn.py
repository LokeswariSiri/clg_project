from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os




#root.mainloop()

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
	print("hi")
	file=createnotepad()
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
                
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		orig = cv2.putText(orig, s, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		print(s)
		file.write("\n")
		file.write(s)
	file.close()
	return orig

def main_method():
        
        #image2 = img
        image2 = cv2.imread(x)
        array = [image2]
        
        for i in range(0,1):
                for img1 in array:
                        image0 = cv2.resize(img1, (640,320), interpolation = cv2.INTER_CUBIC)
                        orig = cv2.resize(img1, (640,320), interpolation = cv2.INTER_LINEAR)
                        #cv2.imshow("test",image2)
                        textDetected = text_detector(image0)
                        #cv2.imshow("Orig Image",orig)
                        cv2.imshow("Text Detection", textDetected)
                       
                        
                        time.sleep(2)
                        k = cv2.waitKey(30)
                        if k == 27:
                                break

root = Tk()
bg = PhotoImage(file = "bgimg.png")
  
# Show image using label
label1 = Label( root, image = bg)
label1.place(x = 0, y = 0)
root.title("Text Detection and Recognition")
root.geometry("8000x750+300+150")
root.resizable(width=True, height=True)
#l1=tk.Label(root,text="Text Detection and Recognition",font=("Algerian",20))
w = tk.Label(root, text="Text Detection and Recognition", font=("Algerian",40,'bold'), fg = "brown", bg = "yellow", pady=10, padx=10)
#w.config(anchor=CENTER)
#w.pack()
w.pack()
#l1=tk.Label(root,text="Name",font=("Algerian",20))
#l1.pack(column=0, row=0)
#t1=tk.Entry(root,width=50,bd=5)
#t1.grid(column=1, row=0)
v = StringVar()

v.set("Text \n Text")

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    global x
    global img
    x = openfn()
    img = Image.open(x)
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img, justify=LEFT)
    panel.image = img
    panel.pack()


        

btn = Button(root, text='open image',font = ("Helvetica", 15),  fg = "blue", bg = "orange", command=open_img).pack(pady=20)
btn = Button(root, text='Select_Notepad_RecognizeText',font = ("Helvetica", 15),  fg = "blue", bg = "violet", command=main_method).pack()




                        
cv2.destroyAllWindows()
root.mainloop()

