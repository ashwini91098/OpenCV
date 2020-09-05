import cv2
#create cascade classifier object
face_cascade=cv2.CascadeClassifier("/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
img=cv2.imread("ash.jpg")
#reading image as gray scale img
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#search the co-ord of the img
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.21,minNeighbors=5)
print(type(faces))
print(faces)
#rectangular face box
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
resized=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))    
cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()    
