'''Imports pics from one folder, searches the faces, resizes and saves into the another folder'''
import cv2, glob, pathlib


filepaths = glob.glob('sample-images/*.jpg')
img_names = [pathlib.Path(filepath).stem for filepath in filepaths]
imges = [cv2.imread(filepath) for filepath in filepaths]
gray_imges = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imges]

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = [cascade.detectMultiScale(img, 1.05, 5) for img in gray_imges]

faced_imges = []
for img, faces_per_img in zip(imges, faces):
    for x,y,h,w in faces_per_img:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
    img_ratio = img.shape[0]/300
    img = cv2.resize(img, (int(img.shape[1]/img_ratio), 300))
    faced_imges.append(img)


for img, name in zip(faced_imges, img_names):
    cv2.imwrite(f'resized-images/{name}.jpg', img)


for img in faced_imges:
    cv2.imshow('Face', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()






