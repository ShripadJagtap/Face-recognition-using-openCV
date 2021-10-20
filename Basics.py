import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('pythonProject/ImagesBasic/Avi.jpg')          # lodaing image file by giving appropriate path 
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)                                           # converting image format to RGB from BGR
imgTest = face_recognition.load_image_file('pythonProject/ImagesBasic/Buuger.jpg')          # lodaing image file by giving appropriate path
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)                                         # converting image format to RGB from BGR

faceLoc = face_recognition.face_locations(imgElon)[0]                                      # recognizing face location of the training image
encodeElon = face_recognition.face_encodings(imgElon)[0]                                    # recongnizing face structure of the training image 
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)        # darwing rectangle around the face of the training image

faceLocTest = face_recognition.face_locations(imgTest)[0]                                               # recognizing face location of the test image
encodeTest = face_recognition.face_encodings(imgTest)[0]                                                # recongnizing face structure of the test image
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)     # darwing rectangle around the face of the test image

results = face_recognition.compare_faces([encodeElon],encodeTest)                                          # comparing the encodings to see whether the training image matches the test image
faceDis = face_recognition.face_distance([encodeElon],encodeTest)                                           # showing results of comparison  in percentage
print(results,faceDis)                                                                                      # print results in console
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)     # putting results above the bounding box made before

cv2.imshow('Elon Musk' ,imgElon)                                                            # showing training image on screen
cv2.imshow('Elon Test' ,imgTest)                                                            # showing test image on screen
cv2.waitKey(0)                                                                              # show the images until user interuption