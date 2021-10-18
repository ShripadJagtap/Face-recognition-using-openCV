import cv2
import numpy as np
import face_recognition
import os

path = 'pythonProject/ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)