'''Face Recognition Main File'''
import cv2
import numpy as np
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *

#with CustomObjectScope({'tf': tf}):
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLineEdit, QInputDialog
from multiprocessing import Process, Pool
import re
import sys

class MyWindow(QMainWindow):
	recognize = False
	def __init__(self) :
		super(MyWindow,self).__init__()
		self.setGeometry(800, 300, 500, 500)
		self.setWindowTitle('Demo One Shot Learning.')
		self.setUI()
	def setUI(self  ) :
		# button recognize
		self.buttonRecognize = QtWidgets.QPushButton(self)
		self.buttonRecognize.move(160, 150)
		self.buttonRecognize.setText('Recognizer')
		# button addImage
		self.buttonAddImage = QtWidgets.QPushButton(self)
		self.buttonAddImage.move(160, 200)
		self.buttonAddImage.setText('Add Image')
		# button Stop Recognize
		self.buttonStopRecognize = QtWidgets.QPushButton(self)
		self.buttonStopRecognize.move(160, 250)
		self.buttonStopRecognize.setText('Stop Recognize')
		# button add image from dataset 
		self.button_add_image_from_dataset = QtWidgets.QPushButton(self)
		self.button_add_image_from_dataset.move(160, 300)
		self.button_add_image_from_dataset.setText('Add image from source')
		# button create raw dataset 
		self.button_create_raw = QtWidgets.QPushButton(self)
		self.button_create_raw.move(160, 350)
		self.button_create_raw.setText('create raw')
		# button check camera
		self.button_check_camera = QtWidgets.QPushButton(self)
		self.button_check_camera.move(160, 400)
		self.button_check_camera.setText('Check camera')
		# add event click
		self.buttonAddImage.clicked.connect(self.click_add_image_from_camera)
		self.buttonRecognize.clicked.connect(self.click_recognize)
		self.buttonStopRecognize.clicked.connect(self.click_stop_recognize)
		# self.button_add_image_from_dataset.clicked.connect(
		#     self.add_image_from_database)
		self.button_check_camera.clicked.connect(self.click_check_performance)
		self.button_create_raw.clicked.connect(self.click_create_raw)
	def click_stop_recognize(self):
		self.recognize = False
	def click_recognize(self):
		if self.recognize == True:
			return
		self.recognize = True
		video_capture = cv2.VideoCapture(0)
		while self.recognize == True:
			ret, frame = video_capture.read()
			frame = cv2.flip(frame, 1)

			faces = crop_face(frame)

			for(x,y,w,h) in faces:
				print(x,y,w,h)
				if x <= 0 or y <= 0 or h <= 0 or w <= 0:
					continue
				cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
				roi = frame[y:y+h, x:x+w]
				encoding = img_to_encoding(roi, FR_model)
				min_dist = 100
				identity = "None"

				for(name, encoded_image_list) in face_database.items():
					for encoded_image_name in encoded_image_list:
						dist = np.linalg.norm(encoding - encoded_image_name)
						if(dist < min_dist):
							min_dist = dist
							identity = name
					print('Min dist: ',min_dist)
				if min_dist < 0.7:
					cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
					cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				else:
					cv2.putText(frame, 'No matching faces ' + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
					cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

			cv2.imshow('Face Recognition System', frame)
			if(cv2.waitKey(1) & 0xFF == ord('q')) or self.recognize== False:
				break

		video_capture = video_capture.release()
		cv2.destroyAllWindows()
	def show_input_dialog(self, title,message):
		text, result = QInputDialog.getText(self,title,message)
		if result == True:
			return text
		else:
			return ""
	def show_message(self, title, message):
		title.encode('utf-8')
		message.encode('utf-8')
		message = QMessageBox.question(self, title, message, QMessageBox.Yes)
	def click_create_raw(self):
		if self.recognize == False:
			self.recognize = True
		count = 0
		i = 0
		name = self.show_input_dialog("Thông báo", "Vui lòng nhập tên: ")
		if not os.path.exists("raw/"+name):
			os.makedirs("raw/"+name, exist_ok = 'True')
		number_of_image = self.show_input_dialog("Thông báo", "Vui lòng nhập số lượng ảnh")
		while type(number_of_image) != int:
			try:
				number_of_image = int(number_of_image)
			except:
				number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh phải là 1 số nguyên: ")
		video_capture = cv2.VideoCapture(0)
		while self.recognize == True:
			ret, frame = video_capture.read()
			frame = cv2.flip(frame, 1)
			faces = crop_face(frame)
			frame_ret = frame.copy()
			for (x,y,w,h) in faces:
				cv2.rectangle(frame_ret, (x, y), (x + w, y + h), (255, 255, 0), 2)
			cv2.imshow("frame", frame_ret)
			if len(faces) > 0 and i % 5 == 0:
				path = "raw/"+name+"/"+str(count)+".jpg"
				
				cv2.waitKey(20)
				save_image(frame, path)
				count = count + 1
			i = i+1
			if count == number_of_image:
				self.recognize = False
				self.show_message("Thông báo", "Thêm dữ liệu hoàn tất")
		video_capture = video_capture.release
		cv2.destroyAllWindows()
	def click_add_image_from_camera(self):
		if self.recognize == True:
			return
		self.recognize = True
		video_capture = cv2.VideoCapture(0)
		face_array = []
		count = 0
		name = self.show_input_dialog("Thông báo", "Vui lòng nhập tên: ")
		number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh: ")
		
		im_list = []
		while type(number_of_image) != int:
			try:
				number_of_image = int(number_of_image)
			except:
				number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh phải là 1 số nguyên: ")
		path = 'images'
		directory = os.path.join(path, name)
		while self.recognize == True:
			ret, frame = video_capture.read()
			frame = cv2.flip(frame, 1)

			faces = crop_face(frame)
			
			for(x,y,w,h) in faces:
				print(x,y,w,h)
				roi = frame[y:y+h, x:x+w]
				encoding = img_to_encoding(roi, FR_model)
				if len(face_array) == 0:
					
					print(directory)
					if not os.path.exists(directory):
						os.makedirs(directory, exist_ok = 'True')
					face_array.append(encoding)
					face_database[name] = face_array
					image_name = "images/"+name+"/"+str(count)+".jpg"
					im_list.append(roi)
					count = 1
				else:
					min_dist = 100
					identity = "None"
				
					for(folder, encoded_image_list) in face_database.items():
						for encoded_image_name in encoded_image_list:
							dist = np.linalg.norm(encoding - encoded_image_name)
							if(dist < min_dist):
								min_dist = dist
								identity = folder
					print('Min dist: ',min_dist)
					cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
					if identity == name:
						cv2.putText(frame, "Face : " + identity + str(count), (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
						cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
					else:
						cv2.putText(frame, 'No matching faces', (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
						cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
						face_array.append(encoding)
						
						im_list.append(roi)
						count = count+1

			cv2.imshow('Face Recognition System', frame)
			if(cv2.waitKey(1) & 0xFF == ord('q')) or self.recognize== False or count == number_of_image:
				self.recognize = False
				face_database[name] = face_array.copy()
				self.show_message("Thông báo", "Thêm dữ liệu hoàn tất")
				pool = Pool(8)
				for p in range(0, len(im_list)):
					image_name = "images/"+name+"/"+str(p)+".jpg"
					
					pool.apply_async(save_image, args=(im_list[p],image_name)) 
					# Process(target=save_image,args=(im_list[p],image_name,)).start()
				break

		video_capture = video_capture.release()
		cv2.destroyAllWindows()
	def click_check_performance(self):
		if self.recognize == True:
			return
		self.recognize = True
		video_capture = cv2.VideoCapture(0)
		count = 0
		correct = 0
		false = 0
		name = self.show_input_dialog("Thông báo", "Vui lòng nhập tên: ")
		number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh: ")
		im_list = []
		while type(number_of_image) != int:
			try:
				number_of_image = int(number_of_image)
			except:
				number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh phải là 1 số nguyên: ")
		path = 'images'
		directory = os.path.join(path, name)
		while self.recognize == True:
			ret, frame = video_capture.read()
			frame = cv2.flip(frame, 1)

			faces = crop_face(frame)
			
			for(x,y,w,h) in faces:
				roi = frame[y:y+h, x:x+w]
				encoding = img_to_encoding(roi, FR_model)
				min_dist = 100
				identity = None
			
				for(folder, encoded_image_list) in face_database.items():
					for encoded_image_name in encoded_image_list:
						dist = np.linalg.norm(encoding - encoded_image_name)
						if(dist < min_dist):
							min_dist = dist
							identity = folder
					print('Min dist: ',min_dist)
					cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
				if identity == name:
					cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
					cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
					count = count + 1
					correct = correct + 1
				elif identity != name:
					cv2.putText(frame, 'No matching faces: '+identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
					cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
					false = false+1
					count = count+1

			cv2.imshow('Face Recognition System', frame)
			if(cv2.waitKey(30) & 0xFF == ord('q')) or self.recognize == False or count == number_of_image:
				self.recognize = False
				self.show_message("Thông báo", "Nhận diện hoàn tất")
				print("Correct: ", str(correct))
				print("False: ", str(false))
				# pool = Pool(8)
				# for p in range(0, len(im_list)):
				# 	image_name = "images/"+name+"/"+str(p)+".jpg"
				# 	pool.apply_async(save_image, args=(im_list[p],image_name)) 
				# Process(target=save_image,args=(im_list[p],image_name,)).start()
				break

		video_capture = video_capture.release()
		cv2.destroyAllWindows()
from mtcnn import MTCNN
detector = MTCNN(min_face_size=120,scale_factor=0.9)
def crop_face(image):
	faces = []
	image_to_detect = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	result = detector.detect_faces(image_to_detect)
	if len(result) == 0:
		return []
	i  = 0
	while i<len(result):
		bounding_box = result[i]['box']
		if bounding_box[0] >= 0 and bounding_box[1] >= 0:
			faces.append((bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]))
		i = i+1
	return faces
def save_image(frame, name):
    ret, frame = cv2.imencode(".jpg", frame)
    f = open(name,'wb')
    f.write(frame.tobytes())
    f.close()
def Window():
	app = QApplication(sys.argv)
	win = MyWindow()
	win.show()
	sys.exit(app.exec_())

from inception_resnet_v1 import *

if __name__ == "__main__":
	# FR_model = load_model('nn4.small2.v1.h5')
	FR_model = InceptionResNetV1()
	FR_model.load_weights('facenet_weights.h5')


	print("Total Params:", FR_model.count_params())

	face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
	

	threshold = 0.25

	face_database = {}
	
	for name in os.listdir('raw'):
		face_array = []
		identity = os.path.splitext(os.path.basename(name))[0]
		for image in os.listdir(os.path.join('raw',name)):
			embedding = fr_utils.img_path_to_encoding(os.path.join('raw',name,image), FR_model)
			if embedding !=  "None":
				face_array.append(embedding)
			else:
				print("can't find any face in %s", (os.path.join('raw',name,image)))
		if len(face_array) > 0:
			face_database[identity] = face_array.copy()
		else:
			print("can't find any face in %s", (os.path.join('raw',name)))
	print(face_database)
	Window()
