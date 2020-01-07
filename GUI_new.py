# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import cv2
import numpy as np
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf
import pickle
from fr_utils import *
from inception_blocks_v2 import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLineEdit, QInputDialog
from multiprocessing import Process, Pool
import re
import sys
from mtcnn import MTCNN
import time
import base64
import google_drive
class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(692, 402)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_load_pickle = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_load_pickle.setGeometry(QtCore.QRect(80, 140, 201, 28))
        self.pushButton_load_pickle.setObjectName("pushButton_load_pickle")
        self.textEdit_threshold = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_threshold.setGeometry(QtCore.QRect(90, 30, 191, 31))
        self.textEdit_threshold.setObjectName("textEdit_threshold")
        self.label_threshold = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold.setGeometry(QtCore.QRect(10, 40, 71, 16))
        self.label_threshold.setObjectName("label_threshold")
        self.label_pickle_path = QtWidgets.QLabel(self.centralwidget)
        self.label_pickle_path.setGeometry(QtCore.QRect(10, 90, 71, 16))
        self.label_pickle_path.setObjectName("label_pickle_path")
        self.textEdit_pickle_path = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_pickle_path.setGeometry(QtCore.QRect(90, 80, 191, 31))
        self.textEdit_pickle_path.setObjectName("textEdit_pickle_path")
        self.pushButton_recognize_an_image = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_recognize_an_image.setGeometry(QtCore.QRect(400, 180, 201, 28))
        self.pushButton_recognize_an_image.setObjectName("pushButton_recognize_an_image")
        self.pushButton_add_image_from_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_image_from_video.setGeometry(QtCore.QRect(400, 220, 201, 28))
        self.pushButton_add_image_from_video.setObjectName("pushButton_add_image_from_video")
        self.pushButton_add_image_from_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_image_from_camera.setGeometry(QtCore.QRect(400, 260, 201, 28))
        self.pushButton_add_image_from_camera.setObjectName("pushButton_add_image_from_camera")
        self.pushButton_recognize_realtime_with_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_recognize_realtime_with_camera.setGeometry(QtCore.QRect(400, 140, 201, 28))
        self.pushButton_recognize_realtime_with_camera.setObjectName("pushButton_recognize_realtime_with_camera")
        self.pushButton_take_a_picture = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_take_a_picture.setGeometry(QtCore.QRect(80, 260, 201, 28))
        self.pushButton_take_a_picture.setObjectName("pushButton_take_a_picture")
        self.textEdit_database_path = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_database_path.setGeometry(QtCore.QRect(410, 80, 191, 31))
        self.textEdit_database_path.setObjectName("textEdit_database_path")
        self.label_image_size = QtWidgets.QLabel(self.centralwidget)
        self.label_image_size.setGeometry(QtCore.QRect(310, 40, 71, 16))
        self.label_image_size.setObjectName("label_image_size")
        self.textEdit_image_size = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_image_size.setGeometry(QtCore.QRect(410, 30, 191, 31))
        self.textEdit_image_size.setObjectName("textEdit_image_size")
        self.label_database_path = QtWidgets.QLabel(self.centralwidget)
        self.label_database_path.setGeometry(QtCore.QRect(310, 90, 101, 16))
        self.label_database_path.setObjectName("label_database_path")
        self.pushButton_save_pickle = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save_pickle.setGeometry(QtCore.QRect(80, 180, 201, 28))
        self.pushButton_save_pickle.setObjectName("pushButton_save_pickle")
        self.pushButton_load_database = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_load_database.setGeometry(QtCore.QRect(80, 220, 201, 28))
        self.pushButton_load_database.setObjectName("pushButton_load_database")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 692, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.recognize = False
        self.face_database = {}
        self.THRESHOLD = 0.5
        self.FACE_SIZE = 120
        self.take_a_picture = False
        self.detector = MTCNN(min_face_size=self.FACE_SIZE,scale_factor=0.9)
        self.path = "raw"
        self.retranslateUi(MainWindow)
        self.set_on_click()
        self.textEdit_threshold.setText(str(self.THRESHOLD))
        self.textEdit_image_size.setText(str(self.FACE_SIZE))
        self.textEdit_database_path.setText(str(self.path))
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_load_pickle.setText(_translate("MainWindow", "Load pickle"))
        self.label_threshold.setText(_translate("MainWindow", "Threshold:"))
        self.label_pickle_path.setText(_translate("MainWindow", "Pickle path:"))
        self.pushButton_recognize_an_image.setText(_translate("MainWindow", "Recognize an image"))
        self.pushButton_add_image_from_video.setText(_translate("MainWindow", "Add image from video"))
        self.pushButton_add_image_from_camera.setText(_translate("MainWindow", "Add image from camera"))
        self.pushButton_recognize_realtime_with_camera.setText(_translate("MainWindow", "Recognize video "))
        self.pushButton_take_a_picture.setText(_translate("MainWindow", "Take a picture"))
        self.label_image_size.setText(_translate("MainWindow", "Image size: "))
        self.label_database_path.setText(_translate("MainWindow", "Database path:"))
        self.pushButton_save_pickle.setText(_translate("MainWindow", "Save pickle"))
        self.pushButton_load_database.setText(_translate("MainWindow", "Load database"))
    def set_on_click(self):
        self.pushButton_save_pickle.clicked.connect(self.click_save_pickle)
        self.pushButton_load_pickle.clicked.connect(self.click_load_pickle)
        self.pushButton_load_database.clicked.connect(self.load_database)
        self.pushButton_recognize_realtime_with_camera.clicked.connect(self.click_recognize)
        self.pushButton_recognize_an_image.clicked.connect(self.recognize_image)
        self.pushButton_add_image_from_camera.clicked.connect(self.click_create_raw)
        self.pushButton_take_a_picture.clicked.connect(self.click_take_a_picture)
        self.pushButton_add_image_from_video.clicked.connect(self.click_add_image_from_video)
    def click_save_pickle(self):
        with open("pickle.txt", "wb") as f:
            pickle.dump(self.face_database, f)
            self.show_message("Thông báo", "Lưu thành công")
    def getText(self, object):
        text = object.toPlainText()

        return text
    def click_load_pickle(self):
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        print(self.face_database)
        with open("pickle.txt", "rb") as f:
            self.face_database = pickle.load(f)
            self.show_message("Thông báo", "Load dữ liệu thành công")
        
    def load_database(self):
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        for name in os.listdir(self.path):
            face_array = []
            identity = os.path.splitext(os.path.basename(name))[0]
            for image in os.listdir(os.path.join(self.path,name)):
                embedding = fr_utils.img_path_to_encoding(os.path.join(self.path,name,image), FR_model)
                if embedding !=  "None":
                    face_array.append(embedding)
                else:
                    print("can't find any face in %s", (os.path.join(self.path,name,image)))
            if len(face_array) > 0:
                self.face_database[identity] = face_array.copy()
            else:
                print("can't find any face in %s", (os.path.join(self.path,name)))
        print(self.face_database)
        self.show_message("Thông báo", "Load dữ liệu thành công")

    def show_message(self, title, message):
        title.encode('utf-8')
        message.encode('utf-8')
        message = QMessageBox.question(self, title, message, QMessageBox.Yes)
    def show_input_dialog(self, title,message):
        text, result = QInputDialog.getText(self,title,message)
        if result == True:
            return text
        else:
            return ""
    def click_create_raw(self):
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        if self.recognize == False:
            self.recognize = True
        count = 0
        i = 0
        face_array = []
        name = self.show_input_dialog("Thông báo", "Vui lòng nhập tên: ")
        if name == "":
            self.recognize = False
            self.show_message("Thông báo", "Hủy thêm dữ liệu thành công")
            return
        if not os.path.exists("raw/"+name):
            os.makedirs("raw/"+name, exist_ok = 'True')
        number_of_image = self.show_input_dialog("Thông báo", "Vui lòng nhập số lượng ảnh")
        while type(number_of_image) != int:
            try:
                if number_of_image == "":
                    self.recognize = False
                    self.show_message("Thông báo", "Hủy thêm dữ liệu thành công")
                    return
                number_of_image = int(number_of_image)
            except:
                number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh phải là 1 số nguyên: ")
        video_capture = cv2.VideoCapture(0)
        self.detector = MTCNN(min_face_size=self.FACE_SIZE,scale_factor=0.9)
        while self.recognize == True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            faces = crop_face(frame, self.detector)
            frame_ret = frame.copy()
            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                encoding = img_to_encoding(roi, FR_model)
                face_array.append(encoding)
                cv2.rectangle(frame_ret, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.imshow("frame", frame_ret)
            if len(faces) > 0 and i % 5 == 0:
                path = "raw/"+name+"/"+str(count)+".jpg"
                
                cv2.waitKey(1)
                save_image(frame, path)
                count = count + 1
            i = i+1
            if count == number_of_image:
                self.recognize = False
                self.face_database[name] = face_array
                self.show_message("Thông báo", "Thêm dữ liệu hoàn tất")
        video_capture = video_capture.release
        cv2.destroyAllWindows()

    def click_add_image_from_video(self):
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        if self.recognize == False:
            self.recognize = True
        count = 0
        i = 0
        face_array = []
        name = self.show_input_dialog("Thông báo", "Vui lòng nhập tên: ")

        if not os.path.exists("raw/"+name):
            os.makedirs("raw/"+name, exist_ok = 'True')
        number_of_image = self.show_input_dialog("Thông báo", "Vui lòng nhập số lượng ảnh")
        while type(number_of_image) != int:
            try:
                number_of_image = int(number_of_image)
            except:
                number_of_image = self.show_input_dialog("Thông báo", "Số lượng ảnh phải là 1 số nguyên: ")
        video_path = self.show_input_dialog("Thông báo", "Vui lòng nhập đường dẫn video: ")
        video_capture = cv2.VideoCapture(video_path)
        self.detector = MTCNN(min_face_size=self.FACE_SIZE,scale_factor=0.9)
        while self.recognize == True:
            ret, frame = video_capture.read()
            if ret == False:
                self.show_message("Thông báo", "Không thể đọc được Video")
                self.recognize = False
                break
            # frame = cv2.flip(frame, 1)
            faces = crop_face(frame, self.detector)
            frame_ret = frame.copy()
            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                encoding = img_to_encoding(roi, FR_model)
                face_array.append(encoding)
                cv2.rectangle(frame_ret, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.imshow("frame", frame_ret)
            if len(faces) > 0 and i % 5 == 0:
                path = "raw/"+name+"/"+str(count)+".jpg"
                
                cv2.waitKey(10)
                save_image(frame, path)
                count = count + 1
            i = i+1
            if count == number_of_image or cv2.waitKey(1) == ord('q'):
                self.recognize = False
                self.face_database[name] = face_array
                self.show_message("Thông báo", "Thêm dữ liệu hoàn tất")
        video_capture = video_capture.release
        cv2.destroyAllWindows()
    def click_take_a_picture(self):
        if self.take_a_picture == False:
            self.take_a_picture = True
        else:
            self.take_a_picture = False
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        if len(self.face_database)==0:
            self.show_message("Thông báo", "Tập dữ liệu khuôn mặt rỗng.")
            return 
        detector = MTCNN(min_face_size=self.FACE_SIZE,scale_factor=0.9)
        if self.recognize == True:
            return
        self.recognize = True
        video_capture = cv2.VideoCapture(0)
        while self.recognize == True:
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            faces = crop_face(frame, detector)

            for(x,y,w,h) in faces:
                print(x,y,w,h)
                if x <= 0 or y <= 0 or h <= 0 or w <= 0:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                roi = frame[y:y+h, x:x+w]
                encoding = img_to_encoding(roi, FR_model)
                min_dist = 100
                identity = "None"
                for(name, encoded_image_list) in self.face_database.items():
                    for encoded_image_name in encoded_image_list:
                        dist = np.linalg.norm(encoding - encoded_image_name)
                        if(dist < min_dist):
                            min_dist = dist
                            identity = name
                    print('Min dist: ',min_dist)
                if min_dist < self.THRESHOLD:
                    cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No matching faces ' + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                    cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            cv2.imshow('Face Recognition System', frame)
            if(cv2.waitKey(1) == ord('q')) or self.recognize== False or self.take_a_picture == False:
                self.recognize = False
                name_ori = self.show_input_dialog("Thông báo", "Nhập tên bức ảnh: ")
                name_and_code = name_ori.split("_")
                name = name_and_code[0]
                code = "None_Code"
                if len(name_and_code)>=2:
                    code = name_and_code[1]
                seconds = time.time()
                local_seconds = time.localtime(seconds)
                path_time = str(local_seconds.tm_hour) + "_"+str(local_seconds.tm_min)+"_"+str(local_seconds.tm_sec)+"_"+str(local_seconds.tm_mday)+"_"+str(local_seconds.tm_mon)+"_"+str(local_seconds.tm_year)
                path_name = "take_a_pic//"+name_ori+"_"+path_time+".jpg"
                path_name_copy = "take_a_pic//"+name_ori+"_copy_"+path_time+".jpg"
                save_image(frame,path_name)
                save_image(frame_copy,path_name_copy)
                # frame_copy = cv2.resize(frame_copy,(160,160))
                # im_send = cv2.cvtColor(frame_copy,cv2.COLOR_BGR2RGB)
                # Im_To_Byte = base64.b64encode(im_send)
                # Byte_to_String = Im_To_Byte.decode()
                
                hour = str(local_seconds.tm_hour)
                minute = str(local_seconds.tm_min)
                seconds = str(local_seconds.tm_sec)
                day = str(local_seconds.tm_mday)
                month = str(local_seconds.tm_mon)
                year = str(local_seconds.tm_year)
                notify(name,code,path_time, path_name_copy,hour, minute, seconds, day,month,year)
                self.show_message("Thông báo", "Lưu thành công")
                break
        # path = self.show_input_dialog("Thông báo","Vui lòng nhập đường dẫn đến ảnh")
        # while path == "":
        #     path = self.show_input_dialog("Thông báo","Vui lòng nhập đường dẫn đến ảnh")
        # recognize_image(path, self.THRESHOLD, self.face_database,self.detector)
        video_capture = video_capture.release()
        cv2.destroyAllWindows()
    def click_recognize(self):
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        if len(self.face_database)==0:
            self.show_message("Thông báo", "Tập dữ liệu khuôn mặt rỗng.")
            return 
        detector = MTCNN(min_face_size=self.FACE_SIZE,scale_factor=0.9)
        if self.recognize == True:
            return
        self.recognize = True
        video_path = self.show_input_dialog("Thông báo", "Nhập đường dẫn video: ")
        video_path_copy = ""
        try:
            video_path_copy = int(video_path)
            video_path = video_path_copy
        except:
            video_path_copy = ""
        video_capture = cv2.VideoCapture(video_path)
        while self.recognize == True:
            ret, frame = video_capture.read()
            if ret == False:
                self.show_message("Thông báo", "Không thể đọc video")
                self.recognize = False
                break
            # if video_path_copy !="":
            #     frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            faces = crop_face(frame, detector)

            for(x,y,w,h) in faces:
                print(x,y,w,h)
                if x <= 0 or y <= 0 or h <= 0 or w <= 0:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                roi = frame[y:y+h, x:x+w]
                encoding = img_to_encoding(roi, FR_model)
                min_dist = 100
                identity = "None"
                for(name, encoded_image_list) in self.face_database.items():
                    for encoded_image_name in encoded_image_list:
                        dist = np.linalg.norm(encoding - encoded_image_name)
                        if(dist < min_dist):
                            min_dist = dist
                            identity = name
                    print('Min dist: ',min_dist)
                if min_dist < self.THRESHOLD:
                    cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    name_and_code = identity.split("_")
                    name = name_and_code[0]
                    code = name_and_code[1]
                    seconds = time.time()
                    local_seconds = time.localtime(seconds)
                    path_time = str(local_seconds.tm_hour) + "_"+str(local_seconds.tm_min)+"_"+str(local_seconds.tm_sec)+"_"+str(local_seconds.tm_mday)+"_"+str(local_seconds.tm_mon)+"_"+str(local_seconds.tm_year)
                    path_name = "take_a_pic/"+name+"_"+code+"_"+path_time+".jpg"
                    path_name_copy = "take_a_pic/"+name+"_"+code+"_copy_"+path_time+".jpg"
                    save_image(frame,path_name)
                    save_image(frame_copy,path_name_copy)
                    # frame_copy = cv2.resize(frame_copy,(160,160))
                    # im_send = cv2.cvtColor(frame_copy,cv2.COLOR_BGR2RGB)
                    # Im_To_Byte = base64.b64encode(im_send)
                    # Byte_to_String = Im_To_Byte.decode()
            
                    notify(name,code,path_time, path_name_copy)
                    self.show_message("Thông báo", "Lưu thành công")
                else:
                    cv2.putText(frame, 'No matching faces ' + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                    cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            cv2.imshow('Face Recognition System', frame)
            if(cv2.waitKey(1) & 0xFF == ord('q')) or self.recognize== False:
                self.recognize = False
                break
        # path = self.show_input_dialog("Thông báo","Vui lòng nhập đường dẫn đến ảnh")
        # while path == "":
        #     path = self.show_input_dialog("Thông báo","Vui lòng nhập đường dẫn đến ảnh")
        # recognize_image(path, self.THRESHOLD, self.face_database,self.detector)
        video_capture = video_capture.release()
        cv2.destroyAllWindows()

    def click_recognize_from_video(self):
        video_path = self.show_input_dialog("Thông báo", "Nhập địa chỉ video: ")
        try:
            self.FACE_SIZE = int(self.textEdit_image_size.toPlainText())
        except:
            self.show_message("Thông báo", "FaceSize phải là một số nguyên")
            return
        threshold = self.getText(self.textEdit_threshold)
        try:
            self.THRESHOLD = float(threshold)
        except:
            self.show_message("Thông báo", "THRESHOLD phải là một số thực")
            return
        if len(self.face_database)==0:
            self.show_message("Thông báo", "Tập dữ liệu khuôn mặt rỗng.")
            return 
        detector = MTCNN(min_face_size=self.FACE_SIZE,scale_factor=0.9)
        if self.recognize == True:
            return
        self.recognize = True
        
        video_capture = cv2.VideoCapture(video_path)
        while self.recognize == True:
            ret, frame = video_capture.read()
            if ret == False:
                self.show_message("Thông báo", "Không thể đọc video")
                self.recognize = False
                break
            frame = cv2.flip(frame, 1)

            faces = crop_face(frame, detector)

            for(x,y,w,h) in faces:
                print(x,y,w,h)
                if x <= 0 or y <= 0 or h <= 0 or w <= 0:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                roi = frame[y:y+h, x:x+w]
                encoding = img_to_encoding(roi, FR_model)
                min_dist = 100
                identity = "None"
                for(name, encoded_image_list) in self.face_database.items():
                    for encoded_image_name in encoded_image_list:
                        dist = np.linalg.norm(encoding - encoded_image_name)
                        if(dist < min_dist):
                            min_dist = dist
                            identity = name
                    print('Min dist: ',min_dist)
                if min_dist < self.THRESHOLD:
                    cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No matching faces ' + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                    cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            cv2.imshow('Face Recognition System', frame)
            if(cv2.waitKey(1) & 0xFF == ord('q')) or self.recognize== False:
                self.recognize = False
                break
        video_capture = video_capture.release()
        cv2.destroyAllWindows()
    def recognize_image(self, path, THRESHOLD, face_database,detector):
        faces = {}
        try:
            frame = cv2.imread(path)
            faces = crop_face(frame, detector)
        except:
            print("Không load được ảnh")
        

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
            if min_dist < THRESHOLD:
                cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No matching faces ' + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.imshow('Face Recognition System', frame)
        key = cv2.waitKey(0)
        if key == ord("q"):
            return
from pusher_push_notifications import PushNotifications
def notify(name,code, time, image_path, hour, minute, seconds, day, month, year):
    service = google_drive.get_service()

    image_id = google_drive.upload_file(name+"_"+code+"_"+time,image_path,service) 
    pn_client = PushNotifications(
        instance_id='d565ede4-c39f-4978-8d91-2b249efd7eee',
        secret_key='ACD47A2B2F277C3EFE35FC53535019E716E43CF0085F18A1FFC24C893FC7F943',
    )
    print("Push a message to {}".format(code))
    response = pn_client.publish_to_interests(
        interests=[code],
        publish_body={
            'apns': {
                'aps': {
                    'alert': 'Hello!'
                }
            },
            'fcm': {
                
                "data": {
                    "body":"Em "+name +" đã có mặt tại trường lúc: ",
                    "title":"Xin chào phụ huynh em "+name,
                    "image_id": image_id,
                    "time": time,
                    "code": code,
                    "name": name,
                    "day": day,
                    "month": month,
                    "year": year,
                    "hour": hour,
                    "minute": minute,
                    "seconds": seconds
                }
            }
        }
    )
def crop_face(image, detector):
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
if __name__ == "__main__":
    import sys
    FR_model = InceptionResNetV1()
    FR_model.load_weights('facenet_weights.h5')
    print("Total Params:", FR_model.count_params())
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
