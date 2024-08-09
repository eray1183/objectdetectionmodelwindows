import sqlite3
import pandas as pd

import pickle
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit,QLineEdit,QPushButton)
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QPixmap


from ultralytics import YOLO
import cv2 
import matplotlib.pyplot as plt


import sys,os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog
 

class MLModelWindow(QWidget):
    def __init__(self):
        super().__init__()
 
        # Ana widget ve layout
        self.setWindowTitle("ML Model Uygulaması")
        self.setGeometry(100, 100, 800, 600)  # Pencere boyutu
 
        # Layoutlar
        main_layout = QVBoxLayout()  # Ana layout dikey olacak
        top_layout = QHBoxLayout()   # Üst kısım için yatay layout
        bottom_layout = QVBoxLayout()  # Alt kısım için dikey layout
 
        # Foto Seç: label ve text edit
        model_label = QLabel("Fotoğraf Seçin:")#MODEL SEÇ LABELI
        model_label.setFont(QFont('Arial', 14))
        top_layout.addWidget(model_label)
 
        #self.model_edit = QLineEdit("modelYolo")
        #self.model_edit.setFont(QFont('Arial', 14))
        #top_layout.addWidget(self.model_edit)
        self.run_button = QPushButton("Galeri")#çalıştır
        self.run_button.clicked.connect(self.model_tani)
        top_layout.addWidget(self.run_button)

        # text label
        self.class_label = QLabel()# CLASS SONUÇ LABEL TEXT
        self.class_label.setFont(QFont('Arial', 24))  # Yazı boyutunu büyütüyoruz
        self.class_label.setAlignment(Qt.AlignCenter)  # Ortalıyoruz
        bottom_layout.addWidget(self.class_label)
 
        # Görsel
        self.image_label = QLabel(self)#CLASS SONUÇ LABEL İMAGE
        #pixmap = QPixmap("/path/to/your/image.png")  # Buraya kendi resim dosyanızın yolunu koymalısınız
        #self.image_label.setPixmap(pixmap.scaled(400, 400, aspectRatioMode=True))
        self.image_label.setAlignment(Qt.AlignCenter)  # Ortalıyoruz
        bottom_layout.addWidget(self.image_label)
 
        # Layoutları ana layouta ekleme
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)
 
        # Ana layoutu widgeta ekleme
        self.setLayout(main_layout)






        #---------------------------------------------------------------------------#
        
    

    def model_tani(self):
        model_path='./model/best.pt'
        dosya_yolu, _ = QFileDialog.getOpenFileName(self,"Görüntü Seç", 
                                                    os.path.expanduser("~/Desktop"),"Görüntü Dosyaları (*.jpg *.jpeg *.png)")
        source=''
        target='C:/Users/AINOS-1/Desktop/yolov10/test/sonuç/test.jpg'
        
        model = YOLO(model_path) # Model dosyasını okuma işlemi

        results =model.predict(dosya_yolu)
        
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(cls)] for cls in class_indices]  # Map indices to names
            print(class_names)
            textOut=class_names[0]#iç içe for tüm classları al
            self.class_label.setText(textOut)
            #self.sonuc_label.setText(results)

        processed_img = results[0].plot()  # Bu işlem sonucu matplotlib formatında olacaktır

        # Görüntüyü kaydetmek için, matplotlib'in imshow ve savefig fonksiyonları
        plt.imshow(processed_img)
        
        plt.axis('off')  
        plt.savefig(target, bbox_inches='tight', pad_inches=0) # Elde edilen çıktının kaydedileceği konum
        plt.close()
        self.image_label.setPixmap(QtGui.QPixmap(target))


    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MLModelWindow()
    window.show()
    sys.exit(app.exec_())