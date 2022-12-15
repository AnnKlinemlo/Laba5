import io
from tkinter import Image

import keras
from PIL import ImageQt
from keras.models import load_model
import numpy as np

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import io
from PIL import Image
from tkinter.messagebox import showinfo

model = keras.models.load_model(r'C:\Users\Аня\Desktop\Work\ISIT\mymodel')
name = np.array(['Круг', 'Квадрат', 'Звезда', 'Треугольник'])


def predict_digit(img):
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    img.save(buffer, "PNG")
    img = Image.open(io.BytesIO(buffer.data()))
    img = img.convert('RGB')
    img = img.resize((200, 200))
    img = np.array(img)
    x = np.asarray([np.array(img)])
    predict_result = model.predict(x, verbose=2)
    result_ind = np.argmax(predict_result)
    return name[result_ind]

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная5")
        self.setGeometry(400, 400, 400, 400)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.label = QLabel("", self)
        self.label.setGeometry(125, 100, 100, 100)
        self.drawing = False
        self.brushSize = 40
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
        mainMenu = self.menuBar()
        recognizeAction = QAction("Ответ", self)
        mainMenu.addAction(recognizeAction)
        recognizeAction.triggered.connect(self.recognize)
        clearAction = QAction("Очистить", self)
        mainMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def recognize(self):
        o = predict_digit(self.image)
        showinfo(title="Ответ", message="Вы нарисовали: " + o)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

def main():
    app = QApplication(sys.argv)
    window = App()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
