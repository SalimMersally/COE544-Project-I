# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import sys
from utils.classifiers import *
from utils.features import *
from utils.preprocess import *
import os
from utils.nlp import correct_word
from PyQt5.QtWidgets import QPushButton


# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Paint")

        # setting geometry to main window
        self.setGeometry(100, 100, 800, 600)

        mainWindow = QWidget()

        pybutton = QPushButton("Predict", self)
        pybutton.resize(100, 32)
        pybutton.move(300, 500)
        pybutton.clicked.connect(self.predict)

        pybutton = QPushButton("Clear", self)
        pybutton.resize(100, 32)
        pybutton.move(400, 500)
        pybutton.clicked.connect(self.clear)

        self.prediction = QLabel(self)
        self.prediction.setText("Prediction: ")
        self.prediction.move(300, 530)
        
        self.likely = QLabel(self)
        self.likely.setText("Other: ")
        self.likely.move(300, 550)
        self.likely.resize(200, 40)
        self.likely.setWordWrap(True)

        # creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)

        # making image color to white
        self.image.fill(Qt.white)

        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 12
        # default color
        self.brushColor = Qt.black

        # QPoint object to tract the point
        self.lastPoint = QPoint()

        # creating menu bar
        mainMenu = self.menuBar()

        # creating file menu for save and clear action
        fileMenu = mainMenu.addMenu("File")

        # adding brush color to ain menu
        b_color = mainMenu.addMenu("Brush Color")

        # creating save action
        saveAction = QAction("Save", self)
        # adding short cut for save action
        saveAction.setShortcut("Ctrl + S")
        # adding save to the file menu
        fileMenu.addAction(saveAction)
        # adding action to the save
        saveAction.triggered.connect(self.save)

        # creating clear action
        clearAction = QAction("Clear", self)
        # adding short cut to the clear action
        clearAction.setShortcut("Ctrl + C")
        # adding clear to the file menu
        fileMenu.addAction(clearAction)
        # adding action to the clear
        clearAction.triggered.connect(self.clear)

        # creating predict action
        predictAction = QAction("Predict", self)
        fileMenu.addAction(predictAction)
        predictAction.triggered.connect(self.predict)

        # creating options for brush color
        # creating action for black color
        black = QAction("Black", self)
        # adding this action to the brush colors
        b_color.addAction(black)
        # adding methods to the black
        black.triggered.connect(self.blackColor)

        # similarly repeating above steps for different color
        white = QAction("White", self)
        b_color.addAction(white)
        white.triggered.connect(self.whiteColor)

        green = QAction("Green", self)
        b_color.addAction(green)
        green.triggered.connect(self.greenColor)

        yellow = QAction("Yellow", self)
        b_color.addAction(yellow)
        yellow.triggered.connect(self.yellowColor)

        red = QAction("Red", self)
        b_color.addAction(red)
        red.triggered.connect(self.redColor)

    # method for checking mouse cicks
    def mousePressEvent(self, event):

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:

            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(
                QPen(
                    self.brushColor,
                    self.brushSize,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin,
                )
            )

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())

            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    # method for saving canvas
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) "
        )

        if filePath == "":
            return
        self.image.save(filePath)

    # method for clearing every thing on canvas
    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

    def predict(self):
      self.image.save("./image.png")
      img4 = cv2.imread("./image.png")
      svmModel = getSVM(None, None)
      
      try:
        dummy, letters = find_bounding_box(img4) 
        word = ""
        for letter in letters: 
            hog = get_HOG(letter)
            sift = get_dense_SIFT(letter)
            result = svmModel.predict([np.concatenate((sift, hog), axis=0)])
            word += result[0]
        
        corrected = correct_word(word)
        
        if(corrected==None):
                print(word)
                self.prediction.setText("Prediction: " + word)
                self.likely.setText("Other: ")
                os.remove("./image.png")
                self.clear()
        else: 
                print(word)
                word = corrected[0]
                self.prediction.setText("Prediction: " + word)
                print(corrected[1])
                self.likely.setText("Other: "+corrected[1])
                os.remove("./image.png")
                self.clear()
      except: 
            # print("error!")
            os.remove("./image.png")
            self.likely.setText("Other: ")
            self.clear()    
        

    # methods for changing brush color
    def blackColor(self):
        self.brushColor = Qt.black

    def whiteColor(self):
        self.brushColor = Qt.white

    def greenColor(self):
        self.brushColor = Qt.green

    def yellowColor(self):
        self.brushColor = Qt.yellow

    def redColor(self):
        self.brushColor = Qt.red
