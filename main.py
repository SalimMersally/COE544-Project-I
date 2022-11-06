from utils.classifiers import *
from utils.preprocess import *
from utils.features import *
from utils.featureChoice import *
from pprint import pprint
from sklearn.metrics import accuracy_score
from paint import *

print("reading XY")
X, Y = get_dataSet_or_process_images()
print("splitting")
(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)

print("reading svm")
svmModel = getSVM(X_train, Y_train)
print("predicting")
Y_predict = svmModel.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)
print(str(accuracy))

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the window
window.show()

# start the app
App.exec()
