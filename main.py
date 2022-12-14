from utils.classifiers import *
from utils.preprocess import *
from utils.features import *
from utils.clustering import *
from pprint import pprint
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from paint import *

# print("cluster dataset")
# cluster_dataSet()
# get_excel()
# print("reading XY")
# X, Y = get_dataSet_or_process_images()
# print("splitting")
# (X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)

# print("reading DT")
# # svmModel = getSVM(X_train, Y_train)
# print("predicting")
# Y_predict = svmModel.predict(X_test)
# accuracy = accuracy_score(Y_test, Y_predict)
# precision = precision_score(Y_test, Y_predict, average="weighted")
# recall = recall_score(Y_test, Y_predict, average="weighted")
# fscore = f1_score(Y_test, Y_predict, average="weighted")

# print("validation metrics: ")
# print("accuracy: " + str(accuracy))
# print("precision: " + str(precision))
# print("recall: " + str(recall))
# print("F score: " + str(fscore))

# cnn = getCNN()
# X, Y = get_dataSet_or_process_images()

# X = np.array(X).astype("float32")
# X = np.expand_dims(X, axis=-1)

# le = LabelEncoder()
# Y = le.fit_transform(Y)
# Y = keras.utils.to_categorical(Y, 62)

# preds = cnn.predict(X)

# CalculatedAccuracy = sum(preds == Y) / len(Y)
# metric = keras.metrics.Accuracy()
# acc = metric(Y, preds)
# print(acc, CalculatedAccuracy)

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the window
window.show()

# start the app
App.exec()
