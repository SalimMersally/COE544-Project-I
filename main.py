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
print("reading XY")
X, Y = get_dataSet_or_process_images()
Y_differ = get_character_vs_number(Y)

(X_train_1, X_test_1, Y_train_1, Y_test_1) = splitDataSet(X, Y_differ)
svmModel = getSVM(X_train_1, Y_train_1, "svm_split")
Y_predict_1 = svmModel.predict(X_test_1)
accuracy_1 = accuracy_score(Y_test_1, Y_predict_1)

print("splitting")
(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)
print("validation metrics: ")
print("accuracy: " + str(accuracy_1))

# print("reading DT")
# svmModel = getSVM(X_train, Y_train, "svm")
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
# # Creating a instance of label Encoder.
# le = LabelEncoder()
# # Using .fit_transform function to fit label
# # encoder and return encoded label
# import keras

# Y = le.fit_transform(Y)


# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# print("New shape of train data:", x_train.shape)

# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# print("New shape of test data:", x_test.shape)

# y_train = keras.utils.to_categorical(y_train, num_classes=62, dtype="int")
# y_test = keras.utils.to_categorical(y_test, num_classes=62, dtype="int")


# preds = cnn.predict(x_test)

# for pred in preds:
#     for i in range(len(pred)):
#         if pred[i] == pred.max():
#             pred[i] = 1.0
#         else:
#             pred[i] = 0.0

# print(preds[0])
# print(y_test[0])

# CalculatedAccuracy = sum(preds == y_test) / len(y_test)
# metric = keras.metrics.Accuracy()
# acc = metric(y_test, preds)
# print(acc)

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the window
window.show()

# start the app
App.exec()
