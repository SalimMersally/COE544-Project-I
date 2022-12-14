from utils.classifiers import *
from utils.preprocess import *
from utils.features import *
from utils.clustering import *
from pprint import pprint
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from paint import *


character = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]

print("cluster dataset")
# cluster_dataSet()
# get_excel()
print("reading XY")
X, Y = get_dataSet_or_process_images()
Y2 = []

for y in Y:
    index = 0
    for i in range(len(character)):
        if y == character[i]:
            index = i
            break
    if index <= 9:
        Y2.append(0)
    else:
        Y2.append(1)

print("splitting")
(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y2)

print("reading DT")
svmModel = getSVM(X_train, Y_train)
print("predicting")
Y_predict = svmModel.predict(X_test)
print(Y_predict)
accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict, average="weighted")
recall = recall_score(Y_test, Y_predict, average="weighted")
fscore = f1_score(Y_test, Y_predict, average="weighted")

print("validation metrics: ")
print("accuracy: " + str(accuracy))
print("precision: " + str(precision))
print("recall: " + str(recall))
print("F score: " + str(fscore))

cnn = getCNN()
X, Y = get_dataSet_or_process_images()


X = np.array(X).astype("float32")
# Creating a instance of label Encoder.
le = LabelEncoder()
# Using .fit_transform function to fit label
# encoder and return encoded label
import keras

Y = le.fit_transform(Y)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
print("New shape of train data:", x_train.shape)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print("New shape of test data:", x_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes=62, dtype="int")
y_test = keras.utils.to_categorical(y_test, num_classes=62, dtype="int")


preds = cnn.predict(x_test)

for pred in preds:
    for i in range(len(pred)):
        if pred[i] == pred.max():
            pred[i] = 1.0
        else:
            pred[i] = 0.0

print(preds[0])
print(y_test[0])

CalculatedAccuracy = sum(preds == y_test) / len(y_test)
metric = keras.metrics.Accuracy()
acc = metric(y_test, preds)
print(acc)

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the window
window.show()

# start the app
App.exec()
