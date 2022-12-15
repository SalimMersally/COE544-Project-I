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

print("reading XY")
X, Y = get_Clustered_No_Feature()

(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)

X_feature = []
for x in X_train:
    X_feature.append(
        np.concatenate(
            (
                get_dense_SIFT(x),
                get_HOG(x),
                get_proj_histogram_horz(x),
                get_proj_histogram_vert(x),
            ),
            axis=0,
        )
    )

Y_differ = get_character_vs_number(Y_train)
(X_train_1, X_test_1, Y_train_1, Y_test_1) = splitDataSet(X_feature, Y_differ)
svmModel = getSVM(X_train_1, Y_train_1, "svm_split")
Y_predict_1 = svmModel.predict(X_test_1)
accuracy_1 = accuracy_score(Y_test_1, Y_predict_1)
print("Accuracy of split: " + str(accuracy_1))

X_no_digit = []
Y_no_digit = []

for i in range(len(X_feature)):
    y = Y_train[i]
    for j in range(len(character)):
        if character[j] == y:
            if j >= 10:
                X_no_digit.append(X_feature[i])
                Y_no_digit.append(Y_train[i])

(X_train_2, X_test_2, Y_train_2, Y_test_2) = splitDataSet(X_no_digit, Y_no_digit)

mlp = getMLP(X_no_digit, Y_no_digit, "clf-character")
Y_predict_2 = mlp.predict(X_test_2)
accuracy_2 = accuracy_score(Y_test_2, Y_predict_2)
print("validation metrics: ")
print("accuracy: " + str(accuracy_2))

cnn = getCNN()

Y_predict = []

for x in X_test:
    feature = np.concatenate(
        (
            get_dense_SIFT(x),
            get_HOG(x),
            get_proj_histogram_horz(x),
            get_proj_histogram_vert(x),
        ),
        axis=0,
    )
    predict_1 = svmModel.predict([feature])[0]
    if predict_1 == 0:
        x_np = np.array([x]).astype("float32")
        x_np = np.expand_dims(x_np, -1)
        result = cnn.predict(x_np)
        for pred in result:
            for i in range(len(pred)):
                if pred[i] == pred.max():
                    pred[i] = 1
                else:
                    pred[i] = 0
        decoded = decodeResult(result[0])
        Y_predict.append(decoded[0])
    else:
        Y_predict.append(mlp.predict([feature])[0])

accuracy = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict, average="weighted")
recall = recall_score(Y_test, Y_predict, average="weighted")
fscore = f1_score(Y_test, Y_predict, average="weighted")

print("validation metrics: ")
print("accuracy: " + str(accuracy))
print("precision: " + str(precision))
print("recall: " + str(recall))
print("F score: " + str(fscore))
