from utils.classifiers import *
from utils.preprocess import *
from utils.features import *
from utils.clustering import *
from pprint import pprint
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from paint import *

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the window
window.show()

# start the app
App.exec()
