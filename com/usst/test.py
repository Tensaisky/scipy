import sys
import json
import pandas as pd
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.externals import joblib
import pywt

a = np.array([1,2,3,4,5,6])
print(a)
b = a.reshape(1,-1)
print(b)