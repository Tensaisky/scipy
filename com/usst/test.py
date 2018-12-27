import sys
import json
import pandas as pd
import datetime
import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.externals import joblib

result = sys.argv[1:]

print(json.dumps(result[1]))