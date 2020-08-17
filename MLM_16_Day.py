# Lesson 1
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('Python: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('Python: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('Python: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('Python: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('Python: {}'.format(sklearn.__version__))


# Lesson 2
# dataframe
my_array = np.array([[1,2,3],[4,5,6]])
row_names = ['a', 'b']
col_names = ['one', 'two', 'three']
my_dataframe = pd.DataFrame(my_array, index = row_names, columns = col_names)
print(my_dataframe)


# Lesson 3
# Load CSV using Pandas from URL
from pandas import read_csv
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names = names)
print(data.shape)


# Lesson 4
# Statistical Summary
description = data.describe()
print(description)


# Lesson 5
# Scatter Plot Matrix
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names = names)
scatter_matrix(data)
plt.show()


# Lesson 6
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import numpy as np
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names = names)
array = dataframe.values
# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = StandardScaler().fit(X)
rescaled_X = scaler.transform(X)
# Summarize transformed data
np.set_printoptions(precision = 3)
print(rescaled_X[0:5, :])
