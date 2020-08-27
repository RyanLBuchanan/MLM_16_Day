# Lesson 1
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('Python: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('Python: {}'.format(np.__version__))
# matplotlib
import matplotlib
print('Python: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('Python: {}'.format(pd.__version__))
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

# Lesson 7 
# Evaluate Using Cross-Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0));

# Lesson 8: Algorithm Evaluation Metrics
# Cross-Validation Classification LogLoss
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
url = 'https://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("LogLoss: %.3f (%.3f)" % (results.mean(), results.std()))

# Lesson 9: Spot-Check Algorithms
# KNN Regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
url = 'https://goo.gl/FmJUSM'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())

# Lesson 10: Model Comparison and Selection
# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Load dataset
url = 'http://goo.gl/bDdBiA'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
X = array[:, 0:8]
Y = array[:, 8]
# Prepare models
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
# Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Lesson #11: Improve Accuracy with Algorithm Tuning
# Grid search for Algorithm Tuning
from pandas import read_csv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
url = "https://goo.gl/bDdBiA"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:,8]
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)