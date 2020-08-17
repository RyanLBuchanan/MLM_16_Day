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

# dataframe
my_array = np.array([[1,2,3],[4,5,6]])
row_names = ['a', 'b']
col_names = ['one', 'two', 'three']
my_dataframe = pd.DataFrame(my_array, index = row_names, columns = col_names)
print(my_dataframe)
