import pandas

from sklearn.neighbors import KNeighborsRegressor

dataset = pandas.read_csv("abalone.csv")

target = dataset.iloc[:,8]
target = target + 1.5
target = target.values

data = dataset.iloc[:,0:8]
data = pandas.get_dummies(data, columns=['Sex'])
data = data.values
print(data)
