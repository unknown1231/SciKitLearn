import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('breast-cancer-wisconsin.data.txt')
data.replace('?', 0, inplace=True)
data.drop('id', 1, inplace=True)

x = np.array(data.drop('Class', 1))
y = np.array(data['Class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print('accuracy is:', accuracy)

example_test = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_test = example_test.reshape(1, -1)
prediction = clf.predict(example_test)

if prediction == [2]:
    print('cancer is benign')

else:
    print('cancer is malignant')

plt.plot(x, y)
plt.show()

