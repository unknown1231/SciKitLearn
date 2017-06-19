import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

x = np.array([['unknown', 9], [3, 5], [7, 6], [21, 8], [12, 10], [13, 14], [1.3, 1.4], [15.3, 16.7]])


# plt.scatter(x[:, 0], x[:, 1], s=150)
# plt.show()

classifier = KMeans(n_clusters=2)
classifier.fit(x)

centroid = classifier.cluster_centers_
lables = classifier.labels_

colors = ["g.", "b.", "r."]

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[lables[i]], markersize=25)
plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=150, linewidths=5)
plt.show()

