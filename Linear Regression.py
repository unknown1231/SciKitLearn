from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


x = np.array([65, 65, 62, 67, 69, 65, 61, 67], dtype=np.float64)
y = np.array([105, 125, 110, 120, 140, 135, 95, 130], dtype=np.float64)



def best_fit_slope(x, y):
    m = ( (((mean(x) * mean(y)) - mean(x*y)) /
         ((mean(x) * mean(x)) - mean(x*x))) )
    b = mean(y) - (m * mean(x))
    return m, b

m, b = best_fit_slope(x, y)

regression_line = [(m*x) + b for ex in x]

predict_x = 71.9
predict_y = (m * predict_x) + b

print('given height to predict weight(in inches): ', predict_x)
print('weight in pound: ', predict_y)

plt.scatter(predict_x, predict_y)
plt.plot(x, y)
plt.show()


