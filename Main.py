import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['x1', 'x2', 'x3', 'x4', 'y']
y = pd.read_csv('iris.data.txt', names=names, usecols=[4]).values
x = pd.read_csv('iris.data.txt', names=names, usecols=[0, 1, 2, 3]).values
new_x = np.ones((150,1))
new_x = np.hstack((new_x, x))
x_transpose = np.transpose(new_x)
for i in range(0, 150):
    if y[i] == 'Iris-setosa':
        y[i] = -1
    elif y[i] == 'Iris-versicolor':
        y[i] = 0
    elif y[i] == 'Iris-virginica':
        y[i] = 1
xt_mul_x = np.matmul(x_transpose, new_x)
x_inv = np.linalg.inv(xt_mul_x)
x_inv_mul_xt = np.matmul(x_inv,x_transpose)
y = y.astype(float)
beta = np.matmul(x_inv_mul_xt, y)
print(beta)

#calculating the y_prediction

y_prediction = np.matmul(new_x, beta)
print(y_prediction.shape)

#Calculating the error

Total_error = 0
for i in range(0, 150):
    error = (y_prediction[i]-y[i])*(y_prediction[i]-y[i])
    Total_error = Total_error + error

print(Total_error)
print("Accuracy is %d",  100 - Total_error)

#K-fold..
k1 = [3, 5, 6, 10, 15 ]
data1 = np.hstack((new_x, y))
np.random.shuffle(data1)

#data1 = np.array(data1)
#temp = np.split(data1, k)
avg_error_list = []
error_list = []
for k in k1:
    for i in range(0, k):
        temp = np.array(np.split(data1, k))
        #print(temp.shape)
        test_matrix=temp[i]
        row = int(test_matrix.shape[0])
        train = np.delete(temp, i, 0).reshape((150-row, 6))
        test_x = test_matrix[:,[0, 1, 2, 3, 4]]
        test_y = test_matrix[:,[5]]
        train_x = train[:,[0, 1, 2, 3, 4]]
        train_y = train[:,[5]]
        t_train_x = np.transpose(train_x)
        step1 = np.matmul(t_train_x, train_x)
        step2 = np.linalg.inv(step1)
        step3 = np.matmul(step2, t_train_x)
        k_beta = np.matmul(step3, train_y)
        test_prediction = np.matmul(test_x, k_beta)
        Total_error_k = 0
        for i in range(0, row):
            k_error = (test_prediction[i] - test_y[i]) * (test_prediction[i] - test_y[i])
            Total_error_k = Total_error_k + k_error

        error_list.append(Total_error_k)

    print(error_list)
    sum = 0
    for j in error_list:
        sum = sum+j
        avg_sum = sum/k
    print("Error for k=%d:", k)
    print(avg_sum)
    avg_error_list.append(avg_sum)

# Data for plotting
fig, ax = plt.subplots()
ax.plot(k1, avg_error_list)

ax.set(xlabel='N value (s)', ylabel='Error Value',
       title='Error Vs N value')
ax.grid()
plt.show()

