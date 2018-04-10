"""
Final

https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

Breast Cancer Classification

Number of Instances: 699
Number of Attributes: 8
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

## x: Attributes
## y: Class (1=malignant, 0=benign)
x = []
y = []

attrCols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]

with open("breast-cancer-wisconsin.data") as data_file:
    lines = data_file.readlines()
    entries = []
    for line in lines:
        entry = line.strip().split(",")

        entry = [int(entry[i]) for i in attrCols]

        if entry[9] == 2:
            entry[9] = 0
        elif entry[9] == 4:
            entry[9] = 1
        

        entries.append(entry)
        x_temp = [entry[i] for i in range(1, 9)]
        
        x.append(np.array(x_temp))
        y.append(np.array(entry[9]))


headers = ['ID Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', \
          'Single Epithelial Cell Size', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

## DataFrame of Data
data_df = pd.DataFrame(entries, columns=headers)

print data_df.head(15), '\n'

for i in range(15):
    #print entries[i]
    print x[i], y[i]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

print 'x_train:', '('+str(len(x_train))+',', str(len(x_train[0]))+'),', \
      'x_test:', '('+str(len(x_test))+',', str(len(x_test[0]))+')'
print 'y_train:', '('+str(len(y_train))+',', str(1)+'),', \
      'y_test:', '('+str(len(y_test))+',', str(1)+')'


## Logistic Regression Classifier
logRegModel = LogisticRegression(max_iter=300)

logRegModel.fit(x_train, y_train)
logRegPredictTrain = logRegModel.predict(x_train)
logRegPredictTest = logRegModel.predict(x_test)


accuracy = 0
for i in range(len(logRegPredictTest)):
    if logRegPredictTest[i] == y_test[i]:
        accuracy += 1

print "Mean Squared Error - Logistic Regression:\n\tTrain(", \
      round(metrics.mean_squared_error(logRegPredictTrain, y_train), 5), ")\n\tTest(", \
      round(metrics.mean_squared_error(logRegPredictTest, y_test), 5), ")"

print 'Accuracy:', str(accuracy) + '/' + str(len(logRegPredictTest)), \
      '---', str(100 * round(float(accuracy) / len(logRegPredictTest), 4)) + '%'



## Neural Net - Multi-Layer Perceptron Classifier
neuralNetModel = MLPClassifier(max_iter=300)

neuralNetModel.fit(x_train, y_train)
neuralNetPredictTrain = neuralNetModel.predict(x_train)
neuralNetPredictTest = neuralNetModel.predict(x_test)


accuracy = 0
for i in range(len(neuralNetPredictTest)):
    if neuralNetPredictTest[i] == y_test[i]:
        accuracy += 1

print "Mean Squared Error - Multi-Layer Perceptron:\n\tTrain(", \
      round(metrics.mean_squared_error(neuralNetPredictTrain, y_train), 5), ")\n\tTest(", \
      round(metrics.mean_squared_error(neuralNetPredictTest, y_test), 5),")"

print 'Accuracy:', str(accuracy) + '/' + str(len(neuralNetPredictTest)), \
      '---', str(100 * round(float(accuracy) / len(neuralNetPredictTest), 4)) + '%'



## Dimensionality Reduction via PCA and Support Vector Classifier

pcaModel = PCA(n_components=2)
svmModel = SVC(kernel='rbf', gamma=0.7, C=0.6)

pcaReduce = pcaModel.fit_transform(x)

print "After Dimensionality Reduction:", pcaReduce.shape

x_r_train, x_r_test, y_r_train, y_r_test = train_test_split(pcaReduce, y, \
                                                            test_size=0.3, random_state=15)


svmModel.fit(x_r_train, y_r_train)
svmPredictTrain = svmModel.predict(x_r_train)
svmPredictTest = svmModel.predict(x_r_test)

accuracy = 0
for i in range(len(svmPredictTest)):
    if svmPredictTest[i] == y_r_test[i]:
        accuracy += 1


print "Mean Squared Error - Support Vector Machine:\n\tTrain(", \
      round(metrics.mean_squared_error(svmPredictTrain, y_r_train), 5), ")\n\tTest(", \
      round(metrics.mean_squared_error(svmPredictTest, y_r_test), 5),")"

print 'Accuracy:', str(accuracy) + '/' + str(len(svmPredictTest)), \
      '---', str(100 * round(float(accuracy) / len(svmPredictTest), 4)) + '%'


## Rearrange Data for Plotting

pos = []; neg = []
pos0 = []; pos1 = []
neg0 = []; neg1 = []
X = []; Y = []

for i in range(len(svmPredictTest)):
    
    if svmPredictTest[i] == 1:
        pos0.append(x_r_test[i][0])
        pos1.append(x_r_test[i][1])
    elif svmPredictTest[i] == 0:
        neg0.append(x_r_test[i][0])
        neg1.append(x_r_test[i][1])

pos.append(pos0); pos.append(pos1)
neg.append(neg0); neg.append(neg1)

X.append(pos[0]); X.append(neg[0])
X = [x for sublist in X for x in sublist]
Y.append(pos[1]); Y.append(neg[1])
Y = [y for sublist in Y for y in sublist]

## Create Decision Boundary
h = .02

x_min, x_max = min(X) - 1, max(X) + 1
y_min, y_max = min(Y) - 1, max(Y) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svmModel.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)


## Plot
fig, ax = plt.subplots()

fig.patch.set_facecolor("#e6e6e6")
fig.patch.set_alpha(0.35)


ax.contour(xx, yy, Z, cmap=plt.cm.Paired)
        

ax.plot(neg[0], neg[1], 'bo', label=r"$benign$")
ax.plot(pos[0], pos[1], 'rx', label=r"$malignant$")

xax = ax.set_xlabel("$Z_1$", fontsize=15)
yax = ax.set_ylabel("$Z_2$  ", fontsize=15)
yax.set_rotation(0)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

ax.legend(loc=0, ncol=2, fontsize=12, frameon=False)
ax.set_title("Classifying Breast Cancer -- SVM + PCA Test Set", fontdict=font)

plt.savefig("classification-plot.pdf", facecolor=fig.get_facecolor(), edgecolor='none')

plt.show()

