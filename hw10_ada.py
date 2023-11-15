###
### CS667 Data Science with Python, Homework 10, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df_cmg = pd.read_csv("cmg_weeks.csv")
df_spy = pd.read_csv("spy_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")



X_train = df_cmg[['Avg_Return', 'Volatility']][df_cmg['Week'] <= 50].values
Y_train = df_cmg[['Color']][df_cmg['Week'] <= 50].values
X_test = df_cmg[['Avg_Return', 'Volatility']][(df_cmg['Week'] > 50) & (df_cmg['Week'] <= 100)].values
Y_test = df_cmg[['Color']][(df_cmg['Week'] > 50) & (df_cmg['Week'] <= 100)].values


# Use decision tree, gaussian, and logist regression
DT = tree.DecisionTreeClassifier(criterion = 'entropy')
NB = GaussianNB()
LR = LogisticRegression(solver='liblinear', random_state=0)

base_clfs = [DT, NB, LR]

n_estim = np.arange(1, 16)

acc_dfs = []
for clf in base_clfs:
	n_acc05 = []
	n_acc1 = []
	for n in n_estim:
		model05 = AdaBoostClassifier(n_estimators=n, base_estimator=clf, learning_rate=0.5)
		model1 = AdaBoostClassifier(n_estimators=n, base_estimator=clf, learning_rate=1)
		model05.fit(X_train , Y_train.ravel())
		model1.fit(X_train , Y_train.ravel())
		n_acc05.append([n, model05.score(X_test, Y_test), "0.5" + str(clf)])
		n_acc1.append([n, model1.score(X_test, Y_test), "1" + str(clf)])
	temp_df05 = pd.DataFrame(n_acc05, columns=['n', 'Accuracy', 'rate_type'])
	temp_df1 = pd.DataFrame(n_acc1, columns=['n', 'Accuracy', 'rate_type'])
	acc_dfs.append(temp_df05)
	acc_dfs.append(temp_df1)

fig, ax = plt.subplots()
ax.plot(acc_dfs[0]['n'], acc_dfs[0]['Accuracy'], label="DT, rate = 0.5")
ax.plot(acc_dfs[1]['n'], acc_dfs[1]['Accuracy'], label="DT, rate = 1")
ax.plot(acc_dfs[2]['n'], acc_dfs[2]['Accuracy'], label="NB, rate = 0.5")
ax.plot(acc_dfs[3]['n'], acc_dfs[3]['Accuracy'], label="NB, rate = 1")
ax.plot(acc_dfs[4]['n'], acc_dfs[4]['Accuracy'], label="LR, rate = 0.5")
ax.plot(acc_dfs[5]['n'], acc_dfs[5]['Accuracy'], label="LR, rate = 1")
ax.set(xlabel='n Value', ylabel='Accuracy', title='n Accuracy by Value')
fig.legend(loc="center right")
ax.grid()
print("Saving Q1 graph...")
fig.savefig("Q1_nAccuracy_Graph.png")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
best_dt = acc_dfs[0].nlargest(1, 'Accuracy')



