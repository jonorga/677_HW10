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
best_nb = acc_dfs[2].nlargest(1, 'Accuracy')
best_lr = acc_dfs[4].nlargest(1, 'Accuracy')
print("For learning rate 0.5...")
print("Decision tree n* =", best_dt['n'].iloc[0])
print("Naive bayesian n* =", best_nb['n'].iloc[0])
print("Logistic regression n* =", best_lr['n'].iloc[0])


print("\n")
# Question 3 ========================================================================================
print("Question 3:")
print("For learning rate 0.5...")
print("Decision tree accuracy =", best_dt['Accuracy'].iloc[0])
print("Naive bayesian accuracy =", best_nb['Accuracy'].iloc[0])
print("Logistic regression accuracy =", best_lr['Accuracy'].iloc[0])


print("\n")
# Question 4 ========================================================================================
print("Question 4:")

def Ada(df, n, base_est):
	ada = AdaBoostClassifier(n_estimators=n, base_estimator=base_est, learning_rate=0.5)
	X_train = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
	Y_train = df[['Color']][df['Week'] <= 50].values
	X_test = df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values
	actual = df[['Color']][(df['Week'] > 50) & (df['Week'] <= 100)].values
	actual = actual.reshape(1, -1)[0]

	ada = ada.fit(X_train, Y_train.ravel())
	predictions = ada.predict(X_test)
	i = 0
	balance = 100
	file_len = len(actual)
	while i < file_len:
		today_stock = balance / df['Close'].iloc[i + 50]
		tmr_stock = balance / df['Close'].iloc[i + 51]
		difference = abs(today_stock - tmr_stock)
		if actual[i] == predictions[i]:
			balance += difference * df["Close"].iloc[i + 51]
		else:
			balance -= difference * df["Close"].iloc[i + 51]
		i += 1
	return round(balance, 2)

dt_score = best_dt['Accuracy'].iloc[0]
nb_score = best_nb['Accuracy'].iloc[0]
lr_score = best_lr['Accuracy'].iloc[0]

if dt_score > nb_score and dt_score > lr_score:
	print("The decision tree classifier is the strongest base estimator for my data")
	cmg_ada_balance = Ada(df_cmg, best_dt['n'].iloc[0], DT)
	spy_ada_balance = Ada(df_spy, best_dt['n'].iloc[0], DT)
elif nb_score > dt_score and nb_score > lr_score:
	print("The naive bayesian classifier is the strongest base estimator for my data")
	cmg_ada_balance = Ada(df_cmg, best_nb['n'].iloc[0], NB)
	spy_ada_balance = Ada(df_spy, best_nb['n'].iloc[0], NB)
else:
	print("The logistic regression classifier is the strongest base estimator for my data")
	cmg_ada_balance = Ada(df_cmg, best_lr['n'].iloc[0], LR)
	spy_ada_balance = Ada(df_spy, best_lr['n'].iloc[0], LR)


print("\n")
# Question 5 ========================================================================================
print("Question 5:")

def BNH(df):
	y2_start = df['Close'].iloc[50]
	y2_end = df['Close'].iloc[100]
	stock = 100 / y2_start
	return round(stock * y2_end, 2)

cmg_bnh_balance = BNH(df_cmg)
spy_bnh_balance = BNH(df_spy)

if cmg_ada_balance > cmg_bnh_balance:
	cmg_result = "more"
else:
	cmg_result = "less"
if spy_ada_balance > spy_bnh_balance:
	spy_result = "more"
else:
	spy_result = "less"

print("For year 2 of the Chipotle stock, Adaboost ($" + str(cmg_ada_balance) + ") was " 
	+ cmg_result + " effective than buy-and-hold ($" + str(cmg_bnh_balance) + ")")

print("For year 2 of the S&P-500 stock, Adaboost ($" + str(spy_ada_balance) + ") was " 
	+ spy_result + " effective than buy-and-hold ($" + str(spy_bnh_balance) + ")")






