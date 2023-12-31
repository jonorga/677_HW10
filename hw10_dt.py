###
### CS667 Data Science with Python, Homework 10, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn import tree

df_cmg = pd.read_csv("cmg_weeks.csv")
df_spy = pd.read_csv("spy_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")


X_train_c = df_cmg[['Avg_Return', 'Volatility']][df_cmg['Week'] <= 50].values
Y_train_c = df_cmg[['Color']][df_cmg['Week'] <= 50].values
X_test_c = df_cmg[['Avg_Return', 'Volatility']][(df_cmg['Week'] > 50) & (df_cmg['Week'] <= 100)].values
Y_test_c = df_cmg[['Color']][(df_cmg['Week'] > 50) & (df_cmg['Week'] <= 100)].values

clf_c = tree.DecisionTreeClassifier(criterion = 'entropy')
clf_c = clf_c.fit(X_train_c, Y_train_c)

year2_cmg_score = clf_c.score(X_test_c, Y_test_c)
print("Decision tree year 2 score:", year2_cmg_score)


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
print("Decision tree year 2 confusion matrix:")

y2_cmg_prediction = clf_c.predict(X_test_c)


print(Y_test_c)
print(y2_cmg_prediction)
y_actu = pd.Series(Y_test_c.reshape(1, -1)[0], name='Actual')
y_pred = pd.Series(y2_cmg_prediction, name='Predicted')

cm = pd.crosstab(y_actu, y_pred)
print(cm)


print("\n")
# Question 3 ========================================================================================
print("Question 3:")

TP = cm['Green'].iloc[0]
TN = cm['Red'].iloc[1]
FP = cm['Green'].iloc[1]
FN = cm['Red'].iloc[0]

TPR = round((TP / (TP + FN) * 100), 2)
TNR = round((TN / (TN + FP) * 100), 2)
print("Decision tree year 2 true positive rate: " + str(TPR) + "%")
print("Decision tree year 2 true negative rate: " + str(TNR) + "%")


print("\n")
# Question 4 ========================================================================================
print("Question 4:")

def BNH(df):
	y2_start = df['Close'].iloc[50]
	y2_end = df['Close'].iloc[100]
	stock = 100 / y2_start
	return round(stock * y2_end, 2)


def DT(df):
	clf = tree.DecisionTreeClassifier(criterion = 'entropy')
	X_train = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
	Y_train = df[['Color']][df['Week'] <= 50].values
	X_test = df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values
	actual = df[['Color']][(df['Week'] > 50) & (df['Week'] <= 100)].values
	actual = actual.reshape(1, -1)[0]

	clf = clf.fit(X_train, Y_train)
	predictions = clf.predict(X_test)
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


cmg_bnh_bal = BNH(df_cmg)
spy_bnh_bal = BNH(df_spy)
cmg_dt_bal = DT(df_cmg)
spy_dt_bal = DT(df_spy)

if cmg_bnh_bal > cmg_dt_bal:
	cmg_result = "better"
else:
	cmg_result = "worse"
if spy_bnh_bal > spy_dt_bal:
	spy_result = "better"
else:
	spy_result = "worse"

print("For year 2 of the Chipotle stock, buy-and-hold ($" + str(cmg_bnh_bal) + ") was " 
	+ cmg_result + " than the decision tree classifier strategy ($" + str(cmg_dt_bal) + ")")
print("For year 2 of the S&P-500 stock, buy-and-hold ($" + str(spy_bnh_bal) + ") was " 
	+ spy_result + " than the decision tree classifier strategy ($" + str(spy_dt_bal) + ")")





