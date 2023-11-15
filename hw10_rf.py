###
### CS667 Data Science with Python, Homework 10, Jon Organ
###

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt

df_cmg = pd.read_csv("cmg_weeks.csv")
df_spy = pd.read_csv("spy_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")

X_train = df_cmg[['Avg_Return', 'Volatility']][df_cmg['Week'] <= 50].values
Y_train = df_cmg[['Color']][df_cmg['Week'] <= 50].values
X_test = df_cmg[['Avg_Return', 'Volatility']][(df_cmg['Week'] > 50) & (df_cmg['Week'] <= 100)].values
Y_test = df_cmg[['Color']][(df_cmg['Week'] > 50) & (df_cmg['Week'] <= 100)].values


N_vals = [1, 3, 5, 7, 9]
d_vals = [1, 2, 3, 4, 5]
rfc_scores = []


for N in N_vals:
	line = []
	for d in d_vals:
		model = RFC(n_estimators=N, max_depth=d, criterion='entropy')
		model.fit(X_train, Y_train.ravel())
		y_pred = model.predict(X_test)
		error_rate = zero_one_loss(Y_test, y_pred)
		rfc_scores.append([N, d, error_rate])

plot_df = pd.DataFrame(rfc_scores, columns=['N', 'd', 'error_rate'])
scatter_plot = plt.figure()
ax = scatter_plot.add_subplot(1, 1, 1)

ax.scatter(plot_df["N"], plot_df["d"], s=plot_df['error_rate'] * 400)
ax.set_title("Scatter plot for random forest")
ax.set_xlabel("N value")
ax.set_ylabel("d value")
print("Saving scatter plot...")
scatter_plot.savefig("Q1_scatterplot.png")

best_vals = plot_df.nsmallest(1, 'error_rate')
print("Best N value:", best_vals['N'].iloc[0])
print("Best d value:", best_vals['d'].iloc[0])


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
print("Random forest year 2 confusion matrix:")

model = RFC(n_estimators=best_vals['N'].iloc[0], max_depth=best_vals['d'].iloc[0], criterion='entropy')
model.fit(X_train, Y_train.ravel())
y_prediction = model.predict(X_test)


y_actu = pd.Series(Y_test.reshape(1, -1)[0], name='Actual')
y_pred = pd.Series(y_prediction, name='Predicted')

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

def RF(df):
	clf = RFC(n_estimators=best_vals['N'].iloc[0], max_depth=best_vals['d'].iloc[0], criterion='entropy')
	X_train = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
	Y_train = df[['Color']][df['Week'] <= 50].values
	X_test = df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values
	actual = df[['Color']][(df['Week'] > 50) & (df['Week'] <= 100)].values
	actual = actual.reshape(1, -1)[0]

	clf = clf.fit(X_train, Y_train.ravel())
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
cmg_rf_bal = RF(df_cmg)
spy_rf_bal = RF(df_spy)

if cmg_bnh_bal > cmg_rf_bal:
	cmg_result = "better"
else:
	cmg_result = "worse"
if spy_bnh_bal > spy_rf_bal:
	spy_result = "better"
else:
	spy_result = "worse"

print("For year 2 of the Chipotle stock, buy-and-hold ($" + str(cmg_bnh_bal) + ") was " 
	+ cmg_result + " than the random forest classifier strategy ($" + str(cmg_rf_bal) + ")")
print("For year 2 of the S&P-500 stock, buy-and-hold ($" + str(spy_bnh_bal) + ") was " 
	+ spy_result + " than the random forest classifier strategy ($" + str(spy_rf_bal) + ")")





