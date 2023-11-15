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




