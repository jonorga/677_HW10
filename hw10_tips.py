###
### CS667 Data Science with Python, Homework 10, Jon Organ
###

import pandas as pd
import numpy as np

df = pd.read_csv("tips.csv")

# Question 1 ========================================================================================
print("Question 1:")

lunch_bill_avg = df['total_bill'][df['time'] == "Lunch"].sum() / df['total_bill'][df['time'] == "Lunch"].count()
lunch_tip_avg = df['tip'][df['time'] == "Lunch"].sum() / df['tip'][df['time'] == "Lunch"].count()
lunch_avg_tip = lunch_tip_avg / lunch_bill_avg

dinner_bill_avg = df['total_bill'][df['time'] == "Dinner"].sum() / df['total_bill'][df['time'] == "Dinner"].count()
dinner_tip_avg = df['tip'][df['time'] == "Dinner"].sum() / df['tip'][df['time'] == "Dinner"].count()
dinner_avg_tip = dinner_tip_avg / dinner_bill_avg

print("Average lunch tip as a percentage of meal cost: " + str(round(lunch_avg_tip * 100, 2)) + "%")
print("Average dinner tip as a percentage of meal cost: " + str(round(dinner_avg_tip * 100, 2)) + "%")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")
# Sun, Sat, Thur, Fri

def Q2(day):
	day_bill_avg = df['total_bill'][df['day'] == day].sum() / df['total_bill'][df['day'] == day].count()
	day_tip_avg = df['tip'][df['day'] == day].sum() / df['tip'][df['day'] == day].count()
	day_avg_tip = day_tip_avg / day_bill_avg
	print("Average " + day + " tip as a percentage of meal cost: " + str(round(day_avg_tip * 100, 2)) + "%")

days = ["Thur", "Fri", "Sat", "Sun"]
for day in days:
	Q2(day)


print("\n")
# Question 3 ========================================================================================
print("Question 3:")
def AddTipPercs(row):
	return row['tip'] / row['total_bill']

df['tip_perc'] = df.apply(AddTipPercs, axis=1)

highest_num = 0
for day in days:
	cur_mean = df['tip_perc'][(df['time'] == "Lunch") & (df['day'] == day)].mean()
	if cur_mean > highest_num:
		highest_num = cur_mean
		highest_name = ["lunch", day]
	cur_mean = df['tip_perc'][(df['time'] == "Dinner") & (df['day'] == day)].mean()
	if cur_mean > highest_num:
		highest_num = cur_mean
		highest_name = ["dinner", day]

print("Highest tips: \nTime: " + highest_name[0] + "\nDay: " + highest_name[1] + "\nAverage tip: "
	+ str(round(highest_num * 100, 2)) + "%")



print("\n")
# Question 4 ========================================================================================
print("Question 4:")
df_corr = df.corr()
print("Correlation between tips and meal prices: " + str(round(df_corr['tip'].iloc[0], 5)) )


print("\n")
# Question 5 ========================================================================================
print("Question 5:")
print("The correlation between tips and size of group is " + str(round(df_corr['size'].iloc[1], 5))
	+ ". This is slightly less related to one another than tips and meal prices.")


print("\n")
# Question 6 ========================================================================================
print("Question 6:")


