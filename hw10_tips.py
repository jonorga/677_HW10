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
total_smokers = df['size'][df['smoker'] == "Yes"].sum()
total_non_smokers = df['size'][df['smoker'] == "No"].sum()
all_people = total_smokers + total_non_smokers
smokers_perc = str(round((total_smokers / all_people) * 100, 2))
print(smokers_perc + "% of customers are smokers.")


print("\n")
# Question 7 ========================================================================================
print("Question 7:")

file_len = len(df.index)
i = 0
period_end = 0
coefs = []

while i < file_len:
	cur_day = df['day'].iloc[i]
	prev_day = df['day'].iloc[i]
	period_start = i
	while cur_day == prev_day:
		cur_day = df['day'].iloc[i]
		i += 1
	period_end = i - 1

	x = np.arange(1, period_end - period_start + 2)
	y = df['tip_perc'][(df.index >= period_start) & (df.index <= period_end )].to_numpy()
	z = np.polyfit(x, y, 1)
	coefs.append(z[0])

np_coefs = np.array(coefs)
print("After applying linear regression to each day, and taking the average of the slope of each day"
	+ " it would appear that tips are increasing each day with an average slope coefficient of "
	+ str(np_coefs.mean()))


print("\n")
# Question 8 ========================================================================================
print("Question 8:")
df_smokers = df[df['smoker'] == "Yes"]
df_non_smokers = df[df['smoker'] == "No"]
smo_corr = df_smokers.corr()
non_corr = df_non_smokers.corr()



print("Yes, there is a difference in correlation between smokers and non-smokers.")
print("Smokers:")
print("Correlation between tip and total bill : " + str(smo_corr['tip'].iloc[0]))
print("Correlation between tip and size : " + str(smo_corr['tip'].iloc[2]))
print("Non-smokers:")
print("Correlation between tip and total bill : " + str(non_corr['tip'].iloc[0]))
print("Correlation between tip and size : " + str(non_corr['tip'].iloc[2]))



