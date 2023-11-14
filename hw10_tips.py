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

Q2("Sun")
Q2("Sat")
Q2("Thur")
Q2("Fri")