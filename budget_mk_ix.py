import numpy as np
import scipy
import pandas as pd
import sklearn as sk

import matplotlib as plt

# Set some constant values that will never change

avg_work_hrs_yr = 40 *	52
#TODO set function for hours in order to account for non-standard work weeks


# set initial hours
# needs to be based on time
hours = pd.readcsv('work_hours.csv')
hr_per_week = pd.readcsv('work_week.csv')

# set hourly rate
rates = pd.readcsv('hourly_rate.csv')

deduction = pd.readcsv('deductions.csv')

# deductions block from other budget file
# DEDUCTION CALCULATION LEADING TO USEABLE INCOME
'''
def deduction_calc():
    if 0.00 < total_gross_income <= 9525.99:
        ded_taxes = (total_gross_income - deductions()) * 0.10
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
    elif 9525.99 < total_gross_income <= 38700.99:
        ded_taxes = 952.50 + (total_gross_income - marital_status() - deductions()) * 0.12
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
    elif 38700.99 < total_gross_income <= 82500.99:
        ded_taxes = 4453.50 + (total_gross_income - marital_status() - deductions()) * 0.22
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
    elif 82500.99 < total_gross_income < 157500.99:
        ded_taxes = 14089.50 + (total_gross_income - marital_status() - deductions()) * 0.24
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
    elif 157500.99 < total_gross_income < 2000000.99:
        ded_taxes = 32089.50 + (total_gross_income - marital_status() - deductions()) * 0.32
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
    elif 2000000.99 < total_gross_income < 5000000:
        ded_taxes = 45689.50 + (total_gross_income - marital_status() - deductions()) * 0.35
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
    else:
        ded_taxes = 150689.50 + (total_gross_income - marital_status() - deductions()) * 0.37
        ded_taxes = round(ded_taxes, 2)
        return ded_taxes
'''
# compute gross income

wage = (np.sum(hours) +	np.sum(hr_per_week)) @ rates

total_gross_income = np.sum(wage) - 40 @ rates

'''
TODO ASYNCHRONOUS FUNCTION HERE

Essentially, it's to count the number of pay periods in order to compute the future pay periods.

Future pay periods are computed based on current time/hours worked.

'''

#Projected Annual Income
proj_ann_income = avg_work_hrs_yr @ rates

proj_ann_income = proj_ann_income - tax_deduction @ proj_ann_income

# compute net income
net_income_01 = total_gross_income - tax_deduction @ total_gross_income

# compute remaining income
bills_01 = pd.readcsv('bills01.csv')
expenses_01 = np.sum(bills_01)

rem_income = net_income01 - expenses_01

rem_ann_income = proj_ann_income - rem_income

i = np.array([1:len(bills_01)])
m = sum((i - np.average(i)) @ bills_01) / sum((i - np.average(i)) ** 2)
c = np.average(bills_01) - m * np.average(i)

# Average spending per data point in bills
# TODO	convert to average spending per day
avg_per_day = np.average(bills_01, axis = 0)

# General disparity
despair_ity = (proj_ann_income - rem_income) / proj_ann_income

# Plot Data
''''
def budget():
    # Pie chart for the bills
    labels = 'Rent', 'Water', 'Power', 'Retirement', 'Health Care', 'Car Insurance', 'Rent Insurance', 'PAY PERIOD'
    sizes = [rental, wat_bill, pow_bill, ret_ded, health_ded, car_ins, rent_ins, PAY_PERIOD()]
    explode = (0, 0, 0, 0, 0, 0, 0, 0.1)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode = explode, labels = labels, autopct = '%1.2f%%',
            shadow = True, startangle = 90)
    ax1.axis('equal') # aspect ratio to make it a circle

    plt.show()
'''

#TODO	insert GRADIENT DESCENT algo here w.r.t. lowest cost

'''
Determine the capacity for what can be done in the vein of gradient descent on cost here
'''

'''
import urllib2
from bs4 import BeautifulSoup
def cost_scrape() :
	pass
'''