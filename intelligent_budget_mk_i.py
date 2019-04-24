import numpy as np
import scipy
import sklearn as sk
import matplotlib.pyplot as plt
import math
from math import pi
from statistics import *
from scipy.stats import *
import statsmodels.api as sm
from decimal import Decimal
import seaborn as sns

from googlesearch import search

import googleplaces
from googleplaces import GooglePlaces, types, lang
from geolocation.main import GoogleMaps

import geocoder

import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

import gspread
from gspread import *
import pandas as pd

from oauth2client.service_account import ServiceAccountCredentials

import time
from datetime import *

import random

from io import StringIO
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score

import os
import re
import sys
import timeit
import string
from datetime import datetime
from time import *
from dateutil.parser import parse

# Update to store keys in drive

def perform_google_search(s):
    for search_result in search(s, tld="com", num=10, stop=5, pause=2):
        print(search_result)

def get_and_load_api_keys():
    google_places = GooglePlaces("api_key_data_frame['GooglePlaces'][1]")
    google_maps = GoogleMaps("api_key_data_frame['GooglePlaces'][1]")

    return google_places, google_maps,

def get_current_location():
    current_location = geocoder.ip('me')
    latitude = current_location.latlng[0]
    longitude = current_location.latlng[1]
    return latitude, longitude

def check_data_frame_data_type():
    print('Expense Categories: \n')
    print(expense_categories_data_frame.columns.to_series().groupby(expense_categories_data_frame.dtypes).groups)
    print('Work Caregories: \n')
    print(work_hours_categories_data_frame.columns.to_series().groupby(work_hours_categories_data_frame.dtypes).groups)


def load_expense_data():
    expense_categories_data_frame_source = pd.DataFrame(expense_categories.get_all_records())
    expense_categories_data_frame = expense_categories_data_frame_source.copy()
    expense_categories_data_frame.columns
    return expense_categories_data_frame.head()

def check_expense_data_for_null():
    expense_categories_data_frame_source = pd.DataFrame(expense_categories.get_all_records())
    expense_categories_data_frame = expense_categories_data_frame_source.copy()
    expense_categories_data_frame.columns
    return expense_categories_data_frame.info()

def load_work_hours_data():
    work_hours_categories_data_frame_source = pd.DataFrame(work_hours_categories.get_all_records())
    work_hours_categories_data_frame = work_hours_categories_data_frame_source.copy()
    work_hours_categories_data_frame.columns
    return work_hours_categories_data_frame.head()

def check_work_hours_data_for_null():
    work_hours_categories_data_frame_source = pd.DataFrame(work_hours_categories.get_all_records())
    work_hours_categories_data_frame = work_hours_categories_data_frame_source.copy()
    work_hours_categories_data_frame.columns
    return work_hours_categories_data_frame.info()

def load_all_data():
    return load_expense_data(), load_work_hours_data(), get_and_load_api_keys()

def awaiting_next_pay_period(pay_schedule):
    # sleep(seconds per minute * minutes per hour * hours per day * days per week * weeks per period)
    if pay_schedule == 1:
        time.sleep(60*60*24*7*1)
        wage.append(hours*rates)

    elif pay_schedule == 2:
        time.sleep(60*60*24*7*2)
        wage.append(hours*rates)

    elif pay_schedule == 3:
        time.sleep(60*60*24*7*3)
        wage.append(hours*rates)

    elif pay_schedule == 4:
        time.sleep(60*60*24*7*4)
        wage.append(hours*rates)

def remaining_income():
    for i in purchase_amounts:
        remaining_income = actual_net_salary - purchase_amounts
        return remaining_income

def regression_plot(x, y):
    plt.plot(x, y.T, 'rx')
    plt.plot(x, y.T)

    i = range(0, len(expense_categories_data_frame['purchase_amount'])-expense_categories_data_frame['purchase_amount'].value_counts()[0]+1)

    m, c = np.polyfit(x, y.T, 1)

    plt.plot(m*i + c)

    plt.savefig('regression.png')

    plt.xlabel('purchase number')
    plt.ylabel('amount spent per day')
    plt.show()

def purchase_analysis():
    #In days or number of data points
    regression_plot(range(0, len(expense_categories_data_frame['purchase_amount'])-expense_categories_data_frame['purchase_amount'].value_counts()[0]+1),
                    purchase_amounts[0:len(expense_categories_data_frame['purchase_amount'])-expense_categories_data_frame['purchase_amount'].value_counts()[0]+1])

def purchase_analytics_text():
    print("With mean: {:0.4f} and standard deviation: {:0.4f} ".format(mean(purchase_amounts), stdev(purchase_amounts)))
    print("With this trend, your expenses could possibly increase to {:0.2f} at your next purchase.".format(mean(purchase_amounts)+stdev(purchase_amounts)))

def determine_expense_category_value_counts_and_ranges(category):
    print(expense_categories_data_frame[category].value_counts())
    print('With overall size of: ', len(expense_categories_data_frame[category]))
    print('So end the range at, ', len(expense_categories_data_frame[category])-expense_categories_data_frame[category].value_counts()[0])

def visits_by_merchant_category():
    merchant_category_frame = expense_categories_data_frame['merchant_category'][:len(expense_categories_data_frame['merchant_category'])-expense_categories_data_frame['merchant_category'].value_counts()[0]
].value_counts()
    merchant_category_frame.plot(kind='barh', title='Number of Occurences by Merchant Type')

    plt.savefig('Number of Occurences by Merchant Type.png')

def visits_by_merchant_name():
    merchant_name_category_frame = expense_categories_data_frame['merchant_with_highest_amount'][:len(expense_categories_data_frame['merchant_with_highest_amount'])-expense_categories_data_frame['merchant_with_highest_amount'].value_counts()[0]
].value_counts()
    merchant_name_category_frame.plot(kind='barh', title='Number of Occurences by Merchant')

    plt.savefig('Number of Occurences by Merchant.png')

def purchases_by_category():
    purchase_category_frame = expense_categories_data_frame['purchase_category'][:len(expense_categories_data_frame['purchase_category'])-expense_categories_data_frame['purchase_category'].value_counts()[0]
].value_counts()
    purchase_category_frame.plot(kind='barh', title='Number of Occurences by Purchase Type')

    plt.savefig('Number of Occurences by Purchase Type.png')

def purchases_by_distance():
    pass

def find_nearby_places():
    query_results = google_places.nearby_search(
            lat_lng={'lat': 30.184624, 'lng': -81.552726},
            radius=5000)

    for place in query_results.places:
         place.get_details()
         print('%s %s' % (place.name, place.types))

def places_near_me():
    get_current_location()
    query_results = google_places.nearby_search(
            lat_lng={'lat': get_current_location()[0], 'lng': get_current_location()[1]},
            radius=5000)

    for place in query_results.places:
         place.get_details()
         print('%s %s' % (place.name, place.types))

def find_nearby_restaurants():
    query_results = google_places.nearby_search(
            lat_lng={'lat': 30.184624, 'lng': -81.552726},
            radius=3200, types=[types.TYPE_RESTAURANT])

    for place in query_results.places:
         place.get_details()
         print('%s' % (place.name))

def restaurants_near_me():
    get_current_location()
    query_results = google_places.nearby_search(
            lat_lng={'lat': get_current_location()[0], 'lng': get_current_location()[1]},
            radius=3400, types=[types.TYPE_RESTAURANT])

    for place in query_results.places:
         place.get_details()
         print('%s' % (place.name))

def find_nearby_cafes():
    query_results = google_places.nearby_search(
            lat_lng={'lat': 30.184624, 'lng': -81.552726},
            radius=3200, types=[types.TYPE_CAFE])

    for place in query_results.places:
         place.get_details()
         print('%s' % (place.name))

def cafes_near_me():
    get_current_location()
    query_results = google_places.nearby_search(
            lat_lng={'lat': get_current_location()[0], 'lng': get_current_location()[1]},
            radius=3400, types=[types.TYPE_CAFE])

    for place in query_results.places:
         place.get_details()
         print('%s' % (place.name))

def find_nearby_bars():
    query_results = google_places.nearby_search(
            lat_lng={'lat': 30.184624, 'lng': -81.552726},
            radius=3200, types=[types.TYPE_BAR])

    for place in query_results.places:
         place.get_details()
         print('%s' % (place.name))

def bars_near_me():
    get_current_location()
    query_results = google_places.nearby_search(
            lat_lng={'lat': get_current_location()[0], 'lng': get_current_location()[1]},
            radius=3400, types=[types.TYPE_BAR])

    for place in query_results.places:
         place.get_details()
         print('%s' % (place.name))

def view_test_budget(person, data):
    
    features = ["Rent", "Water", "Power", "Entertainment", "Health Care", "Insurance: Car", "Insurance: Housing", "Else"]
    features_of_interest = len(features)

    data += data [:1]

    angles = [n / (features_of_interest) * 2 * pi for n in range(features_of_interest)]
    angles += angles [:1]

    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], features)
    ax.plot(angles, data)
    ax.fill(angles, data, 'red', alpha=0.1)

    ax.set_title(person)
    plt.show()

def determine_work_category_value_counts_and_ranges(category):
    print(work_hours_categories_data_frame[category].value_counts())
    print('With overall size of: ', len(work_hours_categories_data_frame[category]))
    print('So end the range at, ', len(work_hours_categories_data_frame[category])-work_hours_categories_data_frame[category].value_counts()[0])

def view_annual_budget():

    grouped_purchases_source = expense_categories_data_frame.groupby(['purchase_category']).agg({'purchase_amount': 'sum'}).reset_index()
    features = grouped_purchases_source['purchase_category'][1:]

    features_of_interest = len(features)

    data = grouped_purchases_source['purchase_amount']
#    data += data[:1]

    angles = [n / (features_of_interest) * 2 * pi for n in range(features_of_interest)]
    angles += angles [:1]

    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], features)
    ax.plot(angles, data)
    ax.fill(angles, data, 'blue', alpha=0.3)

    ax.set_title("Your Annual Budget")

    plt.savefig("Your Annual Budget.png")
    plt.show()

def view_merchant_spending():

    grouped_merchant_source = expense_categories_data_frame.groupby(['merchant_with_highest_amount']).agg({'purchase_amount': 'sum'}).reset_index()[1:]

    features = grouped_merchant_source['merchant_with_highest_amount'][1:]
    features_of_interest = len(features)

    data = grouped_merchant_source['purchase_amount']

    angles = [n / (features_of_interest) * 2 * pi for n in range(features_of_interest)]
    angles += angles [:1]

    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], features)
    ax.plot(angles, data)
    ax.fill(angles, data, 'blue', alpha=0.3)

    ax.set_title("Your Most Visited Merchants")
    plt.savefig("Your Most Visited Merchants.png")

    plt.show()

# Begin credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

pi_credentials = ServiceAccountCredentials.from_json_keyfile_name('json/the intelligent budget-7edf2de93cb8.json', scope)

sojourner_credentials = gspread.authorize(pi_credentials)

expense_related_worksheet = sojourner_credentials.open_by_key('1tKPle0EOUtjTcFtqLqHXcM_iPxlf3MV4RYHfi59d8k0')

work_hours_related_worksheet = sojourner_credentials.open_by_key('1RdACxeor-Y4NZlmiAU1eQopvcU_I2J54KOpc2AWYfU8')

key = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
api_keys = ServiceAccountCredentials.from_json_keyfile_name('json/the intelligent budget-583b0c86574c.json', key)
api_credentials = gspread.authorize(api_keys)
key_relations = api_credentials.open_by_key('1VuylGh-QIee1dHsWnaDoq_mVMmEtP76KZbV59qe3PCE')

# End credentials

# Begin loading data

expense_categories = expense_related_worksheet.sheet1

work_hours_categories = work_hours_related_worksheet.sheet1

keys = key_relations.sheet1

api_key_data_frame_source = pd.DataFrame(keys.get_all_records())

api_key_data_frame = api_key_data_frame_source.copy()

expense_categories_data_frame_source = pd.DataFrame(expense_categories.get_all_records())
expense_categories_data_frame = expense_categories_data_frame_source.copy()

work_hours_categories_data_frame_source = pd.DataFrame(work_hours_categories.get_all_records())
work_hours_categories_data_frame = work_hours_categories_data_frame_source.copy()

get_and_load_api_keys()
load_all_data()

actual_net_salary = work_hours_categories_data_frame['actual_net_annual_salary'].apply(pd.to_numeric)
purchase_amounts = expense_categories_data_frame['purchase_amount'].apply(pd.to_numeric, errors='coerce').fillna(0)
distances_from_home = expense_categories_data_frame['distance_from_home_vector'].apply(pd.to_numeric, errors='coerce').fillna(0)
google_places = GooglePlaces(api_key_data_frame['GooglePlaces'][1])

# End loading data

purchase_analysis()

purchase_analytics_text()

purchases_by_distance()

visits_by_merchant_category()

purchases_by_category()

visits_by_merchant_name()

restaurants_near_me()
cafes_near_me()
bars_near_me()

view_annual_budget()

view_merchant_spending()




###
# Plot Data
# def budget_image():
#     features = list(data)
#     number_of_features = len(features)

#     values_for_person_one = data.iloc[0].tolist()
#     values_for_person_one += values_for_person_one[:1]

#     values_for_person_two = data.iloc[1].tolist()
#     values_for_person_two += values_for_person_two[:1]

#     angles = [n / float(number_of_features) * 2 * pi for n in range(number_of_features)]
#     angles += angles[:1]

#     ax = plt.subplot(111, polar = True)

#     plt.xticks(angles[:-1], features)

#     ax.plot(angles, values_for_person_one)
#     ax.plot(angles, values_for_person_two)

#     ax.fill(angles, values_for_person_one, 'blue', alpha = 0.1)
#     axis.set_title("Person 1")
#     plt.show()

#     ax.fill(angles, values_for_person_two, 'red', alpha = 0.1)
#     axis.set_title("Person 2")
#     plt.show()

# # Begin back propagation...or not since it's just a more complicated version of gradient descent

# sigma = lambda z : 1 / (1 + np.exp(-z))
# d_sigma = lambda z : np.cosh(z/2)**(-2) / 4

# #initialize network structure and clear past trainings
# def reset_network (n1 = 6, n2 = 7, random=np.random) :
#     global W1, W2, W3, b1, b2, b3
#     W1 = random.randn(n1, 1) / 2
#     W2 = random.randn(n2, n1) / 2
#     W3 = random.randn(2, n2) / 2
#     b1 = random.randn(n1, 1) / 2
#     b2 = random.randn(n2, 1) / 2
#     b3 = random.randn(2, 1) / 2

# #generate the network
# def network_function(a0):
#     z1 = W1 @ a0 + b1
#     a1 = sigma(z1)
#     z2 = W2 @ a1 + b2
#     a2 = sigma(z2)
#     z3 = W3 @ a2 + b3
#     a3 = sigma(z3)
#     return a0, z1, a1, z2, a2, z3, a3

# #Cost function
# def cost(x, y) :
#     return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size

# #first node
# def J_W3 (x, y) :
#     a0, z1, a1, z2, a2, z3, a3 = network_function(x)
#     J = 2 * (a3 - y)
#     J = J * d_sigma(z3)
#     J = J @ a2.T / x.size
#     return J

# def J_b3 (x, y) :
#     a0, z1, a1, z2, a2, z3, a3 = network_function(x)
#     J = 2 * (a3 - y)
#     J = J * d_sigma(z3)
#     J = np.sum(J, axis=1, keepdims=True) / x.size
#     return J

# #second node
# def J_W2 (x, y) :
#     a0, z1, a1, z2, a2, z3, a3 = network_function(x)
#     J = 2 * (a3 - y)
#     J = J * d_sigma(z3)
#     J = (J.T @ W3).T
#     J = J * d_sigma(z2)
#     J = J @ a1.T / x.size
#     return J

# def J_b2 (x, y) :
#     a0, z1, a1, z2, a2, z3, a3 = network_function(x)
#     J = 2 * (a3 - y)
#     J = J * d_sigma(z3)
#     J = (J.T @ W3).T
#     J = J * d_sigma(z2)
#     J = np.sum(J, axis=1, keepdims=True) / x.size
#     return J

# #third node
# def J_W1 (x, y) :
#     a0, z1, a1, z2, a2, z3, a3 = network_function(x)
#     J = 2 * (a3 - y)
#     J = J * d_sigma(z3)
#     J = (J.T @ W3).T
#     J = J * d_sigma(z2)
#     J = (J.T @ W2).T
#     J = J * d_sigma(z1)
#     J = J @ a0.T / x.size
#     return J

# def J_b1 (x, y) :
#     a0, z1, a1, z2, a2, z3, a3 = network_function(x)
#     J = 2 * (a3 - y)
#     J = J * d_sigma(z3)
#     J = (J.T @ W3).T
#     J = J * d_sigma(z2)
#     J = (J.T @ W2).T
#     J = J * d_sigma(z1)
#     J = np.sum(J, axis=1, keepdims=True) / x.size
#     return J
# # %% markdown
# # Begin machine learning prediction MK II: SGD Classification
# # %%
# from sklearn.linear_model import SGDClassifier

# # classifier = SGDClassifier(loss="L2", penalty="none", max_iter=10)
# #L2 norm not supported
# # %% markdown
# # ###### Begin prediction MK I: gradient descent
# # %% markdown
# # Notes:
# #     Python doesn't immediately intepret the full set of dimensions for a multidimensional array if one of the the dimentions is 1.
# #      Culprit 1: So must cast as np.matrix to compensate.
# #      Culprit 2: Not able to handle very large numbers. So we can import Decimal.
# #      Culprit 3: Possibly using np.power() might bypass computational error
# #      Culprit 4: np.square also induces computational error
# #      Culprit 5: Cross products don't work since they will result in an orthogonal vector
# #      Culprit 6: np.vectorize ????
# #      Culprit 7: reduce(fxn, array)
# #      Culprit 8: Fuck it. Run without any instances of np.matrix.
# # %%
# x = np.matrix('1, 2; 3, 4; 5, 0')
# squarer = lambda t: t ** 2
# #vfunc = np.vectorize(squarer)
# #vfunc(x)
# # %%
# from functools import reduce
# def my_reduce(x, y):
#     length=len(x[0])-1
#     newY = y**2
#     # print('x',
#     # x,
#     # 'beg len',
#     # len(x[0]),
#     # 'end len',
#     # x[0][len(x[0])-1]
#     # ,'y', y)


#     # 1) component wise squaring
#     x[0].append(newY)
#     # 2) dot product for each value.
#     x[1] = x[0][length]+newY
#     # 3) delta of component wise by dot product.
#     x[2] = (x[1] - x[0][length+1])

#     return x

# print(reduce(my_reduce, [1, 2, 3], [[0],0,0]))
# # %%
# from decimal import Decimal
# from functools import reduce

# # def squared(x):
# #     return x**2

# def gradientDescent(x, y, theta, alpha, N, iterations):
#     for iter in range(iterations):
#         x_transpose = x.transpose()
#         hypothesis = np.dot(x, theta)
#         squared_errors = (hypothesis - y) **2
#         cost = squared_errors / (2 * N)
# #        print("At iteration {}, the cost is {}".format(iter, cost))

#         gradient = np.dot(x, squared_errors) / N
#         # update theta
#         theta = theta - alpha * gradient
#     return theta

# # print(type(i))
# # print(type(bills_01))

# # print((np.dot(i, theta) - bills_01).shape)
# # print((bills_01).shape)
# # print(np.matrix(i))


# N = j
# #temp_N = len(temp_i)
# # iterations = 10000
# # alpha = 0.005

# theta = np.ones(N)
# # temp_theta = np.ones(temp_N)
# theta = gradientDescent(i, bills_01, theta, alpha, j, iterations)