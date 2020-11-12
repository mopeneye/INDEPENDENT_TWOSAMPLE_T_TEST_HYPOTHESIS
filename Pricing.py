# PRICING
# A game company gave gift coins to its users for purchasing items in a game.
# Users buy various vehicles for their characters using these virtual coins.
# The game company did not specify a price for an item and provided users to buy this item at the price they wanted.
# For example, for the item named shield, users will buy this shield by paying the amounts they see fit.
# For example, a user can pay with 30 units of virtual money given to him, while the other user can pay with 45 units.
# Therefore, users can buy this item with the amounts they can afford to pay.

# Problems to be solved:
# 1. Does the item's price differ by category? Express it statistically.
# 2. What should the price of the item be depending on the first question? Explain why?
# 3. It is desirable to be "mobile" about the price. Create a decision support system for the price strategy and
# 4. Simulate item purchases and income for possible price changes.

# imports

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pylab
import scipy.stats as stats
from scipy.stats import shapiro
from itertools import combinations
from scipy.stats import mannwhitneyu
import statsmodels.stats.api as sms

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load_Pricing_data():
    data = pd.read_csv(r'E:\PROJECTS\dsmlbc\Pricing\datasets\pricing.csv', sep = ';')

    return data

df = load_Pricing_data()

df_bck = df.copy()

df.head()

# Grouping samples by their category id and calculating their mean and median values
df.groupby('category_id').agg({'price': ['mean', 'median']})

df['category_id'] = df['category_id'].astype(str)

df.info()
# list prices grouped by category_id
x = df.groupby('category_id')['price'].apply(list)

# conversion to series and transpose operation
df = x.apply(pd.Series).T

# Hypothesis
# H0: Statistically, there is no difference among categories payments
# H1: Statistically, there's a significant difference among categories payments

# Assumption1 Control (Normallity Assumption)

# H0: Assumption of Normal distribution is OK
# H1: Assumption of Normal distribution is NOT OK!

# Assumption2 Control (Homogenity of  Assumption)
# H0: Variances are homogenous!
# H1: Variances are not homogenous!


for c in list(combinations(df.columns, 2)):
    # print(col) # all combinations of categories
    statistic, pvalue = shapiro(df[c[0]].dropna())
    statistic, pvalue1 = shapiro(df[c[1]].dropna())

    alpha = 0.05
    print("Categoried {} and {}".format(c[0], c[1]))

    if ((pvalue < 0.05) == False) and ((pvalue1 < 0.05) == False):

        print('Null Hypothesis for Assumption1 could not be rejected!, so distribution is Normal')

             # Variance homogenity test

        test_stats, lev_pvalue = levene(df[c[0]], df[c[1]])

        if (lev_pvalue < 0.05) == False:

            test_stats_main, pvalue_main = ttest_ind(df[c[0]], df[c[1]], equal_var=True)

            if (pvalue_main < 0.05):
                print('Variances are homogenous!')
                print("Test statistics =%.4f, p-value= %.4f" % (test_stats_main, ind_pvalu))
                print('H0 hypothesis has been rejected, so there is a significant difference among categories payments')

            else:
                print('Variances are homogenous!')
                print("Test statistics =%.4f, p-value= %.4f" % (test_stats_main, ind_pvalu))
                print('H0 hypothesis could not be  rejected, so there is not a significant difference among categories payments')

        else:
            test_stats, indv_pvalue = ttest_ind(df[c[0]], df[c[1]], equal_var=False)

            if (vnh_pvalue < 0.05):
                print('Variances are not homogenous!')
                print("Test statistics =%.4f, p-value= %.4f" % (test_stats, vnh_pvalue))
                print('H0 has been rejected. It means statistically there is significant difference among categories payments')
            else:

                print('Variances are not homogenous!')
                print("Test statistics =%.4f, p-value= %.4f" % (test_stats, vnh_pvalue))
                print('H0 could not be rejected, It means statistically there is no  significant difference among categories')

    else:

            stats, pmw_value = mannwhitneyu(df[c[0]].dropna(), df[c[1]].dropna())
            if (pmw_value<0.05):
                print('Normal Distribution error!')
                print("NON-Parametric TEST")
                print("Shapiro p-value = %.4f" % (pvalue))
                print("Mannwhitneyu p-value = %.4f" % (pmw_value))
            else:
                print('Normal Distribution error!')
                print("NON-Parametric TEST")
                print("Shapiro p-value = %.4f" % (pvalue))
                print("Mannwhitneyu p-value = %.4f" % (pmw_value))
                print(" H0 could not be rejected, It means statistically there is not a significant difference between {} and {} group by the means of price".format(c[0], c[1]))

# Q2:
# It seems that there are statistically differences among some categories
# and there is not a difference between some categories, so it is up to us to choose median price
prices_mean = df.mean()[:].tolist() # median values of categories
prices_median = df.median()[:].tolist()

print('We may choose ', np.mean(prices_median), ' for all categories')

# Q3:
# Confidence interval for man values
interval1, interval2 = sms.DescrStatsW(prices_median).tconfint_mean()
interval1 = round(interval1)
interval2 = round(interval2)
print( "Confidence interval is from {}  to {}".format(interval1, interval2))


# Q4:
numbers= ([interval1, interval2])
mean= round(np.mean(numbers))
price = input("Please select a price {}, {}, or {} ". format(interval1, mean, interval2))

total_movement = 0

for i in df.columns:
    total_movement += df[i].count()

if int(price) == interval1:
    revenue = total_movement * interval1
    print( "If the price confidence interval is lower, our total income becomes %.2f "% revenue)
elif int(price) == mean:
    revenue = total_movement * mean
    print("If the price  is mean of lower and upper stage of confidence level, our total income becomes %.2f " % revenue)
else:
    revenue = total_movement * interval2
    print("If the price confidence interval is upper, our total income becomes %.2f " % revenue)
