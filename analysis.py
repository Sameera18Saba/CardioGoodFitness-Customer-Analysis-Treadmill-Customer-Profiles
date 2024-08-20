
Objective
Objective is to perform a Preliminary Data Analysis by exploring cardio fitness data set - "CardioGoodFitness.csv" and perform the below analysis --

Come up with a customer profile (characteristics of a customer) of the different products.
Perform uni-variate and multi-variate analyses.
Generate a set of insights and recommendations that will help the company in targeting new customer.
Data structure
The data is for customers of the treadmill product(s) of a retail store called Cardio Good Fitness. It contains the following variables

Product - the model no. of the treadmill
Age - in no of years, of the customer
Gender - of the customer
Education - in no. of years, of the customer
Marital Status - of the customer
Usage - Avg. # times the customer wants to use the treadmill every week
Fitness - Self rated fitness score of the customer (5 - very fit, 1 - very unfit)
Income - of the customer
Miles- expected to run
Import the necessary libraries - pandas, numpy, seaborn, matplotlib.pyplot
import warnings
warnings.filterwarnings('ignore') #to avoid warnings
# Import necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# To enable plotting graphs in Jupyter notebook
%matplotlib inline
Read the dataset
mydata = pd.read_csv('CardioGoodFitness.csv')
# create a copy of this data to avoid changes in the original data
fit = mydata.copy()
View the dataset
# view first 5 rows of the data - "fit"
fit.head()
Product	Age	Gender	Education	MaritalStatus	Usage	Fitness	Income	Miles
0	TM195	18	Male	14	Single	3	4	29562	112
1	TM195	19	Male	15	Single	2	3	31836	75
2	TM195	19	Female	14	Partnered	4	3	30699	66
3	TM195	19	Male	12	Single	3	3	32973	85
4	TM195	20	Male	13	Partnered	4	2	35247	47
Observations

Variables - 'Product', 'Gender' and 'MaritalStatus' are categorical variables.
Remaining all the variables are numericals.
Check the shape of the dataset
fit.shape
(180, 9)
The dataset has 180 rows and 9 columns.
Check the dataset info
fit.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 180 entries, 0 to 179
Data columns (total 9 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Product        180 non-null    object
 1   Age            180 non-null    int64 
 2   Gender         180 non-null    object
 3   Education      180 non-null    int64 
 4   MaritalStatus  180 non-null    object
 5   Usage          180 non-null    int64 
 6   Fitness        180 non-null    int64 
 7   Income         180 non-null    int64 
 8   Miles          180 non-null    int64 
dtypes: int64(6), object(3)
memory usage: 12.8+ KB
Observations

There are no null values in this dataset
Five-point summary of the dataset
fit.describe(include='all').T
count	unique	top	freq	mean	std	min	25%	50%	75%	max
Product	180	3	TM195	80	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Age	180	NaN	NaN	NaN	28.7889	6.9435	18	24	26	33	50
Gender	180	2	Male	104	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Education	180	NaN	NaN	NaN	15.5722	1.61705	12	14	16	16	21
MaritalStatus	180	2	Partnered	107	NaN	NaN	NaN	NaN	NaN	NaN	NaN
Usage	180	NaN	NaN	NaN	3.45556	1.0848	2	3	3	4	7
Fitness	180	NaN	NaN	NaN	3.31111	0.958869	1	3	3	4	5
Income	180	NaN	NaN	NaN	53719.6	16506.7	29562	44058.8	50596.5	58668	104581
Miles	180	NaN	NaN	NaN	103.194	51.8636	21	66	94	114.75	360
Observations

Product variable has 3, Gender has 2 and MaritalStatus has 2 unique categories.
The most bought product is TM195.
There are more number of Male customers than females.
There are more married customers than single ones.
75% of customers are below 33 years of age
We will further explore in univariate analysis

Exploratory Data Analysis
Univariate analysis
# ******  Referring from Uber case study  *********
# While doing uni-variate analysis of numerical variables we want to study their central tendency 
# and dispersion.
# Let us write a function that will help us create boxplot and histogram for any input numerical 
# variable.
def hist_box(mycol):
    f, (ax_box, ax_hist) = plt.subplots(nrows=2, sharex=True, gridspec_kw = {"height_ratios": (.25, .75)})
    
    sns.boxplot(mycol, ax=ax_box, showmeans=True, color='red')
    sns.distplot(mycol, kde=False, ax=ax_hist, color='blue')
    ax_hist.axvline(np.mean(mycol), color='g', linestyle='--') # Add mean to the histogram
    ax_hist.axvline(np.median(mycol), color='black', linestyle='-') # Add median to the histogram
hist_box(fit.Age)

Observations

The distribution of Age is right skewed.
Majority of customers are aged between 21 to 35.
There are very less outliers.
hist_box(fit.Education)

Observations

CUstomers have an average of 15 years of education.
hist_box(fit.Fitness)

Observations

Average self-rating of customers is 3.3 with approximately 60% of customers.
This implies that customers are decently fit.
hist_box(fit.Income)

Observations

Income is skewed to right.
There are many outliers with higher ioncome range.
Mean is quite close to median, indicating that 50% of customers have income approximately same as the average income - 53K.
hist_box(fit.Miles)

Observations

Miles is also right skewed, indicating some outliers.
50% of cusotmers expect to run approximately 94 miles.
hist_box(fit.Usage)

Observations

Usage is indicating a right skew.
50% of customers plan to use the treadmill 3-4 times a week.
Exploring the categorical variables
# use countplot to plot number of units sold by model of treadmill
sns.countplot(x='Product', data=fit)
plt.show()

Observations

Close to 80 customers have bought TM195, which is approx 44% amking it most preferred product.
# Gender distribution
sns.catplot('Gender', data=fit, kind='count')
plt.show()

Observations

THere are more Male customers than females.
# Marital status distribution
sns.catplot('MaritalStatus', data=fit, kind='count')
plt.show()

Observations

There are more married customers than single ones.
Multi-variate Analysis
Finding the correlation
# slice out the numerical variabales from dataset to check the correlation between the numericals.

fit_num = fit.select_dtypes(include='int64')
fit_num.head()
Age	Education	Usage	Fitness	Income	Miles
0	18	14	3	4	29562	112
1	19	15	2	3	31836	75
2	19	14	4	3	30699	66
3	19	12	3	3	32973	85
4	20	13	4	2	35247	47
corr = fit_num.corr()
corr
Age	Education	Usage	Fitness	Income	Miles
Age	1.000000	0.280496	0.015064	0.061105	0.513414	0.036618
Education	0.280496	1.000000	0.395155	0.410581	0.625827	0.307284
Usage	0.015064	0.395155	1.000000	0.668606	0.519537	0.759130
Fitness	0.061105	0.410581	0.668606	1.000000	0.535005	0.785702
Income	0.513414	0.625827	0.519537	0.535005	1.000000	0.543473
Miles	0.036618	0.307284	0.759130	0.785702	0.543473	1.000000
# plot the heatmap to analyse the correlation between all the numerical variables
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

Observations

Fitness, Miles & Usage have high correlation with each other.
Income has a good correlation with Education.
Income is also correlated with all the other variables.
# Looking at relations between the numerical variables
sns.pairplot(fit)
<seaborn.axisgrid.PairGrid at 0x19a5178ea60>

# Analyze how Gender affects product preference
sns.catplot(x='Product', data=fit, hue='Gender', kind='count')
plt.show()

Observations

TM195 is the most preferable choice for both Male & Female customers.
TM798 is mostly bought by Male customers.
# Analyze how MaritalStatus affects product preference
sns.catplot(x='Product', data=fit, hue='MaritalStatus', kind='count')
plt.show()

Observations

Chances for Partnered people to buy a treadmill is higher than Single people.
# Analyze the age distribution for each product.
sns.boxplot(x='Product', y='Age', data=fit)
plt.show()

Observations

TM798 has a comparatively smaller age range (23-37 years of age) with some outliers (40-47yrs).
TM195 & TM498 have a similar age range of customers.
# Analyse the affect of income range on product preference.
sns.boxplot(x='Product', y='Income', data=fit)
plt.show()

Observations

Customers with high-income range prefer TM798. Customers with low-income prefer either TM195 or TM498.
# Analyse how fitness rating & no:of miles they target to run are affecting their product preference.
sns.swarmplot(x='Fitness', y='Miles', data=fit, hue='Product')
plt.show()

Observations

Customers with higher fitness level expect to run more miles and mostly prefer TM798.
Customers around avegrage fitness rating prefer TM195 & TM498
# Analyse how education & income affect product prefernce of customers.
sns.swarmplot(x='Fitness', y='Usage', data=fit, hue="Product")
plt.show()

Observations

Customers with high fitness ratings use the treadmills more times in a week.
TM798 is preferred by customers who have high fitness ratings and high usage.
TM195 & TM498 users plan to use 2-3 times.
# Analyse income as per the gender and how they affect product preference.
sns.swarmplot(x='Product', y='Income', data=fit, hue='Gender')
plt.show()

Observations

Male cusotmers have comparatively higher income than Females and so are more likely to buy TM798.
sns.swarmplot(x='Education', y="Income", data=fit, hue='Product')
plt.show()
