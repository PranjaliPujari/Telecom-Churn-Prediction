#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ### Business problem overview
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# For many incumbent operators, retaining high profitable customers is the top business goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# 
# In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.

# # Problem Statement
# 
# The telecom company in Southeast Asia is facing an increase in customer churn and wants to reduce the loss of high-value customers. They aim to predict churn in the ninth month using data from the previous three months. The company plans to analyze various factors such as demographics, usage patterns, service quality, and complaints to identify variables that affect churn. They will focus on high-value customers and build machine learning models for churn prediction. These models will help predict churn likelihood and identify important variables to address underlying issues and improve customer satisfaction.
# 
# These models will serve two purposes.
# 
# *   First, they will predict whether a high-value customer is likely to churn in
# the near future. By gaining insights into this aspect, the company can take proactive steps such as offering special plans, discounts, or personalized offers to retain these customers.
# *   Second, the models will identify important variables that strongly predict churn. These variables will shed light on why customers choose to switch to other networks, enabling the company to address underlying issues and improve customer satisfaction.
# 
# 
# 
# 

# # Tasks Involved
# 
# **Task 1: Import libraries and load the dataset**
# 
# **Task 2: Understand and explore the data**
# *   Analyze different feature types in the data
# *   Handle missing values by imputation
# *   Identify the relevant data required for the problem
# 
# **Task 3: Conduct feature engineering**
# *   Extract new relevant features from the data set
# *   Filter high-value customers
# *   Derive the target variable “churn” based on the existing features
# 
# **Task 4: Visualize the data**
# *   Analyze the data to extract relevant insights through informative visualizations
# *   Look for any outliers and treat them
# 
# **Task 5: Modeling**
# *   Divide the data into train-test splits
# *   Handle class imbalance
# *   Build different machine learning models and evaluate their performance
# *   Tune the hyperparameters to optimize the performance for the best model
# *   Train and evaluate a neural network model with the optimal combination of hyperparameters
# 
# **Task 6: Business insights and recommendations**
# *   Understand the profitability of the telecommunication service program, and estimate the impact of your model using misclassification costs
# *   Propose a solution to leverage customer interaction/feedback data and predict those who are highly likely to churn

# In[1]:


# getting starting time of the notebook

import datetime

start_time = datetime.datetime.now()

print("Starting Date and Time:", start_time)


# # Task 1: Importing the required libraries and loading the data set
# 
# **Description**
# 
# 
# In this task, you will load all the methods and packages required to perform the various tasks in this capstone project.

# First, import the required packages and modules.

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 300)


# Mount Google Drive to your VM.

# In[3]:


# Import the required library to mount Google Drive
# from google.colab import drive

# Mount your Google Drive to the Colab notebook
# drive.mount('/content/drive', force_remount=True)


# Import the training data

# In[4]:


# Read the training data from a CSV file stored in your Google Drive
# churn = pd.read_csv('/content/drive/MyDrive/UMD/telecom_churn_data.csv')


# In[5]:


# read the data
churn = pd.read_csv("telecom_churn_data .csv")


# Checklist:
# 
# 
# *   Imported the required packages
# *   Mounted your Google Drive to access the data
# *   Imported the data
# 

# # Task 2: Understanding and exploring the data

# ### Description
# 
# In this task, you will explore the data that you have just loaded.

# In[6]:


# look at initial rows of the data
churn.head()


# ### Data Description
# 
# There are several types of data that is collected from customers by a telecomminucation service provider. Some of the information that you have to look for data analysis and EDA is given below:
# - Recharging of the service: There are several variables that describe the duration, maximum, total amount and average of the recharge price of the service they avail, which include the 2G service, the 3G service, internet packages and call services
#   - av_rech_amt_data: Average recharge data amount
#   - count_rech_2g: Count of 2G recharges by the customer
#   - count_rech_3g: Count of 3G recharges by the customer
#   - max_rech_data: Maximum recharge for mobile internet
#   - total_rech_data: Total recharge for mobile internet
#   - max_rech_amt: Maximum recharge amount
#   - total_rech_amt: Total recharge amount
#   - total_rech_num: Total number of times customer recharged
# 
# - Call and Internet service: They specify the amount of calls, type of calling service used (STD, ISD, Roaming), type of internet service and amount of internet usage over a specific period of time
#   - total_calls_mou: Total minutes of voice calls
#   - total_internet_mb: Total amount of internet usage in MB
#   - arpu: Average revenue per user
#   - onnet_mou: The minutes of usage for all kind of calls within the same operator network
#   - offnet_mou: The minutes of usage for all kind of calls outside the operator T network
#   - Minutes of usage for outgoing calls for each type of call service:
#     - loc_og_mou
#     - std_og_mou
#     - isd_og_mou
#     - spl_og_mou
#     - roam_og_mou
#     - total_og_mou
#   - Minutes of usage for incoming calls for each type of call service:
#     - loc_ic_mou
#     - std_ic_mou
#     - isd_ic_mou
#     - spl_ic_mou
#     - roam_ic_mou
#     - total_ic_mou
#   - total_rech_num: Total number of recharge
#   - total_rech_amt: Total amount of recharge
#   - max_rech_amt: Maximum recharge amount
#   - total_rech_data: Total recharge for mobile internet
#   - max_rech_data: Maximum recharge for mobile internet
#   - av_rech_amt_data: Average recharge amount for mobile internet
#   - vol_2g_mb: Mobile internet usage volumn for 2G
#   - vol_3g_mb: Mobile internet usage volumn for 3G
# 
# 
# The categorical variables present in the data set are given below:
#   - night_pck_user: Prepaid service schemes for use during specific night hours only
#   - fb_user: Service scheme to avail services of Facebook and similar social networking sites
# 
# 
# 
# Most of the variables have their values recorded for 4 different months. The variable names end with the month number as explained below:
# - *.6: KPI for the month of June
# - *.7: KPI for the month of July
# - *.8: KPI for the month of August
# - *.9: KPI for the month of September
# 
# The rest of variables have been defined in the detailed data description.

# In[7]:


churn.shape


# Print information about the dataframe

# In[8]:


# summary of different feature types
churn.info(verbose=1)


# Display the summary statistics for the data set

# In[9]:


# analysis of data statistics
churn.describe(include='all')


# Create a copy of the original data

# In[10]:


# create backup of data
original = churn.copy()


# Analyze the different types of features present in the data set

# In[11]:


# create column name list by types of columns
id_cols = ['circle_id']

date_cols = ['last_date_of_month_6',
             'last_date_of_month_7',
             'last_date_of_month_8',
             'last_date_of_month_9',
             'date_of_last_rech_6',
             'date_of_last_rech_7',
             'date_of_last_rech_8',
             'date_of_last_rech_9',
             'date_of_last_rech_data_6',
             'date_of_last_rech_data_7',
             'date_of_last_rech_data_8',
             'date_of_last_rech_data_9'
            ]

cat_cols =  ['night_pck_user_6',
             'night_pck_user_7',
             'night_pck_user_8',
             'night_pck_user_9',
             'fb_user_6',
             'fb_user_7',
             'fb_user_8',
             'fb_user_9'
            ]

num_cols = [column for column in churn.columns if column not in id_cols + date_cols + cat_cols]

# print the number of columns in each list
print("#ID cols: %d\n#Date cols:%d\n#Numeric cols:%d\n#Category cols:%d" % (len(id_cols), len(date_cols), len(num_cols), len(cat_cols)))

# check if we have missed any column or not
print(len(id_cols) + len(date_cols) + len(num_cols) + len(cat_cols) == churn.shape[1])


# # Handling missing values

# ### Details on Missing Values
# 
# There are several types of features present in this data set. Some of the information that you have to look for missing value treatment is given below:
# 
# 
# *   If there are missing values in the columns corresponding to 'Recharging of the service' variables, this is because the customer did not recharge that month.
# 
# 
# *   If the columns corresponding to 'Call and Internet service' variables that have more than 70% of missing values, you can drop those variables from the data set. If not, then you can use the MICE technique to impute the values in those missing entries.
# 
# 
# *   If there are missing values in the categorical variables, this means that there is another scheme that the customer has availed from the telecomminucation service.

# Find the ratio of missing values in each column in the data set

# In[12]:


# look at missing value ratio in each column

churn.isna().sum() * 100 / churn.shape[0]


# **Checkpoint:** You must have observed that there are 40 features with more than 70% of the missing values.

# ### i) Impute missing values with zeroes

# Now that we have the information about the amount of missing values in each column, we can go ahead and perform some imputing and deleting.
# 
# First, we will start with the columns corresponding to the "recharging of the service" information.

# In[13]:


# Display summary statistics for the recharge columns
recharge_cols = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
                 'count_rech_2g_6', 'count_rech_2g_7', 'count_rech_2g_8', 'count_rech_2g_9',
                 'count_rech_3g_6', 'count_rech_3g_7', 'count_rech_3g_8', 'count_rech_3g_9',
                 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9',
                 'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
                 ]

churn[recharge_cols].describe(include='all')


# Observe whether the date of the last recharge and the total recharge data value are missing together

# In[14]:


# You can do this by displaying the rows that have null values in these two variables

churn[['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
        'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
        'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9'
       ]].isna().sum()


# Impute missing values with zeroes wherever customer didn't recharge their number that month.

# In[15]:


# create a list of recharge columns where we will impute missing values with zeroes
zero_impute = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'total_rech_data_9',
        'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9',
        'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'max_rech_data_9'
       ]


# In[16]:


# impute missing values with 0 for the above mentioned list of recharge columns

churn[zero_impute] = churn[zero_impute].fillna(0)


# Check whether the imputation has been done correctly

# In[17]:


# now, let's make sure values are imputed correctly
print("Missing value ratio:\n")
print(churn[zero_impute].isnull().sum()*100/churn.shape[0])

# summary
print("\n\nSummary statistics\n")
print(churn[zero_impute].describe(include='all'))


# Drop the id and date columns which are not required in further analyses

# In[18]:


# drop id and all the date columns
print("Shape before dropping: ", churn.shape)

churn.drop(date_cols, axis = 1, inplace = True)

print("Shape after dropping: ", churn.shape)


# ### ii) Replace NaN values in categorical variables

# The categorical variables present in the data set are given below:
#   - night_pck_user: Prepaid service schemes for use during specific night hours only
#   - fb_user: Service scheme to avail services of Facebook and similar social networking sites
# 
# If there are missing values, this means that there is another scheme that the customer has availed from the telecomminucation service.

# We will replace missing values in the categorical values with '-1' where '-1' will be a new category.

# In[19]:


# replace missing values with '-1' in categorical columns

churn[cat_cols] = churn[cat_cols].fillna(-1)


# Check for the missing value ratio

# In[20]:


# missing value ratio
print("Missing value ratio:\n")
print(churn[cat_cols].isnull().sum()*100/churn.shape[0])


# ### iii) Drop variables with more than a given threshold of missing values

# Here, we will be removing the column variables that have more than 70% of its elements missing.

# In[21]:


initial_cols = churn.shape[1]

# Insert the threshold value of missing entries
MISSING_THRESHOLD = 70

# Extract a list of columns that have less than the threshold of missing values
new_cols = [cols for cols in churn.columns if churn[cols].isna().sum()*100/churn.shape[0] < MISSING_THRESHOLD]


# In[22]:


# Include the columns extracted in the above list in the main data set
# These columns will have the percentage of missing values less than the threshold

churn = churn[new_cols]

# Display the number of columns dropped

initial_cols - len(new_cols)


# **Checkpoint:** You must have dropped 16 columns in the above step

# In[23]:


# look at missing value ratio in each column
churn.isnull().sum()*100/churn.shape[0]


# ### iv) Impute missing values using MICE

# [MICE](https://scikit-learn.org/stable/modules/impute.html) is called "Multiple Imputation by Chained Equation". It uses machine learning techniques in order to see what are the trends in the values of that column. Using this information, it will smartly fill in the missing values in that column.
# 
# MICE is now called Iterative Imputer.
# 
# You can specify the machine learning algorithm to be used in order to fill in the missing values of that column.

# In[24]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


# So, we will be using linear regression for filling the missing values in the rest of the numeric columns.

# In[25]:


churn_cols = churn.columns

# using MICE technique to impute missing values in the rest of the columns
lr = LinearRegression()

# Implement the Iterative Imputer technique to impute appropriate values in the missing entries of the rest of the numeric columns
# Note: Set the 'estimator' parameter to 'lr'  - This specifies that we will be using linear regression to estimate the missing values
# Note: Set the 'missing_values' parameter to 'np.nan' - This specifies that we have impute the entries which are NaNs
# Note: Set the 'max_iter' parameter to '1' - This specifies the number of iterations the algorithm scans through the data set
#       to converge to appropriate values it is going to impute in the missing entries. It takes around 6 min to run.
# Note: Set the 'verbose' parameter to '2' - This specifies the amount of details it will show while imputing
# Note: Set the 'imputation_order' parameter to 'roman' - This specifies the order in which features will be imputed. 'roman' means left to right
# Note: Set the 'random_state' parameter to '0' - This is for reproducibility

imp = IterativeImputer(estimator = lr, missing_values = np.nan, max_iter = 1, verbose = 2, imputation_order = 'roman', random_state = 0)

churn = imp.fit_transform(churn)


# In[26]:


churn


# In[27]:


# convert imputed numpy array to pandas dataframe
churn = pd.DataFrame(churn, columns=churn_cols)
print(churn.isnull().sum()*100/churn.shape[0])


# You can now see that we have removed or filled all the missing values from the data set.

# ### Checklist
# - Explored the data set by analyzing the summary statistics
# - Identified the types of features present in the data set
# - Computed the ratio of missing values in each of the features in the data set
# - Imputed missing values with zeroes wherever customer didn't recharge their number for any particular month
# - Replace missing values in the categorical variables with '-1' where '-1' is a new category
# - Removed the column variables that have more than 70% of its elements missing
# - Imputed the remaining  features with missing values using MICE technique
# - Retained the data set required for further analyses by dropping the irrelevant columns
# 

# We will now proceed to feature engineering to further prepare the data for testing machine learning and deep learning models.

# # Task 3: Feature engineering

# ### Description
# 
# In this task, you will extract, select, or create relevant features from your dataset.

# ### Filter high-value customers
# High-value customers are those who have recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge amount in the first two months (the good phase).

# ### Calculate total data recharge amount

# In[28]:


# calculate and store the total data recharge amount for June --> number of data recharges * average data recharge amount
# You have to use the total recharge for data and the average recharge amount for data
# June, July, August and September - The months are encoded as 6, 7, 8 and 9, respectively.

total_data_june = churn.total_rech_data_6 * churn.av_rech_amt_data_6

# calculate and store the total data recharge amount for July --> number of data recharges * average data recharge amount

total_data_july = churn.total_rech_data_7 * churn.av_rech_amt_data_7


# Add total data recharge and call recharge to get total combined recharge amount for a month

# In[29]:


# calculate and store total recharge amount for call and internet data for June --> total call recharge amount + total data recharge amount

total_rech_june = churn.total_rech_amt_6 + total_data_june

# calculate and store total recharge amount for call and internet data for July --> total call recharge amount + total data recharge amount

total_rech_july = churn.total_rech_amt_7 + total_data_july


# Compute the average recharge amount for customers in June and July

# In[30]:


# calculate average data recharge amount done by customer in June and July

churn['avg_rech_6_7'] = (total_rech_june + total_rech_july) / 2


# Find the 70th percentile for average data recharge amount for June and July

# In[31]:


# evaluate and display the 70th percentile average data recharge amount of June and July

percentile_70 = np.percentile(churn['avg_rech_6_7'], 70)

print("The 70th percentile average data recharge amount of June and July:", percentile_70)


# **Checkpoint:** You must have obtained 478 as the recharge amount at 70th percentile.

# Filter the data set for customers who have recharged their mobiles with more than or equal to 70th percentile amount

# In[32]:


# retain only those customers who have recharged their mobiles with more than or equal to 70th percentile amount
# You have seen whether each customer row has the average data recharge amount more than the 70th percentile of the average data recharge amount

churn_filtered = churn[churn['avg_rech_6_7'] >= percentile_70]


# Drop the variables which are no longer required

# In[33]:


# delete variables created to filter high-value customers

churn_filtered = churn_filtered.drop(['circle_id', 'avg_rech_6_7'], axis = 1)


# In[34]:


# Display the number of customers retained in the data set

churn_filtered.shape


# **Checkpoint:** Now you must have 30001 customers in the data set with 196 columns.

# ### Derive churn

# ### Tagging churners and removing the attributes of the churn phase
# Now tag the churned customers (churn=1, else 0) based on the fourth month as follows: those who have not made any calls (either incoming or outgoing) and have not used mobile internet even once in the churn phase. The attributes you must use to tag churners are as follows:
# 
# total_ic_mou_9
# total_og_mou_9
# vol_2g_mb_9
# vol_3g_mb_9
# After tagging churners, remove all the attributes corresponding to the churn phase (all attributes having “_9”, etc. in their names).

# Calculate total incoming and outgoing minutes of usage for the month of September

# In[35]:


# Add total incoming and outgoing minutes of usage for the month of September

churn_filtered['churn'] = 0

sept_totals = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']

churn_filtered.loc[(churn_filtered[sept_totals].sum(axis=1) == 0), 'churn'] = 1


# Calculate the total volumn of 2g and 3g data consumption for the month of September

# In[36]:


# Add the total volumn of 2g and 3g data consumption for the month of September

churn_filtered['sept_total_vol'] = churn_filtered.vol_3g_mb_9 + churn_filtered.vol_2g_mb_9


# Create churn variable by tagging customers who have not used either calls or internet in the month of September as 0 - not churn and 1 - churn otherwise

# In[37]:


# create churn variable: those who have not used either calls or internet in the month of September are customers who have churned using the lambda function
# Here 0 denotes not churn and 1 denotes churn

churn_filtered.loc[(churn_filtered['sept_total_vol'] == 0), 'churn'] = 1


# Drop the derived variables which are no longer required

# In[38]:


# delete derived variables

churn_filtered.drop(columns = sept_totals + ['sept_total_vol'], inplace=True)


# Analyze the class ratio of churn column

# In[39]:


# change the 'churn' variable data type to 'category'

churn_filtered['churn'] = churn_filtered['churn'].astype('category')

# display the churn ratio

churn_filtered.churn.value_counts()/len(churn_filtered.churn)


# ### Calculate difference between 8th and previous months

# Let's derive some variables. The most important feature, in this situation, can be the difference between the 8th month and the previous months. The difference can be in patterns such as usage difference or recharge value difference. Let's calculate difference variable as the difference between 8th month and the average of 6th and 7th month.

# In[40]:


cols =  ['arpu',
         'onnet_mou',
         'offnet_mou',
         'roam_ic_mou',
         'roam_og_mou',
         'loc_og_mou',
         'std_og_mou',
         'isd_og_mou',
         'spl_og_mou',
         'total_og_mou',
         'loc_ic_mou',
         'std_ic_mou',
         'isd_ic_mou',
         'spl_ic_mou',
         'total_ic_mou',
         'total_rech_num',
         'total_rech_amt',
         'max_rech_amt',
         'total_rech_data',
         'max_rech_data',
         'av_rech_amt_data',
         'vol_2g_mb',
         'vol_3g_mb'
         ]

# Create new columns that hold the value of the difference between the variable value
# in the month of August and average of the variable values in the month of June and July

for i in cols:
    churn['diff_' + i] = churn[i + '_8'] - ((churn[i + '_6'] + churn[i + '_7']) / 2)


# In[41]:


# let's look at summary of one of the difference variables
# The variable mentioned below is the total outgoing calls minutes of usage difference between the total OG MOU in August and average of the total OG MOU of June and July

churn['diff_total_og_mou'].describe()


# Delete columns that belong to the churn month (9th month)

# In[42]:


# update num_cols and cat_cols column name list

# extract all names that end with 9

sept_cols = [cols for cols in churn_filtered.columns if cols.endswith('_9')]

# update cal_cols so that all the variables related to the month of September are removed

cat_cols = [cols for cols in cat_cols if cols not in sept_cols]

num_cols = [column for column in churn_filtered.columns if column not in id_cols + date_cols + cat_cols + sept_cols]


# ### Checklist:
# - Extracted high-value customers by filtering those customers who have recharged with an amount more than or equal to the 70th percentile of the average recharge amount in the first two months (the good phase).
# - Dropped the variables created to filter hight value customers
# - Created the churn variable by tagging customers who have not used either calls or internet in the month of September as 0 - not churn and 1 - churn otherwise
# - Derived new features by calculating the total outgoing calls minutes of usage difference between the total OG MOU in August and average of the total OG MOU of June and July
# - Removed the variables related to the churn phase

# # Task 4: Data Visualization

# ### Description:
# In this task, you will visually represent and interpret patterns, trends, and relationships within the features in your dataset.

# Check the data types of the numerical and categorical columns in the data set

# In[43]:


# Ensure that all the numerical and categorical columns are of the correct data types

print('number of true datatypes for numerical columns:', sum(churn_filtered[i].dtype == 'float64' for i in num_cols))
print('number of true datatypes for categorical columns:', sum(churn_filtered[i].dtype == 'categorical' for i in cat_cols))


# In[44]:


# datatype for categorical columns is not correct

for col in cat_cols:
    churn_filtered[col] = churn_filtered[col].astype('category')

print('number of true datatypes for categorical columns:', sum(churn_filtered[i].dtype == 'category' for i in cat_cols))


# Create a function to do the univariate and bivariate analysis of the features present in the data set

# In[45]:


# create plotting functions
def data_type(variable):
    if variable.dtype == np.int64 or variable.dtype == np.float64:
        return 'numerical'
    elif variable.dtype == 'category':
        return 'categorical'

def univariate(variable, stats=True, x_axis_range=None):

    if data_type(variable) == 'numerical':
        sns.distplot(variable)
        if x_axis_range:
            plt.xlim(x_axis_range)
        if stats == True:
            print(variable.describe())

    elif data_type(variable) == 'categorical':
        sns.countplot(variable)
        if stats == True:
            print(variable.value_counts())

    else:
        print("Invalid variable passed: either pass a numeric variable or a categorical vairable.")


# ## Univariate EDA

# In[46]:


# Plot the average revenue per user in June

univariate(churn['arpu_6'], x_axis_range=(-200, 2000))


# In[47]:


# Plot the minutes of usage of local (within same telecom circle) outgoing calls of Operator T to other operator fixed line

univariate(churn['loc_ic_t2o_mou'])


# In[48]:


# Plot the minutes of usage of STD (outside the calling circle) outgoing calls of Operator T to other operator fixed line

univariate(churn['std_og_t2o_mou'])


# In[49]:


# Plot the minutes of usage of all kind of calls within the same operator network for the month of August

univariate(churn['total_og_mou_7'], x_axis_range=(-200, 2000))


# In[50]:


# Plot the minutes of usage of all kind of calls outside the operator T network for the month of September

univariate(churn['roam_og_mou_8'], x_axis_range=(-200, 2500))


# ## Bivariate EDA

# Now visualize and analyse the relationship between different features in the data set

# In[51]:


# Plot the relationship between different variables present in the data set


# In[52]:


# scatter plots

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
for i, j in enumerate([6, 7, 8]):
    sns.scatterplot(data = churn_filtered, x = f'arpu_{j}', y = f'total_rech_amt_{j}', ax = ax[i])

plt.tight_layout()
plt.show()


# In[53]:


# box plots

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
for i, j in enumerate([6, 7, 8]):
    sns.boxplot(data = churn_filtered, x = f'arpu_{j}', y = f'night_pck_user_{j}', ax = ax[i])

plt.tight_layout()
plt.show()


# ### Cap outliers in all numeric variables

# Create a function to deal with outliers using the IQR method

# In[54]:


# function for capping outliers
def cap_outliers(array):

    # Get the 75% quantile of the array
    # Get the 25% quantile of the array
    # Get the interquartile range (IQR) (q3 - q1)

    q3 = np.percentile(array, 75)
    q1 = np.percentile(array, 25)
    iqr = q3 - q1

    # Calculate the upper limit - 75% quartile + 1.5*IQR
    # Calculate the lower limit - 25% quartile - 1.5*IQR

    upper_limit = q3 + 1.5*iqr
    lower_limit = q1 - 1.5*iqr

    # Perform outlier capping
    # Set all the values in the array above the upper limit to be equal to the upper limit
    # Set all the values in the array below the lower limit to be equal to the lower limit

    array[array > upper_limit] = upper_limit
    array[array < lower_limit] = lower_limit

    return array


# The following is an example to help you understand how capping is done to treat outliers

# In[55]:


# example of capping
sample_array = list(range(100))

# add outliers to the data
sample_array[0] = -9999
sample_array[99] = 9999

# cap outliers
sample_array = np.array(sample_array)
print("Array after capping outliers: \n", cap_outliers(sample_array))


# Use the outlier capping function to cap the outliers present in all the numeric columns in the data set

# In[56]:


# cap outliers in all the numeric columns using your outlier capping function

for cols in num_cols:
    churn_filtered[cols] = cap_outliers(np.array(churn_filtered[cols]))


# In[57]:


# viewing boxplots & scatter plots again

fig, ax = plt.subplots(2, 3, figsize = (15, 10))
for i, j in enumerate([6, 7, 8]):
    sns.boxplot(data = churn_filtered, x = f'arpu_{j}', y = f'night_pck_user_{j}', ax = ax[0][i])
    sns.scatterplot(data = churn_filtered, x = f'arpu_{j}', y = f'total_rech_amt_{j}', ax = ax[1][i])

plt.tight_layout()
plt.show()


# **Checklist:**
# - Created functions to carry out univariate and bivariate analysis of the columns in the data set
# - Capped outliers by creating a function and applying it on all the numerical features

# # Task 5: Modeling

# ### Description:
# In this task, you will train and evaluate predictive models using your prepared dataset.

# ## i) Importing necessary libraries for machine learning and deep learning

# In[58]:


get_ipython().system('pip install imblearn')


# In[59]:


#algorithms for sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#baseline linear model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#modules for hyper parameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#modules for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.metrics import precision_score, accuracy_score, f1_score, r2_score
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix


# In[60]:


# Import methods for building neural networks
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Import 'KerasClassifier' from 'keras' for connecting neural networks with 'sklearn' and 'GridSearchCV'
from keras.wrappers.scikit_learn import KerasClassifier


# ## ii) Preprocessing data

# In[61]:


# change churn to numeric
churn_filtered['churn'] = churn_filtered['churn'].astype('float64')


# In[62]:


# Extract input and output data

X = churn_filtered.drop('churn', axis = 1)
y = churn_filtered.churn


# Create dummy variables for the categorical features

# In[63]:


# Use dummy variables for categorical variables

X = pd.get_dummies(X, columns = cat_cols, drop_first = True)


# ### Train Test split

# In[64]:


# Divide data into train and test
# Note: Set the 'random_state' parameter to '4'
# Note: Set the 'test_size' parameter to '0.25'

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4)

# print shapes of train and test sets
X_train.shape
y_train.shape
X_test.shape
y_test.shape


# **Checkpoint:** You must have obtained 22500 observations in the train set and 7501 observations in the test set.

# In[65]:


X_new = X.to_numpy()
y_new = np.array(y).reshape(-1, 1)

#train-test split using stratified K fold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X_new, y_new)

for train_index, test_index in skf.split(X_new,y):
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = y_new[train_index], y_new[test_index]


# ### Handling Class Imbalance

# Classification tasks often involve datasets with class imbalances, where the number of samples in one class significantly outweighs the other(s). Class imbalance can pose significant challenges to the learning algorithms, as they tend to favor the majority class and struggle to accurately predict the minority class. Data augmentation is one such technique that you have studied earlier, however, in this case study we will be exploring class imbalance techniques as an alternative or complementary approach.
# 
# While data augmentation has proven to be a valuable tool in addressing class imbalance, recent research highlights the advantages of leveraging class imbalance techniques as a primary approach or in conjunction with augmentation methods. By explicitly addressing the class imbalance issue, these techniques ensure that the learning algorithm better captures the nuances of all classes, resulting in improved classification performance.
# 
# In this capstone, observe that the dataset is imbalanced. You should get the number of entries with output '1' approximately 1/10th of the number of entries with output '0'. This means that if we run a simple machine learning model, it should already show 90% accuracy.
# 
# It is the most important for the model to predict which customer will churn as this will decide how their business is performing. We have to create a model that will predict the output '1' accurately. But its corresponding number of entries are very less.
# 
# Hence, we will be doing some sampling methods to make the data set balanced.

# 1) **Random Under-Sampling**: This method basically consists of removing data in order to have a more balanced dataset and thus avoiding our models to overfitting.
# 
# We have seen how imbalanced the data set is. With random under-sampling, we have a sub-sample of our dataframe with a 50/50 ratio with regards to our classes. This means that if there are 1221 '0' class data entries, then there will be 1221 '1' class data entries by removing the rest.
# 
# Note: The main issue with "Random Under-Sampling" is that we run the risk that our classification models will not perform as accurate as we would like to since there is a great deal of information loss.

# In[66]:


# random under sampling using imblearn
# Use the RandomUnderSampler (RUS) function to produce new X and y from X_train and y_train
# Use random_state as 1 for reproducibility

rus = RandomUnderSampler(random_state=1)
X_rus, y_rus = rus.fit_resample(X_train, y_train)


# In[67]:


X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_rus, y_rus, test_size=0.2, random_state=42, stratify=y_rus)


# In[68]:


X_train_rus.shape
X_test_rus.shape
y_train_rus.shape
y_test_rus.shape


# 1) **Random Over-Sampling**: This method basically consists of adding data in order to have a more balanced dataset and thus avoiding our models to overfitting.
# 
# We have seen how imbalanced the data set is. With random over-sampling, we have a sub-sample of our dataframe with a 50/50 ratio with regards to our classes. This means that if there are 13780 '1' class data entries, then there will be 13780 '0' class data entries by removing the rest.

# In[69]:


# random over sampling with imblearn
# Use the RandomOverSampler (ROS) function to produce new X and y from X_train and y_train
# Use random_state as 1 for reproducibility

ros = RandomOverSampler(random_state=1)
X_ros, y_ros = ros.fit_resample(X_train, y_train)


# In[70]:


#train Test split
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros, y_ros, test_size=0.2, stratify=y_ros, random_state=42)


# In[71]:


X_train_ros.shape
X_test_ros.shape
y_train_ros.shape
y_test_ros.shape


# Now, let's test different machine learning models over the three data sets, namely, the original cleaned data set, the under-sampled data set and the over-sampled data set.

# ## Logistic Regression

# Build a logistic regression model without applying any techniques to address class imbalance

# In[72]:


# Defining the logistic regression model and fit it on the normal X_train and y_train
# 'penalty' is set to 'none'
# 'solver' is set to 'lbfgs'
# 'random_state' is set to 0
# 'max_iter' is set to 100
# You can change these values or use GridSearchCV to perform hyperparameter tuning to find the optimal performing model
model_name = 'Logistic Regression - without balancing'

lr = LogisticRegression(penalty = 'none', solver = 'lbfgs', random_state = 0, max_iter = 100)
lr_model = lr.fit(X_train, y_train)

# Evaluating the accuracy of the training and validation sets

y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

log_train_acc = accuracy_score(y_train, y_train_pred)
log_test_acc = accuracy_score(y_test, y_test_pred)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = r2_score(y_test, y_test_pred)

# creating a dataframe to compare the performance of different models
model_eval_data = [[model_name, log_train_acc, log_test_acc, f_score, precision, recall]]
evaluate_df = pd.DataFrame(model_eval_data, columns=['Model Name', 'Training Score', 'Testing Score',
                                          'F1 Score', 'Precision', 'Recall'])


# Train a logistic regression model on a balanced data set achieved through random undersampling

# In[73]:


# Defining the logistic regression model and fit it on the random under sampled X_train_rus and y_train_rus
# 'penalty' is set to 'none'
# 'solver' is set to 'lbfgs'
# 'random_state' is set to 0
# 'max_iter' is set to 100
model_name = 'Logistic Regression - Random Undersampling'

lr = LogisticRegression(penalty = 'none', solver = 'lbfgs', random_state = 0, max_iter = 100)
lr_model = lr.fit(X_train_rus, y_train_rus)

# Evaluating the accuracy of the training and validation sets

y_train_pred = lr_model.predict(X_train_rus)
y_test_pred = lr_model.predict(X_test_rus)

log_train_acc = accuracy_score(y_train_rus, y_train_pred)
log_test_acc = accuracy_score(y_test_rus, y_test_pred)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_rus, y_test_pred)
precision = precision_score(y_test_rus, y_test_pred)
recall = r2_score(y_test_rus, y_test_pred)

# adding calculations to dataframe
model_eval_data = [model_name, log_train_acc, log_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a logistic regression model on a balanced dataset achieved through random oversampling

# In[74]:


# Defining the logistic regression model and fit it on the random over sampled X_train_ros and y_train_ros
# 'penalty' is set to 'none'
# 'solver' is set to 'lbfgs'
# 'random_state' is set to 0
# 'max_iter' is set to 100
model_name = 'Logistic Regression - Random Oversampling'

lr = LogisticRegression(penalty = 'none', solver = 'lbfgs', random_state = 0, max_iter = 100)
lr_model_wob = lr.fit(X_train_ros, y_train_ros)

# Evaluating the accuracy of the training and validation sets

y_train_pred = lr_model_wob.predict(X_train_ros)
y_test_pred = lr_model_wob.predict(X_test_ros)

log_train_acc = accuracy_score(y_train_ros, y_train_pred)
log_test_acc = accuracy_score(y_test_ros, y_test_pred)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_ros, y_test_pred)
precision = precision_score(y_test_ros, y_test_pred)
recall = r2_score(y_test_ros, y_test_pred)


# adding calculations to dataframe
model_eval_data = [model_name, log_train_acc, log_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# ## Decision Tree

# Build a decision tree model without applying any techniques to address class imbalance

# In[75]:


# Defining the decision tree model and fit it on the normal X_train and y_train
# 'max_depth' is set to 50
# 'random_state' is set to 0
# You can change these values or use GridSearchCV to perform hyperparameter tuning to find the optimal performing model
model_name = 'Decision Tree - without balancing'

dt = DecisionTreeClassifier(max_depth = 50, random_state = 0)
dt_model = dt.fit(X_train, y_train)

# Evaluating the accuracy of the training and validation sets

y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)

tree_train_acc = accuracy_score(y_train, y_pred_train)
tree_test_acc = accuracy_score(y_test, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = r2_score(y_test, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, tree_train_acc, tree_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a decision tree model on a balanced dataset achieved through random undersampling

# In[76]:


# Defining the decision tree model and fit it on the random under sampled X_train_rus and y_train_rus
# 'max_depth' is set to 50
# 'random_state' is set to 0
model_name = 'Decision Tree - Random Undersampling'

dt = DecisionTreeClassifier(max_depth = 50, random_state = 0)
dt_model = dt.fit(X_train_rus, y_train_rus)

# Evaluating the accuracy of the training and validation sets

y_pred_train = dt_model.predict(X_train_rus)
y_pred_test = dt_model.predict(X_test_rus)

tree_train_acc = accuracy_score(y_train_rus, y_pred_train)
tree_test_acc = accuracy_score(y_test_rus, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_rus, y_pred_test)
precision = precision_score(y_test_rus, y_pred_test)
recall = r2_score(y_test_rus, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, tree_train_acc, tree_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a decision tree model on a balanced dataset achieved through random oversampling

# In[77]:


# Defining the decision tree model and fit it on the random over sampled X_train_ros and y_train_ros
# 'max_depth' is set to 50
# 'random_state' is set to 0
model_name = 'Decision Tree - Random Oversampling'

dt = DecisionTreeClassifier(max_depth = 50, random_state = 0)
dt_model = dt.fit(X_train_ros, y_train_ros)

# Evaluating the accuracy of the training and validation sets

y_pred_train = dt_model.predict(X_train_ros)
y_pred_test = dt_model.predict(X_test_ros)

tree_train_acc = accuracy_score(y_train_ros, y_pred_train)
tree_test_acc = accuracy_score(y_test_ros, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_ros, y_pred_test)
precision = precision_score(y_test_ros, y_pred_test)
recall = r2_score(y_test_ros, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, tree_train_acc, tree_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# ## kNN

# Build a KNN model without applying any techniques to address class imbalance

# In[78]:


# Defining the kNN model and fit it on the normal X_train and y_train
# 'n_neighbors' is set to 14
# You can change these values or use GridSearchCV to perform hyperparameter tuning to find the optimal performing model
model_name = 'kNN - without balancing'

knn = KNeighborsClassifier(n_neighbors = 14)
knn_model = knn.fit(X_train, y_train)

# Evaluating the accuracy of the training and validation sets

y_pred_train = knn_model.predict(X_train)
y_pred_test = knn_model.predict(X_test)

knn_train_acc = accuracy_score(y_train, y_pred_train)
knn_test_acc = accuracy_score(y_test, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = r2_score(y_test, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, knn_train_acc, knn_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a KNN model on a balanced dataset achieved through random undersampling

# In[79]:


# Defining the kNN model and fit it on the random under sampled X_train_rus and y_train_rus
# 'n_neighbors' is set to 14
model_name = 'kNN - Random Undersampling'

knn = KNeighborsClassifier(n_neighbors = 14)
knn_model = knn.fit(X_train_rus, y_train_rus)

# Evaluating the accuracy of the training and validation sets

y_pred_train = knn_model.predict(X_train_rus)
y_pred_test = knn_model.predict(X_test_rus)

knn_train_acc = accuracy_score(y_train_rus, y_pred_train)
knn_test_acc = accuracy_score(y_test_rus, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_rus, y_pred_test)
precision = precision_score(y_test_rus, y_pred_test)
recall = r2_score(y_test_rus, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, knn_train_acc, knn_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a KNN model on a balanced dataset achieved through random oversampling

# In[80]:


# Defining the kNN model and fit it on the random over sampled X_train_ros and y_train_ros
# 'n_neighbors' is set to 14
model_name = 'kNN - Random Oversampling'

knn = KNeighborsClassifier(n_neighbors = 14)
knn_model = knn.fit(X_train_ros, y_train_ros)

# Evaluating the accuracy of the training and validation sets

y_pred_train = knn_model.predict(X_train_ros)
y_pred_test = knn_model.predict(X_test_ros)

knn_train_acc = accuracy_score(y_train_ros, y_pred_train)
knn_test_acc = accuracy_score(y_test_ros, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_ros, y_pred_test)
precision = precision_score(y_test_ros, y_pred_test)
recall = r2_score(y_test_ros, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, knn_train_acc, knn_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# ## Random Forest Classifier

# Build a random forest model without applying any techniques to address class imbalance

# In[81]:


# Defining the Random Forest Classifier model and fit it on the normal X_train and y_train
# 'n_estimators' is set to 200
# 'max_depth' is set to 5
# 'class_weight' is set to 'balanced'
# 'random_state' is set to 123
# You can change these values or use GridSearchCV to perform hyperparameter tuning to find the optimal performing model
model_name = 'Random Forest - without balancing'

rf = RandomForestClassifier(n_estimators = 200, max_depth = 5, class_weight = 'balanced', random_state = 123)
rf_model = rf.fit(X_train, y_train)

# Evaluating the accuracy of the training and validation sets

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

rf_train_acc = accuracy_score(y_train, y_pred_train)
rf_test_acc = accuracy_score(y_test, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = r2_score(y_test, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, rf_train_acc, rf_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a random forest model on a balanced dataset achieved through random undersampling

# In[82]:


# Defining the Random Forest Classifier model and fit it on the random under sampled X_train_rus and y_train_rus
# 'n_estimators' is set to 200
# 'max_depth' is set to 5
# 'class_weight' is set to 'balanced'
# 'random_state' is set to 123
model_name = 'Random Forest - Random Undersampling'

rf = RandomForestClassifier(n_estimators = 200, max_depth = 5, class_weight = 'balanced', random_state = 123)
rf_model = rf.fit(X_train_rus, y_train_rus)

# Evaluating the accuracy of the training and validation sets

y_pred_train = rf_model.predict(X_train_rus)
y_pred_test = rf_model.predict(X_test_rus)

rf_train_acc = accuracy_score(y_train_rus, y_pred_train)
rf_test_acc = accuracy_score(y_test_rus, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_rus, y_pred_test)
precision = precision_score(y_test_rus, y_pred_test)
recall = r2_score(y_test_rus, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, rf_train_acc, rf_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Train a random forest model on a balanced dataset achieved through random oversampling

# In[83]:


# Defining the Random Forest Classifier model and fit it on the random over sampled X_train_ros and y_train_ros
# 'n_estimators' is set to 200
# 'max_depth' is set to 5
# 'class_weight' is set to 'balanced'
# 'random_state' is set to 123
model_name = 'Random Forest - Random Oversampling'

rf = RandomForestClassifier(n_estimators = 200, max_depth = 5, class_weight = 'balanced', random_state = 123)
rf_model = rf.fit(X_train_ros, y_train_ros)

# Evaluating the accuracy of the training and validation sets

y_pred_train = rf_model.predict(X_train_ros)
y_pred_test = rf_model.predict(X_test_ros)

rf_train_acc = accuracy_score(y_train_ros, y_pred_train)
rf_test_acc = accuracy_score(y_test_ros, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_ros, y_pred_test)
precision = precision_score(y_test_ros, y_pred_test)
recall = r2_score(y_test_ros, y_pred_test)

# adding calculations to dataframe
model_eval_data = [model_name, rf_train_acc, rf_test_acc, f_score, precision, recall]
model_eval_dict = {evaluate_df.columns[i]:model_eval_data[i] for i in range(len(model_eval_data))}
evaluate_df = evaluate_df.append(model_eval_dict, ignore_index=True)


# Compare the performances of the different predictive models that you built above

# In[84]:


evaluate_df


# In this case study, the most important factor in the prediction performance of a machine learning model is that it should be able to predict the positive class as accurately as possible. This means that the false negatives and false positives are supposed to be as minimal as possible. This further means that precision and recall should be as high as possible.
# 
# There is another factor to consider. The most important factor which can lead to a company loss is the false negatives. This is because if we predict that a customer did not churn but in reality, the customer did, the company will miss out on the data of churned customers. Hence, observing the recall factor is much more important than precision.

# ## Hyperparameter tuning using GridSearchCV

# Choose the model that performs in a robust manner with good accuracy, precision and recall. Especially look out for the recall value because a good recall value means that it is able to accurately classify the data examples of the customers who churned

# In[85]:


# Define your model and parameter grid
# Make sure to use random_state value as 0

base_rf_model = RandomForestClassifier(random_state = 0)
params_grid = {'n_estimators': np.arange(100, 400, 50), 'max_depth': np.arange(3, 8, 1)}

# Perform GridSearchCV

grid = GridSearchCV(estimator = base_rf_model, param_grid = params_grid, scoring = 'accuracy', cv = 2)
best_model = grid.fit(X_train, y_train)

# Display the best combination of parameters obtained from GridSearchCV

best_n_estimators = best_model.best_params_['n_estimators']
best_max_depth = best_model.best_params_['max_depth']

print('The optimal value of n_estimators is', best_n_estimators)
print('The optimal value of max_depth is', best_max_depth)


# Retrain your model on the combination of parameters obtained from GridSearchCV

# In[86]:


# Re-fit your model with the combination of parameters obtained from GridSearchCV
# Make sure to use random_state value as 0

rf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth = best_max_depth, class_weight = 'balanced', random_state = 0)
best_rf_model = rf.fit(X_train_ros, y_train_ros)

# Evaluating the accuracy of the training and validation sets

y_pred_train = best_rf_model.predict(X_train_ros)
y_pred_test = best_rf_model.predict(X_test_ros)

rf_train_acc = accuracy_score(y_train_ros, y_pred_train)
rf_test_acc = accuracy_score(y_test_ros, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_ros, y_pred_test)
precision = precision_score(y_test_ros, y_pred_test)
recall = r2_score(y_test_ros, y_pred_test)

print('Training accuracy: ', rf_train_acc)
print('Testing accuracy: ', rf_test_acc)
print('f1 score: ', f_score)
print('Precision: ', precision)
print('Recall value: ', recall)


# In[87]:


# Find the importance of all the features according to the optimal model defined above

features = best_rf_model.feature_importances_


# Create a dataframe with the feature importance in descending order so that the highest important features are shown at the start of the dataframe

# In[88]:


# Display the dataframe obtained

features_df = pd.DataFrame({'Feature': X.columns, 'Importance': features})
features_df = features_df.sort_values(by='Importance', ascending=False)

features_df.head(10)


# Assess the performance of your model on different evaluation metrics

# In[89]:


# confusion matrix

ConfusionMatrixDisplay.from_estimator(best_rf_model, X_train_ros, y_train_ros, cmap = plt.cm.Blues);


# In[90]:


# ROC-AUC curve

probs_val = best_rf_model.predict_proba(X_test_ros)[:, 1]
probs_train = best_rf_model.predict_proba(X_train_ros)[:, 1]

auc_val = roc_auc_score(y_test_ros, probs_val)
auc_train = roc_auc_score(y_train_ros, probs_train)

fpr_val, tpr_val, _ = roc_curve(y_test_ros, probs_val)
fpr_train, tpr_train, _ = roc_curve(y_train_ros, probs_train)

plt.plot(fpr_val, tpr_val, marker='.', label=f'Random Forest (validaiton) (AUC = {auc_val:.3f})')
plt.plot(fpr_train, tpr_train, marker='.', label=f'Random Forest (train) (AUC = {auc_train:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

plt.legend()
plt.title(f'ROC Curve for Random Forest - Random Oversampling')

plt.show()


# In[91]:


# precision recall curve

probs_val = best_rf_model.predict_proba(X_test_ros)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_ros, probs_val)

plt.plot(recall, precision)
plt.title(f'Precision-Recall Curve for ROC Curve for Random Forest - Random Oversampling')
plt.ylabel('Precision')
plt.xlabel('Recall')

plt.show()


# ## Neural Networks

# Create a neural network model with the defined set of hyperparameters

# In[92]:


# Define a function to create a neural network model and specify default values for variable hyperparameters
# Note: The number of hidden layers is fixed at 2
# Note: The number of neurons in the second hidden layer is fixed at 64
# Note: The output layer activation function is fixed as 'sigmoid'

# You can change the hyperparameters mentioned as arguments in the create_nn function
# So that you can use them in GridSearchCV hyperparameter tuning
# Feel free to modify the model too and test the model performance
# You can add more types of layers like Dropout, Batch normalization etc.

# Note: The variable hyperparameters list is the activation functions of the hidden layers and number of neurons in the first hidden layer
def create_nn(activation_function='relu', hidden1_neurons=256, learning_rate_value=0.001, input_dim = X_train.shape[1]):

    # Declare an instance of an artificial neural network model using the 'Sequential()' method
    nn = Sequential()

    # keras.Input is the input layer of the neural network

    # Add a hidden layer using the 'add()' and 'Dense()' methods
    # Note: Set the 'units' parameter to 'hidden1_neurons'  - This specifies the number of neurons in the hidden layer
    # Note: Set the 'activation' parameter to 'activation_function' - This specifies the activation function parameter defined in the custom function
    nn.add(Dense(units=hidden1_neurons, activation=activation_function, input_dim=input_dim))

    # Add a hidden layer using the 'add()' and 'Dense()' methods
    # Note: Set the 'units' parameter to 64  - This specifies the number of neurons in the hidden layer
    # Note: Set the 'activation' parameter to 'activation_function' - This specifies the activation function parameter defined in the custom function
    nn.add(Dense(units=64, activation=activation_function))

    # Add the output layer using the 'add()' and 'Dense()' methods
    # Note: Set the 'units' parameter to 1 - Binary classification
    # Note: Set the 'activation' parameter to 'sigmoid' - The sigmoid activation function is used for output layer neurons in binary classification tasks
    nn.add(Dense(units=1, activation='sigmoid'))

    # Compile the model using the 'compile()' method
    # Note: Set the 'loss' parameter to 'binary_crossentropy' - The binary crossentropy loss function is commonly used for binary classification tasks
    # Note: Set the 'metrics' parameter to 'accuracy' - This records the accuracy of the model along with the loss during training
    # Note: Set the 'optimizer' parameter to 'RMSprop' and set its 'learning_rate' parameter to 'learning_rate_value' - This specifies the learning rate value defined in the custom function
    optimizer = RMSprop(learning_rate=learning_rate_value)
    nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return nn


# In[93]:


# Create a default neural network using the 'create_nn' function and train it on the training data
nn1 = create_nn()

# Capture the training history of the model using the 'fit()' method
# Note: Set the 'validation_data' parameter to (X_val, y_val)
# Note: Set the 'epochs' parameter to 10 - This specifies the scope of loss computations and parameter updates
nn1.summary()
print('\n')
nn1_history = nn1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)


# In[94]:


# Convert the neural network history object into a data frame to view its specifics
import pandas as pd
hist = pd.DataFrame(nn1_history.history)
hist['epoch'] = nn1_history.epoch
hist['epoch'] = hist['epoch'].apply(lambda x: x + 1)
hist.set_index('epoch')


# Plot the training and validation accuracies for different values of epoch

# In[95]:


# View the training and validation accuracies as functions of epoch
plt.figure(figsize = (14, 4))

sns.lineplot(data = hist, x = 'epoch', y = 'accuracy', color = 'red', label = 'Training')
sns.lineplot(data = hist, x = 'epoch', y = 'val_accuracy', color = 'blue', label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Epoch');


# Assess the performance of the model on the validation data set

# In[96]:


# Compute the final accuracy of the model on the validation data set using the 'evaluate()' method
performance_test = nn1.evaluate(X_test, y_test)

print('The loss value of the model on the validation data is {}'.format(performance_test[0]))
print('The accuracy of the model on the validation data is {}'.format(performance_test[1]))


# Find the optimal parameters using GridSearchCV

# In[97]:


# Initialize a basic NN object using the 'KerasClassifier()' method
# Note: Set the 'build_fn' parameter to 'create_nn' - This converts the 'create_nn' function into a 'KerasClassifier' object
base_grid_model = KerasClassifier(build_fn = create_nn)

# Define a list of 'activation_function' and 'hidden1_neurons' parameters and store it in a parameter grid dictionary
parameters_grid = {'activation_function': ['relu','sigmoid'],
                   'hidden1_neurons': [256, 512]}

# Perform a grid search using the 'GridSearchCV()' method to obtain a grid on which to fit the training data
# Note: Set the 'estimator' parameter to 'base_grid_model' - This specifies the estimator to be used by 'GridSearchCV()'
# Note: Set the 'param_grid' parameter to 'parameters_grid' - This specifies the grid of parameters to search over
# Note: Set the 'cv' parameter to 2 - This specifies the number of folds in the cross-validation process
# Note: Set the 'verbose' parameter to 4 - This helps show more relevant information during training
grid = GridSearchCV(estimator = base_grid_model, param_grid = parameters_grid, cv = 2, verbose = 4)

# Train the model on the training data using the 'fit()' method
# Note: Use the default batch size or set it to 32
# Note: Set the 'epochs' parameter to 10
# Note: The 'validation_split' parameter isn't particularly required since cross-validation is already in place
grid_model = grid.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10)

# Print the optimal values of 'activation_function' and 'hidden1_neurons'
best_activation_function = grid_model.best_params_['activation_function']
best_hidden1_neurons = grid_model.best_params_['hidden1_neurons']
best_accuracy = grid_model.best_score_

print('\n The optimal value of convolution filter size is', best_activation_function)
print('\n The optimal value of maxpooling filter size is', best_hidden1_neurons)
print('\n The accuracy of the model with these optimal parameters is ', best_accuracy)


# Retrain the model with the optimal combination of hyperparameters and save its training history

# In[98]:


# Use the 'create_nn' function to create a NN with the optimal values of 'filter_size' and 'pool_filter_size'
# Note: Set the 'activation_function' parameter to 'best_activation_function' - This specifies the optimal value for the 'activation_function' parameter
# Note: Set the 'hidden1_neurons' parameter to 'best_hidden1_neurons' - This specifies the optimal value for the 'hidden1_neurons' parameter
nn1 = create_nn(activation_function = best_activation_function, hidden1_neurons = best_hidden1_neurons)

# Capture the training history of the model using the 'fit()' method
# Note: Set the 'validation_data' parameter to (X_val, y_val)
# Note: Use the default batch size or set it to 32
# Note: Set the 'epochs' parameter to 10
nn1.summary()
print('\n')
nn1_history = nn1.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10)
hist = pd.DataFrame(nn1_history.history)
hist['epoch'] = nn1_history.epoch


# Plot the training and validation accuracies for different values of epoch

# In[99]:


# View the training and validation accuracies as functions of epoch
plt.figure(figsize = (14, 4))

sns.lineplot(data = hist, x = 'epoch', y = 'accuracy', color = 'red', label = 'Training')
sns.lineplot(data = hist, x = 'epoch', y = 'val_accuracy', color = 'blue', label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Epoch');


# Evaluate the performance of the model on the testing data

# In[100]:


# Compute the accuracy of the model on the testing data set using the 'evaluate()' method
performance_test = nn1.evaluate(X_test, y_test)

print('The loss value of the model on the test data is {}'.format(performance_test[0]))
print('The accuracy of the model on the test data is {}'.format(performance_test[1]))


# ### Find the optimal parameters using RandomizedSearchCV
# 
# Randomized search cross-validation is a technique used for hyperparameter optimization in machine learning models. It is an alternative to grid search, which exhaustively searches through all possible combinations of hyperparameters.
# 
# In randomized search cross-validation, instead of trying every combination, a fixed number of random combinations of hyperparameters are sampled from a predefined search space. This approach allows for a more efficient exploration of the hyperparameter space, especially when the search space is large.
# 
# 
# You can understand about its implementation [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

# In[101]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter distribution for randomized search
parameters_dist = {
    'activation_function': ['relu', 'sigmoid'],
    'hidden1_neurons': randint(256, 513)  # Range of values for hidden1_neurons
}

# Initialize a basic NN object using the 'KerasClassifier()' method
base_random_model = KerasClassifier(build_fn = create_nn)

# Perform randomized search using the 'RandomizedSearchCV()' method

# Note: Set the 'estimator' parameter to 'base_random_model' - This specifies the estimator to be used by 'RandomizedSearchCV()'
# Note: Set the 'param_distributions' parameter to 'parameters_dist' - This specifies the parameters to search over
# Note: Set the 'cv' parameter to 2 - This specifies the number of folds in the cross-validation process
# Note: Set the 'n_iter' parameter to 2 - This specifies the number of parameter settings that are sampled
# Note: Set the 'verbose' parameter to 4 - This helps show more relevant information during training
# Note: Set the 'random_state' parameter to 0 - This helps generate the consistent results across multiple runs
randomized_search = RandomizedSearchCV(estimator = base_random_model, param_distributions = parameters_dist, cv = 2, n_iter = 2, verbose = 4, random_state = 0)

# Train the model on the training data using the 'fit()' method
# Note: Set the 'epochs' parameter to 10
randomized_model = randomized_search.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10)

# Print the optimal values of 'activation_function' and 'hidden1_neurons'
best_activation_function = randomized_model.best_params_['activation_function']
best_hidden1_neurons = randomized_model.best_params_['hidden1_neurons']
best_accuracy = randomized_model.best_score_

print('\nThe optimal value of activation function is', best_activation_function)
print('\nThe optimal value of hidden1_neurons is', best_hidden1_neurons)
print('\nThe accuracy of the model with these optimal parameters is', best_accuracy)


# Retrain the model on the optimal set of hyperparameters

# In[102]:


# Use the 'create_nn' function to create a NN with the optimal values of 'filter_size' and 'pool_filter_size'
# Note: Set the 'activation_function' parameter to 'best_activation_function' - This specifies the optimal value for the 'activation_function' parameter
# Note: Set the 'hidden1_neurons' parameter to 'best_hidden1_neurons' - This specifies the optimal value for the 'hidden1_neurons' parameter
nn2 = create_nn()

# Capture the training history of the model using the 'fit()' method
# Note: Set the 'validation_data' parameter to (X_val, y_val)
# Note: Use the default batch size or set it to 32
# Note: Set the 'epochs' parameter to 10
nn2.summary()
print('\n')
nn2_history = nn2.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10)
hist = pd.DataFrame(nn2_history.history)
hist['epoch'] = nn2_history.epoch


# Plot the training and validation accuracies for different values of epoch

# In[103]:


# View the training and validation accuracies as functions of epoch
plt.figure(figsize = (14, 4))

sns.lineplot(data = hist, x = 'epoch', y = 'accuracy', color = 'red', label = 'Training')
sns.lineplot(data = hist, x = 'epoch', y = 'val_accuracy', color = 'blue', label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Epoch');


# Evaluate the performance of the model on the testing data

# In[104]:


# Compute the accuracy of the model on the testing data set using the 'evaluate()' method
performance_test = nn2.evaluate(X_test, y_test)

print('The loss value of the model on the test data is {}'.format(performance_test[0]))
print('The accuracy of the model on the test data is {}'.format(performance_test[1]))


# **Checklist:**
# - Importing necessary libraries for machine learning and deep learning
# - Preprocessed the data
# - Divided the data set into train and test splits
# - Handled class imbalance using random undersampling and random oversampling
# - Built and evaluated different machine learning models such as logistic regression, decision trees, KNN and random forest models with and without treating class imbalance
# - Tuned the best machine learning using GridSearchCV for the optimal hyperparameters
# - Built and evaluated a neural network model and tuned for its hyperparameters using GridSearchCV and RandomizedSearchCV

# ## Task 6: Business Insights: Misclassification Costs
# Our first step is to understand the current profitability of the telecomminucation service program, and then to is to estimate the impact of our model. We are going to use misclassification costs to study the impact.
# 
# We are going to use \\\$500 as an approximation company loss for the false negative cost, and \\\$300 company loss for the false positive cost. Note: We are interested in finding the best cut-off that will maximize the benefit of our machine learning model.
# 

# In[105]:


# Define the false positive and false negative missclassification cost here

fn_cost = 500
fp_cost = 300


# #### We will use the optimal model and its corresponding data set that was implemented in the GridSearchCV section. Let's first see the performance metrics of the trained model.

# In[106]:


# Use the most optimal machine learning model that you obtained from the GridSearchCV section and the corresponding data set you used (normal, RUS or ROS)

model_name = 'Random Forest - Random Oversampling'

rf = RandomForestClassifier(n_estimators = 150, max_depth = 7, class_weight = 'balanced', random_state = 123)
optimal_rf_model = rf.fit(X_train_ros, y_train_ros)

# Evaluating the accuracy of the training and validation sets

y_pred_train = optimal_rf_model.predict(X_train_ros)
y_pred_test = optimal_rf_model.predict(X_test_ros)

rf_train_acc = accuracy_score(y_train_ros, y_pred_train)
rf_test_acc = accuracy_score(y_test_ros, y_pred_test)

# Calculate the F1 score, Precision and Recall on the validation set

f_score = f1_score(y_test_ros, y_pred_test)
precision = precision_score(y_test_ros, y_pred_test)
recall = r2_score(y_test_ros, y_pred_test)

# creating a dataframe to compare the performance of different models
new_model_eval_data = [[model_name, tree_train_acc, tree_test_acc, f_score, precision, recall]]
new_evaluate_df = pd.DataFrame(new_model_eval_data, columns=['Model Name', 'Training Score', 'Testing Score',
                                          'F1 Score', 'Precision', 'Recall'])


# In[107]:


new_evaluate_df


# #### We now calculate the current misclassification cost in the validation set.

# In[108]:


# Obtain the count of false positive and false negative classifications from your model

cf = confusion_matrix(y_test, optimal_rf_model.predict(X_test)) # Matrix form of confusion matrix

fp_count = cf[0,1]
fn_count = cf[1,0]

# Calculate the total misclassification cost using the FN and FP cost and FN and FP count

misclassification_cost = fp_count * fp_cost + fn_count * fn_cost

print('Number of False Positives: %d' % fp_count)
print('Number of False Negatives: %d' % fn_count)
print('Prediction Misclassification Cost: %.2f' % misclassification_cost)


# #### We now calculate the misclassification cost as we raise the cut-off value from 0 to 1.

# In[109]:


# Predict probabilities for the training set and retain them for only positive outcomes
lr_probs_train = optimal_rf_model.predict_proba(X_train)[:, 1]

# Predict probabilities for the validation set and retain them for only positive outcomes
lr_probs_val = optimal_rf_model.predict_proba(X_test)[:, 1]


# In[110]:


# Calculate and store the misclassification costs for different values of cut-off probability
cost_train = []
cost_val=[]

for cutoff in np.arange(0, 1, 0.01):
    # Get the classification predictions using the probabilities obtained for the training data set and the cutoff
    # Get the false positive and false negative count from the predictions
    # Calculate the training misclassification cost and append it to the cost_train array
    curr_preds = np.where(lr_probs_train > cutoff, 1, 0)
    curr_cf = confusion_matrix(y_train, curr_preds)
    curr_fp_count = curr_cf[0, 1]
    curr_fn_count = curr_cf[1, 0]

    curr_misclassification_cost = curr_fp_count * fp_cost + curr_fn_count + fn_cost
    cost_train.append(curr_misclassification_cost)

    # Get the classification predictions using the probabilities obtained for the validation data set and the cutoff
    # Get the false positive and false negative count from the predictions
    # Calculate the training misclassification cost and append it to the cost_val array
    curr_preds = np.where(lr_probs_val > cutoff, 1, 0)
    curr_cf = confusion_matrix(y_test, curr_preds)
    curr_fp_count = curr_cf[0, 1]
    curr_fn_count = curr_cf[1, 0]

    curr_misclassification_cost = curr_fp_count * fp_cost + curr_fn_count + fn_cost
    cost_val.append(curr_misclassification_cost)


# Get the X values (cut-off values)
cutoffs = np.arange(0, 1, 0.01)

# Plot misclassification cost against cut-off value
plt.plot(cutoffs,cost_train, label='Training')
plt.plot(cutoffs,cost_val, label='Validaiton')
plt.xlabel('Cut-off')
plt.ylabel('Misclassification Cost')
plt.legend()
plt.show()

# Find the minimum misclassification cost and its associated cut-off value based on the training data
best_cost = min(cost_train)
best_cutoff = cutoffs[cost_train.index(best_cost)]

#apply the cut-off value to the validation data
best_valcost = cost_val[cost_train.index(best_cost)]


print('Best Misclassification Cost on the training is %.2f at Cut-off %.3f' % (best_cost, best_cutoff));
print('Applying that cut-off to the validation data results in Misclassification Cost of %.2f ' % best_valcost);


# In[111]:


# getting total time taken by the notebook to run

end_time = datetime.datetime.now()
total_time = end_time - start_time

print("Total time taken by notebook:", total_time)


# Checklist:
#  - Chose the optimal model and calculated the current misclassification cost in the validation set
#  - Calculated the misclassification cost for different values of cut-off value from 0 to 1
#  - Found the minimum misclassification cost and its associated best cut-off value based on the training data
#  - Applyied the same cut-off to the validation data
