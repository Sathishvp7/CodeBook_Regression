####################### MACHINE LEARNING PYTHON CODEBOOK #############################

# 1. Import Basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to view all columns (max 100) in the dataframe
pd.set_option('display.max_columns', 100)
# to view 100 rows (max without scroll) in the dataframe or other print statements
pd.set_option('display.max_rows', 100)

# 2. Load a Dataset
# Here we going to load Housing Dataset
df = pd.read_csv('train.csv')

# Description about this Dataset, 
'''Here we going to predict price of the house using the 
various Components like Area, size, etc..'''

# 3. Basic inspect 
df.head()
shape_df = df.shape  # to get shape of Dataset
df.dtypes # to get Data type of each column
info_df = df.info() # Information like Datatype number of Null values
describe = df.describe()

# Count the Numbe of int,float,Object columns in the dataset
count_dtypes = df.dtypes.value_counts()

# Note 1  - For a models if input is in Numeric it will learn better

# Now we going to find PMF value
from empiricaldist import Pmf,Cdf
#pmf - probablity Distibution function - Probablity of particular Variable value.
# cdf - Cummulative Disribution Function -  Sum of all possible probablity 
sp = df.SalePrice
Pmf_SalePrice = pd.DataFrame(data= {'Probablity_Mass_Function': Pmf.from_seq(sp),
                                    'Cummulative_Mass_Function' : Cdf.from_seq(sp)},
                                    index= sp).sort_values(['SalePrice'])

#Visulazisation of cdf
#Note 2
'''CDF helps to understand how may precent of the total data 
is below or above a specified threshold'''
cdf = Cdf.from_seq(sp)
cdf.plot()

# 4. DATA WRANGLING 
'''
Inspecting missing values in each variables and trying to impute statistically acceptable values.
Detect outliers and remove those records.
Remove irrelevant records. Ex. Records with negative age etc
'''
# 4.1 Handling Missing values in the Dataset
total =  df.isnull().sum().sort_values(ascending = False)
percentage =  (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

# Now we going replace Null value 
#1. poolqc
df.PoolQC.value_counts()
df.PoolQC.replace(np.nan,'None',inplace=True)
df.PoolQC.value_counts()

#2. MisFeature
df.MiscFeature.value_counts()
df.MiscFeature.replace(np.nan,'None',inplace=True)

#3. Alley
df.Alley.value_counts()
df.Alley.replace(np.nan,'None',inplace=True)

#4. Fence
df.Fence.value_counts()
df.Fence.replace(np.nan,'None', inplace= True)

# 5. Through For loop also we can do and save lot of time
missing_list = ['FireplaceQu', 
                  'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual',
                  'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 
                  'BsmtQual']
for i in missing_list:
    x= df[i]
    x.replace(np.nan,'None',inplace=True)

total =  df.isnull().sum().sort_values(ascending = False)
percentage =  (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

#Remaining we have LotFrontage, GarageYrBlt, MasvnrType, MasvnrArea, Electrical
Remaining_list = ['LotFrontage', 'GarageYrBlt', 'MasVnrType',
                  'MasVnrArea', 'Electrical']

for i in Remaining_list:
    print((i+ ' ' +   str(df[i].dtype)))

# LotFrontage has 259 Null values,we have  2 options
# Option 1 - Remove the row/column
# Option 2 - Replace the good value.
'''We going for Option 2, so we going to predict the value.for that we need to 
find correlation, which Column is correlated with LotFrontage.'''

Corr_LotFrontage = df.corr()['LotFrontage'].sort_values(ascending = False)
# Note if Correlation is greater than 0.5 then we good to select ,
# Here we dont have 0.5 value, Learning Purpose we going to find Prediction.

# We going to predict using Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# step 1- Select Non -  Null values
df_line = df[['LotArea','LotFrontage']].dropna()
len(df_line)

# step 2- remove Outliers
_75_percentile = np.percentile(df_line['LotArea'],75)
_25_percentile = np.percentile(df_line['LotArea'],25)
#InterQuartileRange - iqr
iqr = _75_percentile - _25_percentile
df_line.drop(df_line[df_line['LotArea'] < _25_percentile - (1.5 * iqr)].index,inplace=True)
df_line.drop(df_line[df_line['LotArea'] > _75_percentile + (1.5 * iqr)].index,inplace=True)


lm.fit(df_line[['LotArea']].values,df_line.LotFrontage.values)

type(df_line[['LotArea']])== type(df_line.LotArea)

#Visualizsation
plt.scatter(df_line.LotArea, df_line.LotFrontage, color='blue')


#get Lot Area row value which has lotFrontage as Null
test  = df[df.LotFrontage.isnull()]
len(test)
LotArea_test = test[['LotArea']]
len(LotArea_test) # we have 259.
predict = lm.predict(LotArea_test)

predicted = pd.DataFrame(data = {'LotArea': test.LotArea.values,
                                 'LotFrontage' : [round(i,0) for i in predict]})
predicted.sort_values(['LotArea']).head()

predicted = predicted.sort_values(['LotArea']).reset_index(drop=True)
test  = df[df.LotFrontage.isnull()].reset_index(drop=True)

df1 = pd.merge(predicted,test,left_index =True,right_index=True)
df1.rename(columns={'LotArea_x':'LotArea', 'LotFrontage_x':'LotFrontage'}, inplace=True)
df1.drop(['LotArea_y', 'LotFrontage_y'], axis=1, inplace=True)

# Drop Null LotFrontage rows
df.dropna(subset=['LotFrontage'],inplace=True)
len(df.LotFrontage)
df = pd.concat([df,df1],sort=True)

df.isnull().sum().sort_values(ascending=False).head()

'''
Still we have below missing values, MasVnrType,MasVnrArea, Electrical so we going to remove
GarageYrBlt    81
MasVnrType      8
MasVnrArea      8
Electrical      1
Alley           0
'''
remove_rows = ['MasVnrType','MasVnrArea', 'Electrical']
for x in remove_rows:
    df.dropna(subset = [x],inplace=True)
    
#GarageYrBlt - Replace by min year
gar_min = df.GarageYrBlt.min() # 1900
df.GarageYrBlt.replace(np.nan,gar_min,inplace=True)
    
df.isnull().sum().sort_values(ascending=False).head()
''' Now we clean the Dataset , now we ready to apply model.
YrSold         0
HalfBath       0
ExterCond      0
ExterQual      0
Exterior1st    0
'''
#############################################################################################
'''
Alternative approach to fill missing values is by using the imputer. 
We aren't using it here though.'''
#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values=np.nan, strategy='most_frequent')
#imp.fit_transform(df)
#############################################################################################
# Now we going to Remove Outlier for Entier Dataset
numeric_column = df.select_dtypes(include = [np.number]).columns
df_reg = df

for column in numeric_column:
    before = df_reg.shape[0]
    _75th = np.percentile(df_reg[column],75)
    _25th = np.percentile(df_reg[column],25)
    iqr = _75th - _25th
    df_reg.drop(df_reg[df_reg[column] < _25th - (iqr * 1.5)].index,inplace= True)
    df_reg.drop(df_reg[df_reg[column] > _75th + (iqr * 1.5)].index,inplace= True)
    count  = str(before -  df_reg.shape[0])
    print('Numebr of Outliers removed in the column "{0}" : {1}'.format(before,count))
    
df_reg.shape  


# 5. Exploratory Data Analysis
'''
Now complete cleaning has been done, now we need to select the better column.
for that we going to use correlation method
Why Correltion, if one column is corrlelated more with another column
we can drop that column , because there will no much improvement, 
using this method we can save Memory, model running time.
'''
df_corr = df_reg.corr()
c = df_corr.abs() # ads convert everything in to postive
s = c.unstack()
so = pd.DataFrame(s.sort_values(kind = 'quciksort', ascending = False))
so.head()
so = so.reset_index()
so.rename(columns = {'level_0': 'col1','level_1': 'col2', 0:'Correlation'},inplace=True)

#Just Remove the col1 and col2 have same value
so = so[so.col1 != so.col2]
so['Order_cols'] = so.apply(lambda row : '-'.join(sorted([row.col1,row.col2])),axis=1)
so.shape
so.drop_duplicates(['Order_cols'],inplace= True)    
so.reset_index(drop= True,inplace=True)
so.drop(['Order_cols'],axis = 1, inplace= True)


# Getting Object Datatype to convert in to Numeric using get Dummies
col_object = df_reg.select_dtypes(include = ['object']).columns
print(col_object)

for col in col_object:
    print(col)
    df_main = pd.get_dummies(data=df_reg,drop_first=True)
    
df_main.info()

########################################################################################
##################### MODEL SELECTION ##################################################

X = df_main.drop('SalePrice',axis =1)
Y = df_main['SalePrice']

# 6. Train Test split

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state = 123)

#########################################################################################
#################     8. LINEAR MODEL   #################################################
#Linear Regression - Predicting a line where we choose the slope and intercept in such a way that reduce the error (loss or cost) function.
#Linear Regression is done by defining an error function and choosing a line which minimizes the error function.
#Our predicted line should be close to the actual data points as possible. So we minimize the vertical distance (residuals) between the fit and data.
#Loss function = Sum of square of residuals. OLS (Ordinary Least Squares)
#Same as minimizing the mean squared error of prediction on traning dataset.
#R square = Measure of variance in the target variable that is predicted from the independent variables.


# =============================================================================
# 8.1 Linear Regression Assumption
# Linear relationship - Independent variables vs Dependent variable.
# Little or no multicollinearity - Independent variables should not be related. Why removing multicollonearity , to avoid Over fitting
# Tested using correlation matrix, tolerance & variance inflation factor.
#  VIF < 10; Tolerance = 1 / VIF > 0.1.


# 
# =============================================================================

#Main Functions to remove MultiCollinerity
#1. Correlation ,  cool = df.corr()['SalePrice']
#2. VIF - variance Inflation - 8.2
#3. Tolerance 

# =============================================================================
# Correlation
#     Pearson Correlation
# If Correlation is highle correlated (>0.5) that is recommended Column

# Step 1 - Find Correlataion .
# Step 2 - Select Correlation >0.5
# Step 3 - Now u have Best Column, Here we need to find Correlation between selected Col
# Step 4 - Now u have Best column for this model - is called Person Correlation


# =============================================================================

###############     8.2 VIF - variance Inflation     ###############################################
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

sfs = pd.read_csv('train.csv')
sfs = sfs.loc[:,sfs.dtypes == 'int64'].dropna()
sfs.head()
XXX = add_constant(sfs)
pd.DataFrame([variance_inflation_factor(XXX.values, i) 
               for i in range(XXX.shape[1])], 
              index=XXX.columns)

# 
# =============================================================================
# ##################################################################################################
# 
# # =============================================================================
# # Plot Details
# Noramal Probability Plot :
#     Differences between the value from the mean of all values
#     Error Term = Residual = Y_pred - y_test
#     We can arrange Error term and find Mean and Standard Devivation - Plot in X axix
#     Error value in Y- Axis
# # =============================================================================
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lm = LinearRegression()
lm.fit(train_x,train_y)
predict = lm.predict(test_x)
print(r2_score(test_y,predict))  # r2 score 3.5 so very bad score



# =============================================================================
# ############################   FEATURE SELECTION ###############################
#     #https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# 1. Filter Method - Pearson Correlation
# 2. Wrapper Method -  Backward Elimination , RFE.
# 3. Embedded Method
# =============================================================================
# FILTER METHOD - Find Correlation and Remove remaining.
    # Step 1 - Find Correlation 
sal_corr = df_main.corr()['SalePrice']
sal_corr = sal_corr[sal_corr>0.5] # Correlation value min >0.5

# Wrapper Method
    # Backward Elimination - Name itself we can , First we need to apply one Machine Learning model,
    #Eg - Random Forest with all column and see the performance and iteratively remove the col
    #which leads to wrost performances
    
# Now we going to Ordinary Least Square method - Linear Model
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)

# Fitting  Model
model = sm.OLS(Y,X_1).fit()
model.pvalues # P value is 0.5 or Greater remove those column, we can done using for loop

col = list(X.columns)
#len(col) # Starting We have 114 Column
pmax = 1

while (len(col) >0):
    p =[]
    XX = X[col]
    X_1 = sm.add_constant(XX.to_numpy())
    model = sm.OLS(Y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = col[1:])
    p_max = max(p)
    feature_with_p_max = p.idxmax()
    if(p_max > 0.5):
        col.remove(feature_with_p_max)
    else:
        break
    
selected_feature = col
len(selected_feature) # Now we have 84 Column


# =============================================================================
# # RFE - Recuressive Feature Elimination 
'''
The Recursive Feature Elimination (RFE) method works by recursively removing attributes 
and building a model on those attributes that remain
'''
# =============================================================================


from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,Y)  
#Fitting the data to model
model.fit(X_rfe,Y)
print(rfe.support_)
print(rfe.ranking_)

len(X.columns[0]) # Column Count 114

#In above 7 column is random selection to give accurate result , follow below code.

#no of features
nof_list=np.arange(1,114)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof) # O/p Feature col - 3
print("Score with %d features: %f" % (nof, high_score)) # Model Score 0.319 

cols = list(X.columns)
model = LinearRegression()

#Initializse
rfe = RFE(model,3)
#Fit the RFE
X_rfe = rfe.fit_transform(X,Y)
#Fit the model
model.fit(X_rfe,Y)
print(rfe.support_) # True being relevant feature and False being irrelevant feature.
print(rfe.ranking_) # Under which rank we need to use that col

    
# for Above Optimum Feature is gives 3 column with Score 0.319, for this model Linear Regression,
# is not giving good result, but and template is same we try same thing with better model.

########################## IMPORTANT NOTE ##################################################

#For Better model Slection, we created our own model selection Libirary 
'''
https://pypi.org/project/model-performance-investigator/1.0.1/
'''
############################################################################################

# Before jump in to Embedded we can see Cross validation, main topic for Embedded
# =============================================================================
#                   Multi-fold Cross Validation
'''
https://medium.com/datadriveninvestor/k-fold-cross-validation-for-parameter-tuning-75b6cb3214f
'''
# 
# The data set is divided into k subsets
# Each time, one of the k subsets is used as the validation set and the other k 1 subsets are put together to form a training set.
# Then the average performance across all k trials is computed.
# The advantage of this method is that it matters less how the data gets divided.
# Every data point gets to be in a test set exactly once, and gets to be in a training set k-1 times.

# Compute 5-fold cross-validation scores: cv_scores
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(reg, X, Y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

O/P - [ 0.03497832 -0.3682355   0.38581693  0.54463125 -0.62975518]
      Average 5-Fold CV Score: -0.0065128342056527625
# ============================================================================= 

# =============================================================================
# # Embedded Method
'''
Embedded methods are iterative in a sense that takes care of each iteration of the model training process and 
carefully extract those features which contribute the most to the training for a particular iteration. 
Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.
'''
# =============================================================================

#Now going to use Lazzo Regularisation
'''
If the feature is irrelevant, lasso penalizes it’s coefficient and make it 0. 
Hence the features with coefficient = 0 are removed and the rest are taken.

Note -  lasso run behind the Alpha, when Alpha = 0 , lasso produce same coefficient as linear regression
if Alpha is very very large that make coefficient zero
'''
reg = LassoCV()
reg.fit(X,Y)
print('Best alpha value {}'.format(reg.alpha_)) 
print('Best Score using Lasso {}'.format(reg.score)) # Best alpha value , parameter
coeff = pd.Series(reg.coef_,index= X.columns)

print('Lasso Picked column {} and Lasso Eliminated Coloumn {}'.
      format(str(sum(coeff!=0)),str(sum((coeff == 0)))))

#
'''
Lasso Picked column 13 and Lasso Eliminated Coloumn 101
'''
#Visulaizsation
imp_coef = coeff.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.tight_layout

# Which Feature Selection is better,
# Filter - Less accurate
# Wrapper and Embedded are more accurate, if Column is max(`25 ) we can use these methoda

##############################################################################


#

# =============================================================================
# Differences Beteween Lasso vs Ridge Regularizsation 
# =============================================================================
''''
Ridge regression is that it enforces the β coefficients to be lower, but it does not enforce them to be zero. 
That is, it will not get rid of irrelevant features but rather minimize their impact on the trained model.

Short Definition - Lasso Make irrelevant feature as Zero Ridge doesn't
'''

# ELASTIC NET (Combination of both Ridge and Lasso)
'''
Lasso used the L1 penalty to regularize, while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net. 
In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:
                                                a∗L1+b∗L2
'''
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5, n_jobs=-1)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

o/p - Tuned ElasticNet l1 ratio: {'l1_ratio': 0.9655172413793103}
      Tuned ElasticNet R squared: 0.7383666469811819
      Tuned ElasticNet MSE: 224116037.769474
# =============================================================================
# LOSS FUNCTION
'''
Mean Square Error (MSE), Quadratic loss, L2 Loss : sum of squared distances between our target variable and predicted values.

Mean Absolute Error (MAE), L1 Loss : sum of absolute differences between our target and predicted variables. So it measures the average magnitude of errors in a set of predictions,
without considering their directions. (If we consider directions also, that would be called Mean Bias Error (MBE), which is a sum of residuals/errors).

In short, using the squared error is easier to solve, but using the absolute error is more robust to outliers. 
Model with MSE loss give more weight to outliers than a model with MAE loss. MAE loss is useful if the training data is corrupted with outliers.

Least absolute deviations(L1) and Least square errors(L2) are the two standard loss functions, 
that decides what function should be minimized while learning from a dataset.

L1 - Best option for if u dataset having Outlier and u want those outlier to train.
L2 - Best option if your option doesnt have outlier.

Required Package
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tools.eval_measures import rmse

Code:
mod = GradientBoostingRegressor(loss='lad') #lad- Least Absolute Devivation
mod.fit(train_x,train_y)
predict = mod.predict(test_x)
rmse(predict, test_y) # Higher value higher differences between Predicted and target

mod = GradientBoostingRegressor(loss='ls') #ls- Least Square error
mod.fit(train_x,train_y)
predict = mod.predict(test_x)
rmse(predict, test_y)

For More Datails - https://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/

'''

# =============================================================================


# =============================================================================
# DECISION TREEE REGRESSOR
#Decision Tree Regressor Provides greater flexibility in plotting the regression line. It is not linear. Decison Tree don't require feature scaling
# =============================================================================

from sklearn.tree import DecisionTreeRegressor #  Note Decision tree Classifer is Ginary Classfication use auc - Area under Curve is best metrics to evaluate
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
                           min_samples_leaf=0.13,
                           random_state=123)

params_dt = {
    'max_depth': [3, 4,5, 6], # Maximium Depth of tree, root rode is max depth 0 (Longest bath from Root to Leaf) Max 32 we can give 
    'min_samples_leaf': [0.04, 0.06, 0.08],
    'max_features': [0.2, 0.4,0.6, 0.8] #
}

grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)

grid_dt.best_params_ # to see Best Parameter selection
# Fit 'dt' to the training-set
grid_dt.fit(train_x, train_y)

# Predict test-set labels
y_pred = grid_dt.predict(test_x)
    
# Compute test-set MSE
mse_dt =  MSE(test_y, y_pred)
# Compute test-set RMSE 
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print(rmse_dt)































































