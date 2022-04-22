#!/usr/bin/env python
# coding: utf-8

# # SLR MID EXAM

# ### Data_set: 
# Given are the variable name, variable type, the measurement unit and a brief description. The concrete compressive strength is the regression problem. The order of this listing corresponds to the order of numerals along the rows of the database.

# Attribute Information:
# 
# 
# 
# Name -- Data Type -- Measurement -- Description
# 
#     Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
#     Age -- quantitative -- Day (1~365) -- Input Variable
#     Concrete compressive strength -- quantitative -- MPa -- Output Variable

# In[26]:


# Kindly change the below cells from markdown to code and execute it 

import pandas as pd

import csv

#with open("concrete.csv","r")as file:
#  reader=csv.reader(file)
df=pd.read_csv("data_set.csv")

df.head()
# In[27]:


import pandas as pd
import matplotlib.pyplot as plt


# #### 1.	Data Understanding (5 marks)
# 
# 
# a.	Read the dataset (tab, csv, xls, txt, inbuilt dataset). What are the number of rows and no. of cols & types of variables (continuous, categorical etc.)? (1 MARK)
# 
# b.	Calculate five-point summary for numerical variables (1 MARK)
# 
# c.	Summarize observations for categorical variables – no. of categories, % observations in each category. (1 marks)
# 
# d.	Check for defects in the data such as missing values, null, outliers, etc. (2 marks)

# In[28]:


df=pd.read_csv("data_set.csv")
df.head()


# In[29]:


print("number of rows:", df.shape[0])
print("number of columns:", df.shape[1])


# In[30]:


df.info()


# In[31]:


df.dtypes


# In[32]:


print('there are no categorical variable.')


# In[33]:


#Check for defects in the data such as missing values, null, outliers, etc.
df.isnull().sum()


# In[34]:


column_list=df.columns.tolist()
column_list


# In[35]:


df.describe().T


# In[36]:


df.boxplot()
plt.show()


# In[37]:


#Outlier Treatment
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df < (Q1-1.5*IQR))| (df > (Q3+1.5*IQR) )).any(axis=1)]
df.shape


# In[38]:


df


# #### 2.	Data Preparation (15 marks)
# 
# a.	Fix the defects found above and do appropriate treatment if any. (5 marks)
# 
# b.	Visualize the data using relevant plots. Find out the variables which are highly correlated with target variable? (5 marks)
# 
# c.	Do you want to exclude some variables from the model based on this analysis? What other actions will you take? (2 marks)
# 
# d.	Split dataset into train and test (70:30). Are both train and test representative of the overall data? How would you ascertain this statistically? (3 marks)
# 
#  

# In[42]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[44]:


#Fix the defects found above and do appropriate treatment
df_feature=df[column_list[:-1]]
df_feature


# In[51]:


vif = pd.DataFrame()
vif["VIF_Factor"] = [variance_inflation_factor(df_feature.values,i) for i in range(df_feature.shape[1])]
vif["Features"] = df_feature.columns
vif.sort_values('VIF_Factor',ascending=False).reset_index(drop=True)
vif


# In[52]:


#we are removing features with VIF>10
for i in range(len(df_feature.columns)):
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(df_feature.values,i) for i in range(df_feature.shape[1])]
    vif["Features"] = df_feature.columns
    multi = vif[vif.VIF_Factor > 10]
    if (multi.empty == False):
        df_sorted = multi.sort_values(by='VIF_Factor',ascending=False)
    else:
        print(vif)
        break
    if (df_sorted.empty==False):
        df_feature = df_feature.drop(df_sorted.Features.iloc[0],axis=1)
    else:
        print(vif)


# In[55]:


df_feature.columns


# In[57]:


import seaborn as sns
sns.pairplot(df)
plt.show()


# In[62]:


df.columns


# In[66]:


#Visualize the data using relevant plots. Find out the variables which are highly correlated with target variable
fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(20,20))
for i,subplot in zip(df.columns[:-1],ax.flatten()):
    df.plot.scatter(x=i,y='csMPa',ax=subplot)
plt.show()


# In[69]:


plt.title("Corelation Heatmap")
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[87]:


print("Based on corelation:")
print("cement,superplaster and age are highly corelated with target variable.")
print("water is high negatively corelated with target variable")


# In[86]:


#Do you want to exclude some variables from the model based on this analysis? What other actions will you take
print("based on above we can finalize:")
print("""all predictors are showing somesort of linearity so we are going to include all, 
So out final featture variables are:\n""",df.columns.tolist(),end="")


# In[81]:


from sklearn.model_selection import train_test_split


# In[84]:


#Split dataset into train and test (70:30). Are both train and test representative of the overall data
X= df[df_feature.columns]
y=df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.30)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 
# ### 3.	Model Building (20 marks)
# 
# a.	Fit a base model and observe the overall R- Squared, RMSE and MAPE values of the model. Please comment on whether it is good or not.  (5 marks)
# 
# b.	Check for multi-collinearity and treat the same. (3 marks)
# 
# c.	How would you improve the model? Write clearly the changes that you will make before re-fitting the model. Fit the final model.   (6 marks)
# 
# d.	Write down a business interpretation/explanation of the model – which variables are affecting the target the most and explain the relationship. Feel free to use charts or graphs to explain. (4 marks) 
# 
# e.	What changes from the base model had the most effect on model performance? (2 marks)
# 

# In[89]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,LeaveOneOut,cross_val_score,KFold
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
import numpy as np


# In[99]:


def get_train_rmse(model):
    train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train,train_pred)
    rmse_train = round(np.sqrt(mse_train),4)
    return rmse_train

def get_test_rmse(model):
    test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test,test_pred)
    rmse_test = round(np.sqrt(mse_test),4)
    return rmse_test

def mape(actual,predicted):
    return (np.mean(np.abs((actual-predicted)/actual))*100)

def get_test_mape(model):
    test_pred= model.predict(X_test)
    mape_test = mape(y_test,test_pred)
    return mape_test

def get_score(model):
    r_sq= model.score(X_train,y_train)
    n = X_train.shape[0]
    k = X_train.shape[1]
    r_sq_adj = 1 - ((1-r_sq)*(n-1)/(n-k-1))
    return ([r_sq, r_sq_adj])

score_card = pd.DataFrame(columns=['Model_Name','Alpha(whenever required)','l1_ratio','R-Suqared','Adj. R-Squared',"Test_RSME","Test_Mape"])
def update_score_card(algorithm_name,model,alpha='-',l1_ration='-'):
    global score_card
    score_card = score_card.append({
        'Model_Name': algorithm_name,
        'Alpha(whenever required)': alpha,
        'l1_ratio': l1_ration,
        'R-Suqared': get_score(model)[0],
        'Adj. R-Squared': get_score(model)[1],
        "Test_RSME": get_test_rmse(model),
        "Test_Mape": get_test_mape(model)
    },ignore_index=True)

def plot_coefficient(model,algorithm_name):
    df_coeff = pd.DataFrame({'Variable':X.columns,'Coefficient': model.coef_})
    sorted_coeff = df_coeff.sort_values('Coefficient',ascending=False)
    sns.barplot(x="Coefficient", y="Variable",data=sorted_coeff)
    plt.xlabel("Coefficient from {}".format(algorithm_name),fontsize=15)
    plt.ylabel('Features',fontsize=15)


# In[100]:


#Fit a base model and observe the overall R- Squared, RMSE and MAPE values of the model
linreg = LinearRegression()
MLR_model = linreg.fit(X_train, y_train)
MLR_model.score(X_train,y_train)


# In[102]:


print("RSME on train set: ", get_train_rmse(MLR_model))
print("RSME on test set: ", get_test_rmse(MLR_model))
difference = abs(get_test_rmse(MLR_model)-get_train_rmse(MLR_model))
print("Difference between RMSE on Train and test set: ", difference)


# In[103]:


update_score_card(algorithm_name="Linear Reagression", model=MLR_model)


# In[ ]:





# In[ ]:





# In[104]:


#Check for multi-collinearity and treat the same
#we are removing features with VIF>10
for i in range(len(X_train.columns)):
    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])]
    vif["Features"] = df_feature.columns
    multi = vif[vif.VIF_Factor > 10]
    if (multi.empty == False):
        df_sorted = multi.sort_values(by='VIF_Factor',ascending=False)
    else:
        print(vif)
        break
    if (df_sorted.empty==False):
        df_feature = df_feature.drop(df_sorted.Features.iloc[0],axis=1)
    else:
        print(vif)


# In[107]:


print("Based on VIF < 10, we are keeping above predictors")
X_train = X_train[["cement","slag","flyash","superplasticizer","age"]]
X_test = X_test[["cement","slag","flyash","superplasticizer","age"]]


# In[108]:


linreg = LinearRegression()
MLR_model1 = linreg.fit(X_train, y_train)
MLR_model1.score(X_train,y_train)
print("RSME on train set: ", get_train_rmse(MLR_model1))
print("RSME on test set: ", get_test_rmse(MLR_model1))
difference = abs(get_test_rmse(MLR_model1)-get_train_rmse(MLR_model1))
print("Difference between RMSE on Train and test set: ", difference)


# In[109]:


update_score_card(algorithm_name="Linear Reagression after removing colinearity", model=MLR_model1)


# In[ ]:





# In[ ]:


#How would you improve the model? Write clearly the changes that you will make before re-fitting the model. Fit the final model


# In[ ]:


print("with regularization we can expect improved score")


# In[111]:


ridge = Ridge(alpha = 1, max_iter =500)
ridge.fit(X_train,y_train)
print("RMSE on test set: ", get_test_rmse(ridge))


# In[112]:


update_score_card(algorithm_name="Ridge Regression with alpha=1", model=ridge,alpha=1)
score_card


# In[113]:


ridge = Ridge(alpha = 2, max_iter =500)
ridge.fit(X_train,y_train)
print("RMSE on test set: ", get_test_rmse(ridge))


# In[114]:


update_score_card(algorithm_name="Ridge Regression with alpha=2", model=ridge,alpha=2)
score_card


# In[115]:


lasso = Lasso(alpha = 0.05, max_iter =500)
lasso.fit(X_train,y_train)
print("RMSE on test set: ", get_test_rmse(lasso))


# In[ ]:





# In[116]:


update_score_card(algorithm_name="lasso Regression with alpha=0.05", model=lasso,alpha=0.05)
score_card


# In[118]:


#Write down a business interpretation/explanation of the model – which variables are affecting the target the most and explain the relationship. Feel free to use charts or graphs to explain
print("All models are performing same, as there is minor change in Test_RSME")
print("We can conclude that Lasso regression line we obtained is best fit.")
print("Intercept: ",lasso.intercept_)
print("coefficient: ",lasso.coef_)


# In[119]:


#What changes from the base model had the most effect on model performance
print("using lasso regression was most effective!")


# In[ ]:




