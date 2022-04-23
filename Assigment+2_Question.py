#!/usr/bin/env python
# coding: utf-8

# <table align="center" width=100%>
#     <tr>
#         <td width="15%">
#             <img src="in_class.png">
#         </td>
#         <td>
#             <div align="center">
#                 <font color="#21618C" size=8px>
#                     <b> Assignment <br>(2)
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# ### About the data set  (Concrete_Data.csv data)
# 
# The data set contains information about the Cement strength and the different features that influence the strength of the cement.
# 
# Features
# 
# **cement :** amount of cement
# 
# **Blast :** amount of BlastFurnaceSlag
# 
# **Fly Ash :** amount of FlyAsh
# 
# **Water :**  amount of Water
# 
# **Superplasticizer :**  amount of Superplasticizer
# 
# **CA  :** amount of CoarseAggregate
# 
# **FA  :** amount of FineAggregate
# 
# **Age  :** age of concrete construction
# 
# Response
# 
# **CMS  :** Compressive strength of concrete construction

# ## Table of Content
# 
# 1. **[Import Libraries](#lib)**
# 2. **[Data Preparation](#prep)**
# 3. **[Model Building](#MB)**
# 4. **[Assumptions of Linear Regression](#AoLR)**
# 5. **[Feature Engineering](#FE)**

# <a id="lib"> </a>
# ## 1. Import Libraries

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# import various metrics from 'Scikit-learn' (sklearn)
from sklearn.model_selection import train_test_split

# 'Statsmodels' is used to build and analyze various statistical models
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_rainbow
import statsmodels.tsa.api as smt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson

# 'SciPy' is used to perform scientific computations
from scipy import stats
from scipy.stats import shapiro

# 'metrics' from sklearn is used for evaluating the model performance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# #### Load the Concrete_Data dataset and display the first five records 

# In[7]:


ctdata=pd.read_csv('Concrete_Data.csv')


# <a id="prep"> </a>
# ## 2. Data Preparation

# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>1. Check whether there is any missing value in the dataset.Adopt necessary steps to impute the missing values </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[8]:


ctdata.info()


# In[9]:


print("No missing data")


# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>2. Store the Dependant data seperately </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[11]:


depdata=ctdata[["Cement","Blast","Fly Ash","Water","Superplasticizer","CA","FA","Age"]]
depdata


# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>3. Check whether there is any outlier in the dataset.Adopt necessary steps to treat the outliers </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[16]:


fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for variable, subplot in zip(depdata.columns, ax.flatten()):
    z = sns.boxplot(x = depdata[variable], orient = "h",whis=1.5 , ax=subplot) # plot the boxplot
    z.set_xlabel(variable, fontsize = 20)  


# In[17]:


# obtain the first quartile
Q1 = depdata.quantile(0.25)

# obtain the third quartile
Q3 = depdata.quantile(0.75)

# obtain the IQR
IQR = Q3 - Q1

# print the IQR
print(IQR)


# In[18]:


depdata_iqr = depdata[~((depdata < (Q1 - 1.5 * IQR)) |(depdata > (Q3 + 1.5 * IQR))).any(axis=1)]
depdata_iqr.shape


# In[19]:


fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for variable, subplot in zip(depdata_iqr.columns, ax.flatten()):
    z = sns.boxplot(x = depdata_iqr[variable], orient = "h",whis=1.5 , ax=subplot) # plot the boxplot
    z.set_xlabel(variable, fontsize = 20)  


# <a id="MB"> </a>
# ## 3. Model Building

# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>4. Build the Linear regression model using OLS. See the F-stat, p-value and each feature p-value </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[23]:


fig, ax = plt.subplots(3, 3, figsize=(15, 8))
for variable, subplot in zip(ctdata.columns, ax.flatten()):
    z = sns.boxplot(x = ctdata[variable], orient = "h",whis=1.5 , ax=subplot) # plot the boxplot
    z.set_xlabel(variable, fontsize = 20) 


# In[24]:


ctdata_iqr = ctdata[~((ctdata < (Q1 - 1.5 * IQR)) |(ctdata > (Q3 + 1.5 * IQR))).any(axis=1)]
ctdata_iqr.shape


# In[26]:


feature_cols = ["Cement","Blast","Fly Ash","Water","Superplasticizer","CA","FA","Age"]
x = ctdata_iqr[feature_cols]
y = ctdata_iqr.CMS
lm = LinearRegression()
lm.fit(x,y)
print(lm.intercept_)
print(lm.coef_)


# In[28]:


# selecting independent variables that describe immunization
X = ctdata_iqr[["Cement","Blast","Fly Ash","Superplasticizer","Age"]]

# 'sm.add_constant' adds the intercept to the model
X = sm.add_constant(X)

# set the dependent variable
y = ctdata_iqr['CMS']

# building a model with an intercept
# fit() is used to fit the OLS model
MLR_model = sm.OLS(y,X).fit()

# print the model summary
print(MLR_model.summary())


# <a id="AoLR"> </a>
# ## 4. Assumptions of Linear Regression

# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>5. Check whether there is Multi-collinarity in the dataset.Adopt necessary steps to treat the Multi-collinarity </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>6. Build a linear Regression model on the dataset after removing multicollinearity </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>7. Check normality assumption for residue and if it is violoated, do the transformation on output variable </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>8. Check whether there is any Autocorreation in the dataset.Adopt necessary steps to treat the Autocorreation </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>9. Check whether there is any homoscedasticity.Adopt necessary steps to treat the homoscedasticity </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <a id="FE"> </a>
# ## 5. Feature Engineering

# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>10. Split the data into Train & Test data with a ratio of 70:30 </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>11. Build a Linear Regression model on the dataset and evaluate its performace using R-Square & Mean Square Error. </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>12. Perform Backward elimination & identify the most suitable set of variables </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>13. Perform Forward Selection & identify the most suitable set of variables </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>14. Find the best set of significant variables among all the possible subsets of the variables. Build a linear regression model using the best subset and find the R-Squared value for that model </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# <table align="left">
#     <tr>
#         <td width="6%">
#             <img src="question_icon.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>15. Analyse, which model among Forward Selection , Backward Elimination & Recursive Feature Elimination model performs the best </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:




