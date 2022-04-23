#!/usr/bin/env python
# coding: utf-8

# ## Assignment - 1 - Data Preprocessing

# HR_attrition.csv is a dataset of IBM employee attrition data. Each row is personal and attrition information of an employee. Our project is focusing on using machine learning to predict attrition. We are also providing interesting findings that people usually don't think of regarding employee attrition.

# Attrition: Whether employees are still with the company or whether they’ve gone to work somewhere else.
# 
# Age: 18 to 60 years old
# 
# Gender: Female or Male
# 
# Department: Research & Development, Sales, Human Resources.
# 
# BusinessTravel: Travel_Rarely, Travel_Frequently, Non-Travel.
# 
# DistanceFromHome: Distance between the company and their home in miles.
# 
# MonthlyIncome: Employees' numeric monthly income.
# 
# MaritalStatus: Married, Single, Divorced.
# 
# Education: 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'.
# 
# EducationField: Life Sciences， Medical， Marketing，Technical Degree，Other.
# 
# EnvironmentSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'.
# 
# RelationshipSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'.
# 
# JobInvolvement: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'.
# 
# JobRole: Sales Executive，Research Science, Laboratory Tec, Manufacturing, Healthcare Rep, etc
# 
# JobSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'.
# 
# OverTime: Whether they work overtime or not.
# 
# NumCompaniesWorked: Number of companies they worked for before joinging IBM.
# 
# PerformanceRating: 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'.
# 
# YearsAtCompany: Years they worked for IBM.
# 
# WorkLifeBalance: 1 'Bad' 2 'Good' 3 'Better' 4 'Best'.
# 
# YearsSinceLastPromotion: Years passed since their last promotion.

# ### Import the Libraries and load the dataset

# In[228]:


import pandas as pd
import numpy as np


# In[229]:


hrdata=pd.read_csv('HR_attrition.csv')
hrdata.head(2)


# In[230]:


hrdata.info()


# In[231]:


hrdata.describe(include='object')


# ### Keep the numerical data and categorical data seperately in the variable num_data and cat_data

# In[232]:


num_data = hrdata.select_dtypes(exclude='object')
cat_data = hrdata.select_dtypes(include='object')
cat_data


# ### Print all the unique levels present in the each categorical columns and if required perform  text cleaning 

# In[233]:


for col in cat_data.columns:
    print(f"{col}:", end="\n\t=> ")
    print(cat_data[col].unique())


# In[234]:


cat_data["Attrition"].replace({"yes": "Yes", "no": "No"}, inplace=True)


# In[235]:


for col in cat_data.columns:
    print(f"{col}:", end="\n\t=> ")
    print(cat_data[col].unique())


# ### Print thuniquessing value percentage in each columns in the data

# In[236]:


for col in cat_data.columns:
    print(f"{col}:")#, end="\n\t=> ")
    items = cat_data[col].unique()
    count = cat_data[col].count()
    for item in items:
        print(f"\t=> {item}: {cat_data[cat_data[col]==item][col].count()*100/count:.2f}%")#,cat_data[cat_data[col]==item][col].count()*100/count)
    
    #print(cat_data[col].unique())


# ### Treat the missing value in the age colum using Joblevel group means

# In[237]:


#num_data.Age = num_data.Age.fillna(num_data['JobLevel'].mean())
a = num_data.groupby(num_data["JobLevel"]).mean()["Age"]
rep = dict(a)
index = np.where(num_data.Age.isna())[0]


# In[238]:



#num_data.Age[num_data.Age.isna()==True]
for ind in index: 
    num_data.loc[ind,"Age"] = rep[num_data.loc[ind,"JobLevel"]]
num_data.Age.isna().sum()


# ### Treat the missing value in the MonthlyIncome colum using Joblevel group means

# In[239]:


a = num_data.groupby(num_data["JobLevel"]).mean()["MonthlyIncome"]
rep = dict(a)
index = np.where(num_data.MonthlyIncome.isna())[0]
for ind in index: 
    num_data.loc[ind,"MonthlyIncome"] = rep[num_data.loc[ind,"JobLevel"]]
num_data.MonthlyIncome.isna().sum()


# ### Treat the missing value in the Bussiness_Travel colum using Mode of the same column

# In[240]:


cat_data.BusinessTravel.replace(np.NaN,cat_data.BusinessTravel.mode()[0] ,inplace = True)


# In[241]:


cat_data.BusinessTravel.unique()


# ### Will you prefer 'mode' imputation for Department column ? If not use other logical approach to fill the missing values in the department column

# In[242]:


print(cat_data.Department.unique())
print(cat_data.EducationField.unique())
cat_data.Department.isna().sum()


# In[243]:


def replace_missing(field='Life Sciences'):
    di=dict(cat_data[cat_data.EducationField==field].groupby(cat_data.Department).count()["EducationField"])
    rev_di = {value:key for (key,value) in di.items()}
    return rev_di.get(max(rev_di))


# In[244]:


for edu in cat_data.EducationField.unique():
    index = np.where(cat_data.EducationField == edu)[0]
    for ind in index: 
        cat_data.loc[ind,"Department"] = replace_missing(edu)
cat_data.Department.isna().sum()


# ### Perform a suitable missing value treatment technique for Distance from home column

# In[245]:


print(num_data.DistanceFromHome.isnull().sum())
a = num_data.groupby(num_data["JobLevel"]).mean()["DistanceFromHome"]
rep = dict(a)
index = np.where(num_data.DistanceFromHome.isna())[0]
for ind in index: 
    num_data.loc[ind,"DistanceFromHome"] = rep[num_data.loc[ind,"JobLevel"]]
num_data.DistanceFromHome.isna().sum()


# ### Fill the missing value in JobSatisfaction with 'mode' imputation (as it is having few missing values)

# In[246]:


print(num_data.JobSatisfaction.isna().sum())
num_data.JobSatisfaction.replace(np.NaN,num_data.JobSatisfaction.mode()[0] ,inplace = True)
print(num_data.JobSatisfaction.isna().sum())


# ### Perform standard scaler on MonthlyIncome and Age column in this dataset

# In[247]:


"""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
scaled_MonthlyIncome = scaler.fit_transform(num_data.MonthlyIncome)
scaled_Age = scaler.fit_transform(num_data.Age)"""


# ### Encode the Attrition column with appropriate technique (Note: Attrition is a target column)

# In[248]:


"""from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder()
df_encode= pd.DataFrame(encode.fit_transform(cat_data.Attrition).toarray()),columns=['Attrition_Yes',"Attrition_No"])
cat_data = pd.concat([cat_data,df_encode],axis=1)
cat_data"""


# In[249]:


cat_data = pd.get_dummies(cat_data,columns=["Attrition"],drop_first=True)
cat_data


# ### Encode the overTime columns with the Label encoding technique

# In[250]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
cat_data['encoded_OverTime'] =labelencoder.fit_transform(cat_data.OverTime)
cat_data


# ### Perform Target encoding for JobRole 
# (Note: Consider attrition as target and use the sum of each jobrole group 1's to replace it) 

# In[251]:


out_mean=cat_data.groupby("JobRole")["Attrition_Yes"].mean()


# In[252]:


cat_data["encodedJobRol"] = cat_data["JobRole"].map(out_mean)


# ### Perform one-hot encoding for 'BussinessTravel, Department and Gender columns'

# In[258]:


cat_data.BusinessTravel.unique()
cat_data.Department.unique()
cat_data.Gender.unique()

categorical_cols = ["BusinessTravel","Department","Gender"]
def encode_one_hot(col):
    label = []
    for i in cat_data[col].unique():
        label.append = col + "_" + i
    df_encode= pd.DataFrame(encode.fit_transform(cat_data[col]).toarray(),columns=label)
    return df_encode


# In[259]:


from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder()
for item in categorical_cols:
    cat_data = pd.concat([cat_data,encode_one_hot(item)],axis=1)
cat_data


# ### Perform a Frequency encoding for Education Field

# In[ ]:


encoding = cat_data.groupby("EducationField").size()
encoding = encoding/len(cat_data)
cat_data["EducationField_freq_encoded"] = cat_data.EducationField.map(encoding)
cat_data.head()


# ### Inspect the outliers present in the data and drop the outliers

# In[ ]:




