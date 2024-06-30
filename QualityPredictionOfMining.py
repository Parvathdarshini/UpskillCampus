#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pylab import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



# In[2]:


df = pd.read_csv(r"C:\Users\darshini\OneDrive\Desktop\xyz\miningprocess_flotation_plant_dataset.csv",decimal=',',parse_dates=["date"],infer_datetime_format=True).drop_duplicates()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.head(10)


# In[6]:


df['date'] = pd.to_datetime(df['date'], format= "%d-%m-%Y %H:%M:%S")


# In[7]:


df.info()


# In[8]:


plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), vmin=-1.0, vmax=1.0)
plt.show()


# In[9]:


fig, axs =plt.subplots(4,2,figsize=(8, 8))
fig.suptitle('Possible features correlation to label')


axs[0,0].scatter(df['%IronConcentrate'],df['%SilicaConcentrate'])
axs[0,0].set_xlabel(['%IronConcentrate'])
axs[0,0].set_ylabel(['%SilicaConcentrate'])

axs[0,1].scatter(df['AminaFlow'],df['%SilicaConcentrate'])
axs[0,1].set_xlabel(['AminaFlow'])
axs[0,1].set_ylabel(['%SilicaConcentrate'])

axs[1,0].scatter(df['OrePulpFlow'],df['%SilicaConcentrate'])
axs[1,0].set_xlabel(['OrePulpFlow'])
axs[1,0].set_ylabel(['%SilicaConcentrate'])

axs[1,1].scatter(df['OrePulpDensity'],df['%SilicaConcentrate'])
axs[1,1].set_xlabel(['OrePulpDensity'])
axs[1,1].set_ylabel(['%SilicaConcentrate'])

axs[2,0].scatter(df['FlotationColumn01AirFlow'],df['%SilicaConcentrate'])
axs[2,0].set_xlabel(['FlotationColumn01AirFlow'])
axs[2,0].set_ylabel(['%SilicaConcentrate'])

axs[2,1].scatter(df[ 'FlotationColumn04Level'],df['%SilicaConcentrate'])
axs[2,1].set_xlabel([ 'FlotationColumn04Level'])
axs[2,1].set_ylabel(['%SilicaConcentrate'])


axs[3,0].scatter(df['% Silica Feed'],df['%SilicaConcentrate'])
axs[3,0].set_xlabel(['% Silica Feed'])
axs[3,0].set_ylabel(['%SilicaConcentrate'])

axs[3,1].scatter(df['%SilicaConcentrate'],df['% Silica Feed'])
axs[3,1].set_xlabel(['%SilicaConcentrate'])
axs[3,1].set_ylabel(['% Silica Feed'])


plt.tight_layout()


# In[10]:


df.columns
droplist=['% Iron Feed','StarchFlow',\
          'OrePulppH','FlotationColumn02AirFlow',\
          'FlotationColumn03AirFlow', 'FlotationColumn04AirFlow',\
          'FlotationColumn05AirFlow', 'FlotationColumn06AirFlow',\
          'FlotationColumn07AirFlow', 'FlotationColumn01Level',
          'FlotationColumn02Level', 'FlotationColumn03Level', 'FlotationColumn05Level',
          'FlotationColumn06Level', 'FlotationColumn07Level',]

df=df.drop(droplist,axis=1)
df.shape


# In[11]:


df['%IronConcentrate_power2']=df['%IronConcentrate']**2

df


# In[12]:


df.groupby(['%SilicaConcentrate']).mean()


# In[13]:


df.groupby(['%IronConcentrate']).mean()


# In[14]:


df=df.drop(['date'],axis=1)

y = df['%SilicaConcentrate']
X = df.drop(['%SilicaConcentrate'], axis=1)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_i_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_i_scaled,
                                                    y,
                                                    test_size=0.3,
                                                   random_state=30)


# In[16]:


from sklearn.linear_model import LinearRegression


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_linreg = lin_reg.predict(X_test)


MSE = mean_squared_error(y_test, y_pred_linreg)
print('Our Linear Regression mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_linreg)
print('Our Linear Regression mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_linreg) 
print('Our Linear Regression R2 score is: ', R2)
RMSE =  np.sqrt(mean_squared_error(y_test, y_pred_linreg))
print('Our Linear Regreesion Root Mean Squared Error is:',RMSE)


# In[17]:


import xgboost as xgb
xgb = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


MSE = mean_squared_error(y_test, y_pred_xgb)
print('Our XGBoost mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_xgb)
print('Our XGBoost mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_xgb) 
print('Our XGBoost R2 score is: ', R2)
RMSE =  np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print('Our XGBoost Root Mean Squared Error is:', RMSE)


# In[18]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
y_pred_lasso = lasso_regressor.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_lasso)
print('Our Lasso Regression mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_lasso)
print('Our Lasso Regression mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_lasso) 
print('Our Lasso Regression R2 score is: ', R2)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print('Our Lasso Regression Root Mean Squared Error is:',RMSE)


# In[32]:


from sklearn.linear_model import Ridge


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
y_pred_ridge = ridge_regressor.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_ridge)
print('Our Ridge Regression mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_ridge)
print('Our Ridge Regression mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_ridge) 
print('Our Ridge Regression R2 score is: ', R2)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print('Our Ridge Regression Root Mean Squared Error is:', RMSE)


# In[20]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)


MSE = mean_squared_error(y_test, y_pred_dtr)
print('Our Decision Tree mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_dtr)
print('Our Decision Tree mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_dtr) 
print('Our Decision Tree R2 score is: ', R2)
RMSE =  np.sqrt(mean_squared_error(y_test, y_pred_dtr))
print('Our Decision Tree Root Mean Squared Error is:', RMSE)


# In[21]:


from sklearn.model_selection import RandomizedSearchCV

params={
 "learning_rate"    : [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 0.9, 1.0 ]
}


# In[22]:


random_search= RandomizedSearchCV(estimator=xgb,
                                param_distributions=params,
                                cv=5, n_iter=50,
                                scoring = 'r2',n_jobs = 4,
                                verbose = 1, 
                                return_train_score = True,
                                random_state=42)


# In[23]:


random_search.fit(X_train, y_train)


# In[24]:


random_search.best_estimator_


# In[27]:


import xgboost as xgb
xgb = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1.0, gamma=0.2, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.25, max_delta_step=0, max_depth=10,
             min_child_weight=7, monotone_constraints=None,
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:linear', random_state=42, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
xgb.fit(X_train, y_train)
y_pred_xgb_tunning = xgb.predict(X_test)


# In[28]:


MSE = mean_squared_error(y_test, y_pred_xgb_tunning)
print('Our XGBoost after tunning mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_xgb_tunning)
print('Our XGBoost after tunning mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_xgb_tunning) 
print('Our XGBoost after tunning R2 score is: ', R2)
print('Our XGBoost after tunning Root Mean Squared Error is:', np.sqrt(mean_squared_error(y_test, y_pred_xgb_tunning)))


# In[29]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb_tunning})
result.head(20)


# In[31]:


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.set(title="XG Boost Tunning", xlabel="Actual", ylabel="Predicted")
ax.scatter(y_test, y_pred_xgb_tunning)
ax.plot([0,max(y_test)], [0,max(y_pred_xgb_tunning)], color='r')
fig.show()


# In[ ]:




