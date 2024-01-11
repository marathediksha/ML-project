#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('coin_Litecoin.csv')
df


# In[3]:


df.shape


# In[4]:


df.head(10) 


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[10]:


df['Date'] = pd.to_datetime(df['Date'])

# Filter data for the last five years
five_years_data = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(years=5))]

# Create a line plot for 'Close' prices
plt.figure(figsize=(12, 6))
sns.lineplot(x=five_years_data['Date'], y=five_years_data['Close'], color='royalblue')
plt.title('Litecoin Prices (Last 5 Years)', size=14, fontweight='bold')
plt.xlabel("Date", size=14)
plt.ylabel("Close Price (USD)", size=14)
plt.xticks(size=10)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()


# In[11]:


df.drop(columns=['SNo','Name','Symbol','Date','High','Low','Open','Volume','Marketcap'], inplace=True) #When inplace = True , the data is modified in place, which means it will return nothing and the dataframe is now updated.
#When inplace = False , which is the default, then the operation is performed and it returns a copy of the object.


# In[12]:


df.head(10)


# In[13]:


df.boxplot()


# In[14]:


prediction_days = 25  # A variable for predicting 'n' days out into the future. 


# In[15]:


# Now we will make a new column which will be shifted 'n' units up. It'll contain the 'close' price but the only difference will be, that this column will be shifted 25 rows up.
df['Prediction']=df[['Close']].shift(-prediction_days)


# In[16]:


df.head(10)


# In[17]:


df.tail(26)


# In[28]:


# Create the independent dataset

# Convert the data to a numpy array and drop the prediction column.(Because it contains the target variables)

X=np.array(df.drop(['Prediction'],1))

# We now have to remove the last 'n' rows where 'n' is the prediction_days (25 in our case) 

X=X[:len(df)-prediction_days]

print(X)


# In[29]:


# Create the dependent dataset. 

# Convert the dataframe to a numpy array.

y=np.array(df['Prediction'])

# Get all the values except the last 'n' rows

y=y[:-prediction_days]
print(y)


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1)


# In[31]:


# Now we need to create a new variable "prediction_days_array" and have to make it equal to the last 25 rows from the original dataset (only the 'close' column)

prediction_days_array=np.array(df.drop(['Prediction'],1))[-prediction_days:]
print(prediction_days_array)


# # Linear Regression

# In[32]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[33]:


lr.coef_


# In[34]:


lr.intercept_


# In[35]:


lr.fit(x_train,y_train)


# In[36]:


Y_pred=lr.predict(x_test)


# In[37]:


Y_pred


# In[38]:


#Seeing Model Coeffiecient and Intercept


# In[39]:


lr.coef_


# In[40]:


lr.intercept_


# In[41]:


from sklearn import metrics


# In[42]:


rmse=metrics.r2_score(y_test,Y_pred)
print(rmse)


# In[43]:


mse=metrics.mean_squared_error(y_test,Y_pred)
np.sqrt(mse)


# In[44]:


import matplotlib.pyplot as plt

# Plot the actual Litecoin prices
plt.figure(figsize=(12, 6))
plt.scatter(df.index[-len(y_test):], y_test, color='blue', label='Actual Prices')

# Plot the linear regression predictions
plt.plot(df.index[-len(y_test):], Y_pred, color='red', linewidth=2, label='Linear Regression Prediction')

plt.title('Litecoin Price Prediction using Linear Regression')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# # Gradient Booster

# In[45]:


from sklearn.ensemble  import GradientBoostingRegressor

gb = GradientBoostingRegressor()

gb.fit(x_train,y_train)

pred = gb.predict(x_test)

gb.score(x_test,y_test)


# # Let's Predict the values of litecoin !!!

# In[46]:


gb_prediction= gb.predict(x_test)

print(gb_prediction)

print()
print()

# Print the actual value as well

print(y_test)
gb_prediction= gb.predict(x_test)

print(gb_prediction)


# In[47]:


# Print the model predictions for 'n=25' 

gb_prediction_n= gb.predict(prediction_days_array)
print(gb_prediction_n)

# Print the actual prize of litecoin for last 25 days. 

print(df.tail(prediction_days))


# In[48]:


df['Actual cost of Litecoin of last 25 days'] = pd.DataFrame(prediction_days_array)
df['Predicted Cost of Litecoin of last 25 days'] = pd.DataFrame(gb_prediction_n)
A=df
A.drop(columns=['Close','Prediction'], inplace= True)


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
sns.lineplot(x=df.index, y=df['Actual cost of Litecoin of last 25 days'], label="Actual cost", color='royalblue')
sns.lineplot(x=df.index, y=df['Predicted Cost of Litecoin of last 25 days'], label="Predicted cost", color='tomato')
plt.title('Actual vs. Predicted Litecoin Prices', size=14, fontweight='bold')
plt.xlabel("Days", size=14)
plt.ylabel("Cost (USD)", size=14)
plt.xticks(size=10)
plt.legend()
plt.grid(True)
plt.show()


# # SVR (support vector regression)

# In[50]:


from sklearn.svm import SVR

# Now create and train the Support vector machine (Which is using Regression) using radial basis function.

svr_rbf= SVR(kernel='rbf',C=1e3, gamma=0.00001)


# In[51]:


svr_rbf.fit(x_train, y_train)


# In[52]:


# Let's test our model

svr_score = svr_rbf.score(x_test, y_test)


# In[53]:


print("svr_score: ", svr_score )


# # Implementing Ridege and Lasso

# In[54]:


from sklearn.linear_model import Ridge,Lasso


# In[55]:


rd=Ridge()
rd.fit(x_train,y_train)
rd.score(x_test,y_test)


# In[56]:


ls=Lasso()
ls.fit(x_train,y_train)
ls.score(x_test,y_test)


# In[57]:


rd2=Ridge(alpha=2)
rd2.fit(x_train,y_train)
rd2.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




