#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[2]:


#reading the dataset

data = pd.read_csv("Instagram data.csv", encoding = 'latin1')
data.head()


# In[3]:


#checking the null values in dataset

data.isnull().sum()


# In[4]:


#defining the dataset's data type

data.info()


# In[3]:


#checking the correlation between the datas

data.select_dtypes('number').corr()


# In[4]:


# Plotting the Correlation using the Heatmap

plt.figure(figsize = (20, 5))
px.imshow(data.select_dtypes('number').corr(), text_auto = True, aspect = 'auto')


# In[5]:


# Plotting Reach & Avg. Reach from the Different Sources

x = data[['From Home', 'From Hashtags', 'From Explore', 'From Other']]
px.bar(x, title = 'Plotting the Reach from Different Sources').show()
px.bar(x.mean(), color=x.columns, title = 'Plotting the Avg. Reach from Different Sources')


# In[10]:


#hist plot for impressions from home

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'])
plt.show()


# In[17]:


#hist plot for impressions from Hashtags

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'])
plt.show()


# In[12]:


#hist plot for impressions from explore

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(data['From Explore'])
plt.show()


# In[21]:


plt.figure(figsize=(12, 6))
sns.lineplot(x=data.index, y=data['Impressions'])
plt.xlabel('Post Index')
plt.ylabel('Reach (Impressions)')
plt.title('Trend of Reach over Time')
plt.show()


# In[6]:


#pie chart for impressions

home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()


# In[12]:


text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[13]:


text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[24]:


# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Create a DataFrame
df = pd.DataFrame(data)

# Set the Timestamp column as the index
df.set_index('Timestamp', inplace=True)

# Plotting the reach over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=df.index, y='Reach', marker='o', color='b')
plt.title('Post Reach Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Reach')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[7]:


Total_profile_visits = data['Profile Visits'].sum()
Total_likes = data['Likes'].sum()
Total_follows = data['Follows'].sum()

#print the total counts
print(f'Total Profile Visits: {Total_profile_visits}')
print(f'Total Likes: {Total_likes}')
print(f'Total Follows: {Total_follows}')

#plot bar
plt.figure(figsize=(8,6))
plt.bar(['Profile Visits','Likes','Follows'],[Total_profile_visits,Total_likes,Total_follows])
plt.title('User Behaviour Analysis')
plt.ylabel('count')
plt.show()


# In[7]:


#scatter plot for Relationship Between Likes and Impressions

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


# In[8]:


#scatter plot for Relationship Between comments and Impressions

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Impressions")
figure.show()


# In[9]:


#scatter plot for Relationship Between Shares and Impressions

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Impressions")
figure.show()


# In[10]:


#Scatter plot for relationship between Saves and Impression

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()


# In[8]:


Total_impressions = data['Impressions'].sum()
Total_saves = data['Saves'].sum()
Total_comments = data['Comments'].sum()
Total_shares = data['Shares'].sum()
Total_likes = data['Likes'].sum()

Engagement_Rate = (Total_likes + Total_comments + Total_shares)/Total_impressions
print("Engagement Rate is calculated as: ", Engagement_Rate)


# In[20]:


print(f"Total Impressions: {Total_impressions}")
print(f"Total Saves: {Total_saves}")
print(f"Total Comments: {Total_comments}")
print(f"Total Shares: {Total_shares}")
print(f"Total Likes: {Total_likes}")
print(f"Engagement Rate: {Engagement_Rate: .2%}")


# In[18]:


#finding the Conversion Rate

conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# In[141]:


data.plot(x='Profile Visits', y='Follows', style='o')
plt.title('Relationship')
plt.xlabel('Profile Visits')
plt.ylabel('Follows')
plt.show()


# In[142]:


x = data.iloc[:, :11].values
y = data.iloc[:, 1].values


# In[143]:


print (x)


# In[24]:


# Model Building for RandomForestRegressor


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

regressor = RandomForestRegressor()


# In[25]:


# Assuming you have a DataFrame named 'df' with columns: 'likes', 'comments', 'followers', 'reach'
# Load your data into 'df'

# Create a pair plot
sns.pairplot(data[['Follows', 'Profile Visits']])
plt.show()

# Split the data into features (X) and target variable (y)
X = data[['Profile Visits']]
y = data['Follows']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = model.predict(xtest)


# In[145]:


# Splitting the Dataset

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 123)
print('x train Shape:', xtrain.shape)
print('y train Shape:', ytrain.shape)
print('x test Shape:', xtest.shape)
print('y test Shape:', ytest.shape)


# In[146]:


# Fiting the Model into the RandomForestRegressor

regressor.fit(xtrain, ytrain)


# In[147]:


# Making the Prediction

prediction = regressor.predict(xtest)
prediction.shape


# In[150]:


#model prediction

model = RandomForestRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[149]:


# Instantiate the model
model = RandomForestRegressor(random_state=123)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the testing set
ypred = model.predict(xtest)

# Evaluate the model
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred)

# Calculate reference points
perfect_model_mse = mean_squared_error(ytest, ytest * np.ones_like(ytest))  # MSE for a perfect model (predictions equal to true values)
null_model_mse = mean_squared_error(ytest, np.full_like(ytest, np.mean(ytest)))  # MSE for a null model (predictions equal to mean)

# Scale MSE and RMSE
scaled_mse = mse / (perfect_model_mse - null_model_mse)
scaled_rmse = rmse / np.sqrt(null_model_mse)

# Print the scaled metrics
print(f'Scaled Mean Squared Error (MSE): {scaled_mse}')
print(f'Scaled Root Mean Squared Error (RMSE): {scaled_rmse}')
print(f'R-squared (R²): {r2}')


# In[194]:


df = pd.DataFrame({'Actual':ytest, 'Predicted': ypred})
df


# In[196]:


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Assuming you have your data loaded as xtrain, ytrain, xtest, ytest
# Also, assuming you have already trained the Random Forest Regressor model and made predictions

# Instantiate the model
model = RandomForestRegressor(n_estimators=100, random_state=123)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the testing set
ypred = model.predict(xtest)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))

plt.scatter(ytest, ypred, color='red', edgecolors=(0, 0, 0))
plt.title('Actual vs. Predicted Values for Random Forest Regressor')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[188]:


data.plot(x='Profile Visits', y='Follows', style='o')
plt.title('Relationship')
plt.xlabel('Profile Visits')
plt.ylabel('Follows')
plt.show()


# In[189]:


x = data.iloc[:, :11].values
y = data.iloc[:, 1].values


# In[190]:


print (x)


# In[191]:


# model building for PassiveAggressiveRegressor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, r2_score

regressor = PassiveAggressiveRegressor


# In[28]:



from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
# Create a pair plot
sns.pairplot(data[['Follows', 'Profile Visits']])
plt.show()

# Split the data into features (X) and target variable (y)
X = data[['Follows', 'Profile Visits']]
y = data['Follows']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Passive Aggressive Regressor model
model = PassiveAggressiveRegressor(C=1.0, random_state=42)
model.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = model.predict(xtest)


# In[192]:


# Splitting the Dataset

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 123)
print('x train Shape:', xtrain.shape)
print('y train Shape:', ytrain.shape)
print('x test Shape:', xtest.shape)
print('y test Shape:', ytest.shape)


# In[193]:


#model prediction

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[185]:


# Instantiate the model
model = PassiveAggressiveRegressor(random_state=123)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the testing set
ypred = model.predict(xtest)

# Evaluate the model
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred)

# Calculate reference points
perfect_model_mse = mean_squared_error(ytest, ytest * np.ones_like(ytest))  # MSE for a perfect model (predictions equal to true values)
null_model_mse = mean_squared_error(ytest, np.full_like(ytest, np.mean(ytest)))  # MSE for a null model (predictions equal to mean)

# Scale MSE and RMSE
scaled_mse = mse / (perfect_model_mse - null_model_mse)
scaled_rmse = rmse / np.sqrt(null_model_mse)

# Print the scaled metrics
print(f'Scaled Mean Squared Error (MSE): {scaled_mse}')
print(f'Scaled Root Mean Squared Error (RMSE): {scaled_rmse}')
print(f'R-squared (R²): {r2}')


# In[186]:


#actual & predicted
df = pd.DataFrame({'Actual':ytest, 'Predicted': ypred})
df


# In[170]:


data.plot(x='Profile Visits', y='Follows', style='o')
plt.title('Relationship')
plt.xlabel('Profile Visits')
plt.ylabel('Follows')
plt.show()


# In[171]:


x = data.iloc[:, :11].values
y = data.iloc[:, 1].values


# In[172]:


print (x)


# In[173]:


# model building for GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Create a pair plot
sns.pairplot(data[['Follows', 'Profile Visits']])
plt.show()

# Split the data into features (X) and target variable (y)
X = data[['Follows', 'Profile Visits']]
y = data['Follows']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = model.predict(xtest)


# In[174]:


# Splitting the Dataset

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 123)
print('x train Shape:', xtrain.shape)
print('y train Shape:', ytrain.shape)
print('x test Shape:', xtest.shape)
print('y test Shape:', ytest.shape)


# In[175]:


# Instantiate the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=123)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the testing set
ypred = model.predict(xtest)

# Evaluate the model
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred)

# Calculate reference points
perfect_model_mse = mean_squared_error(ytest, ytest * np.ones_like(ytest))  # MSE for a perfect model (predictions equal to true values)
null_model_mse = mean_squared_error(ytest, np.full_like(ytest, np.mean(ytest)))  # MSE for a null model (predictions equal to mean)

# Scale MSE and RMSE
scaled_mse = mse / (perfect_model_mse - null_model_mse)
scaled_rmse = rmse / np.sqrt(null_model_mse)

# Print the scaled metrics
print(f'Scaled Mean Squared Error (MSE): {scaled_mse}')
print(f'Scaled Root Mean Squared Error (RMSE): {scaled_rmse}')
print(f'R-squared (R²): {r2}')


# In[176]:


#model prediction

model = GradientBoostingRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[177]:


#actual & predicted
df = pd.DataFrame({'Actual':ytest, 'Predicted': ypred})
df


# In[178]:


import matplotlib.pyplot as plt

# Assuming you have your data loaded as xtrain, ytrain, xtest, ytest
# Also, assuming you have already trained the Gradient Boosting Regressor model and made predictions

# Instantiate the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=123)

# Train the model on the training data
model.fit(xtrain, ytrain)

# Make predictions on the testing set
ypred = model.predict(xtest)

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))

plt.scatter(ytest, ypred, color='blue', edgecolors=(0, 0, 0))
plt.title('Actual vs. Predicted Values for Gradient Boosting Regressor')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

