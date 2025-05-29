#!/usr/bin/env python
# coding: utf-8

# # Homework

# In[2]:


import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression


# In[13]:


import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("duration-prediction")


# In[4]:


def read_dataframe(year, month):
    link = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(link)

    print(len(df), 'initial rows')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


# ### Downloading the data

# In[5]:


year = 2023
month = 3
df = read_dataframe(year, month)
print(len(df), 'rows after filtering')


# ### One-hot encoding

# In[6]:


categorical = ['PULocationID', 'DOLocationID']

df_dict = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(df_dict)


# ### Training a model

# In[21]:


with mlflow.start_run() as run:
    y_train = df['duration'].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)

    with open("models/duration_model.bin", "wb") as f_out:
        pickle.dump(lr, f_out)

    mlflow.log_artifact("models/duration_model.bin", artifact_path="models")


# In[9]:


lr.intercept_


run.info.run_id



