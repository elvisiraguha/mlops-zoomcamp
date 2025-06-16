#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[3]:


import pickle
import pandas as pd


# In[5]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[6]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[7]:

from sys import argv

year = argv[1]
month = argv[2]

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet')


# In[8]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


print(y_pred.mean())


# In[10]:


year = 2023
month = 3
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[12]:


# df results = df predictions + ride id
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# In[13]:


df_result


# In[14]:


df_result.to_parquet(
    'output_file.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)

