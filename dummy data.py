#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import random

# Generate dummy data
data = {
    'PatientID': [f"P{1000+i}" for i in range(200)],
    'Gender': random.choices(['Male', 'Female'], k=200),
    'Age': [random.randint(18, 75) for _ in range(200)],
    'Region': random.choices(['North', 'South', 'East', 'West'], k=200),
    'Diagnosis': random.choices(['Yes', 'No'], k=200)
}

df = pd.DataFrame(data)
df.to_csv("dummy_patient_data.csv", index=False)
print(df)


# In[ ]:




