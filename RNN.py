
# coding: utf-8

# In[3]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, CuDNNGRU
from keras import optimizers
from keras.regularizers import l1,l2


# In[4]:


import numpy as np
import pandas as pd


# In[5]:


X_train = np.zeros((30, 7500, 4))

for x in range(0,29):
    X_train[x] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values
X_train = X_train*10**6

# In[6]:





# In[7]:


X_test = np.zeros((10,7500,4))

for x in range(30,39):
    X_test[x-30] = pd.read_excel('{num}.xls'.format(num = x)).drop('Col5',axis=1).values
X_test = X_test*10**6


# In[8]:


Ydata = np.array([0.490, 0.306, 0.418, 0.504, 0.499, 0.848, 0.654, 0.473, 0.453, 0.399, 
                  0.551, 0.425, 0.588, 0.747, 0.443, 0.324, 0.571, 0.667, 0.554, 0.705,
                  0.926, 0.492, 0.715, 0.647, 0.626, 0.743, 1.110, 1.073, 0.684, 0.347,
                  0.636, 0.331, 0.574, 0.473, 0.370, 0.563, 0.845, 0.928, 0.418, 0.404]).reshape(-1,1)


# In[9]:


Y_train = Ydata[0:30]
Y_test = Ydata[30:40]


# In[12]:


model = Sequential()
model.add(CuDNNGRU(5,
    
    return_sequences=False,
    ))   


# In[16]:


model.add(Dense(1, W_regularizer=l2(0.01)))
Nadam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,schedule_decay=0.004)
model.compile(optimizer=Nadam,loss='mse')


# In[17]:


model.fit(X_train, Y_train, epochs=500, batch_size=5)


# In[ ]:


test_data = model.predict(X_test, batch_size=5)
print ("test_data = \n",test_data,'\n\n')
diff1 = abs(test_data-Y_test)
#print ("test_Deviation = \n",diff1,'\n')
diff1_sqr = diff1**2
Test_RMSE =( diff1_sqr.sum()/10 )**0.5
print("Test_RMSE = \n",Test_RMSE)


# In[ ]:


train_data = model.predict(X_train, batch_size=5)
#print ("train_data = \n",train_data,'\n\n')
diff2 = abs(train_data-Y_train)
diff2_sqr = diff2**2
Train_RMSE =( diff2_sqr.sum()/30 )**0.5
Train_RMSE_test = (abs(train_data-Y_train)**2).sum()/30**0.5
print("Train_RMSE = ",Train_RMSE)
print("Train_RMSE_test = ", Train_RMSE_test)



