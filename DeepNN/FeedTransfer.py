
# coding: utf-8

# In[31]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

import h5py as h5py


# In[4]:


import matplotlib.pyplot as plt


# ### Import Model 

# In[23]:


model = InceptionV3(weights='imagenet')


# In[29]:


model.summary() #Prints the architecture of the model


# ### Import Image and Pre-processing

# In[72]:


img_path = '../PICS/pics_erepS/DD3.jpg'
img = image.load_img(img_path, target_size=(250, 400))
img.show()
x = image.img_to_array(img)

plt.imshow(x)
plt.show()
#print(x)

y=x/255
plt.imshow(y)
plt.show()

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print("Shape of Image :")
print(x.shape)
plt.imshow(x[0,:,:,:])
plt.show()


# In[61]:


preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=5)[0])


# In[15]:


preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

