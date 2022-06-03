#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/Users/czkaiweb/Research/ErdosBootCamp/May2022/vanGogh-and-Other-Artist')
from genericVoter import *
from preprocessing.ImageTranform import *
from torchsummary import summary

import os
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


# In[2]:


myVoter = genericVoter()

listTransformer = []
listModel = []
listWeight = []
# For vgg16
myTransform = ImageTransformer((224,224))
myTransform.initTransform()
transformer = myTransform.getTransformer()
listTransformer.append(transformer)

model_ft = models.vgg16()
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, 6)
model_ft = model_ft.to(myVoter.device)
listModel.append(model_ft)

listWeight.append('/Users/czkaiweb/Research/ErdosBootCamp/May2022/vanGogh-and-Other-Artist/model/postTrain/VGG_postTrain_weights_May25.pth')

# For GoogleNet
myTransform = ImageTransformer((224,224))
myTransform.initTransform()
transformer = myTransform.getTransformer()
listTransformer.append(transformer)

model_ft = models.googlenet()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)
model_ft = model_ft.to(myVoter.device)
listModel.append(model_ft)

listWeight.append('/Users/czkaiweb/Research/ErdosBootCamp/May2022/vanGogh-and-Other-Artist/model/postTrain/googleNet_weights.pth')


myVoter.setBagging(listModel,listTransformer,listWeight)


# In[3]:


myTransform = ImageTransformer((224,224))
myTransform.initTransform()
transformer = myTransform.getTransformer()
myVoter.setTransformer(transformer)

# Set up the meta data and path to image dataset
myVoter.setDataset("../data/meta.csv",path = "../data/imgs")

# Split the data by portion, fraction indicate the percentage of data used in the whole dataset. 
# Default: val_size = 0.2, test_size = 0.1 
myVoter.splitData(val_size=0.1,test_size = 0.7,fraction = 0.1)

# Will automatically get the statistic for training set, update the mean/std used for normalization. 
# loadData and checkDataset
myVoter.loadData()
myVoter.checkDataset()
myVoter.getStat()


# In[4]:


myVoter.prepareInputForVoter()


# In[5]:


import xgboost as xgb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

voters = {
#    "xgboost"  : xgb.XGBClassifier(objective="multi:softprob", random_state=42),
    "svm_rbf"  : svm.SVC(kernel='rbf', gamma=0.5, C=0.1),
    "svm_poly" : svm.SVC(kernel='poly', degree=3, C=1),
}

myVoter.setVoterClassifier(voters)
myVoter.fitVoter()
myVoter.evaluateVoter()

