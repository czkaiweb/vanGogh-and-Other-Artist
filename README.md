# StarryNight: Predicting Arstists of Different Paitings
Final project for the Erdos Institute 2022 boot camp. 

Team members: Xiaozhou Feng, Xiaoyu Liu, Estefany Nunez, Bryan Reynolds, Kai Wei
# Project Descriptionton
When we see a paiting, we may not be able to tell its artist immediately. If we were in an art exhibition, then the labels can tell us it. But it may be not very convenient to go to exhibitions frequently to find the artist of a painting which we encounter in life. It will be nice if we can have an app to do this quickly. The neural network provides us an opportunity to fix this problem. Basically, a classifier is consisted of multiple layers of nodes with parameters (weights) trained by an existing dataset. When the input (a paiting in our problem) is determined, it automatically generates an output as the prediction of artist of the painting. Here we have a dataset of paiting of six artists: Monet, Van Gogh, da Vinci, Rembrandt, Picasso and Dali and use this dataset to train several different neural network models. Our testing resutls show that our models can give very good predictions of artist of paintings. A website is established to deploy our trained model. Also, a style transfer is added based on the trained model.
# Dependencies
Conda environment:
```
conda create -n ErdosMay22 python=3.7

conda activate ErdosMay22

pip3 install notebook

pip3 install torch torchvision

pip3 install scikit-learn

pip3 install --upgrade tensorflow
```
For progress bar display
```
pip3 install tqdm
```
For URL download
```
pip3 install wget
```
Kaggle API for dataset download:
```
pip3 install --user kaggle
```
Create Kaggle API token following:  https://www.kaggle.com/docs/api
# 
