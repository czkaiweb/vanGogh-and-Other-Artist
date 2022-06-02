# StarryNight: Predicting Arstists of Different Paitings
Final project for the Erdos Institute 2022 boot camp. 

Team members: Xiaozhou Feng, Xiaoyu Liu, Estefany Nunez, Bryan Reynolds, Kai Wei
# Project Descriptionton
When we see a paiting, we may not be able to tell its artist immediately. If we were in an art exhibition, then the labels can tell us it. But it may be not very convenient to go to exhibitions frequently to find the artist of a painting which we encounter in life. It will be nice if we can have an app to do this quickly. The neural network provides us an opportunity to fix this problem. Basically, a classifier is consisted of multiple layers of nodes with parameters (weights) trained by an existing dataset. When the input (a paiting in our problem) is determined, it automatically generates an output as the prediction of artist of the painting. Here we have a dataset of paiting of six artists: Monet, Van Gogh, da Vinci, Rembrandt, Picasso and Dali and use this dataset to train several different neural network models. Our testing resutls show that our models can give very good predictions of artist of paintings. A website is established to deploy our trained model. Also, a style transfer is added based on the trained model [StyleTransfer](https://huggingface.co/spaces/breynolds1247/StarryNight_StyleTransfer).
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
# General Usage

This repository can easily used. We provide a colab template which people can use to train your dataset and choose your favoriate model. 


## Download data
To download the Von Gogh dataset used in this project
```
!kaggle datasets download -d ipythonx/van-gogh-paintings
```
To download the Monet dataset
```
!kaggle datasets download -d srrrrr/monet2photo
```
To download others paitings (da Vinci, Rembrandt, Picasso and Dali)
```
!kaggle datasets download -d czkaiweb/subwikiarts
```

## Train your model
The neural network is initialized by (Here we use the class genericCNN provided in genericCNN.py)
```
myObj = genericCNN()
```
To set up the data transformer, try
```
myTransform = ImageTransformer((224,224))
myTransform.initTransform()
transformer = myTransform.getTransformer()

myObj.setTransformer(transformer)
```
Data setup (both meta data and the path)
```
myObj.setDataset("meta.csv",path = "imgs")
```

To choose the sizes of your training, validation and test sets (Here we use valitation set: 10%, test set: 70%)
```
myObj.splitData(val_size=0.1,test_size = 0.7,fraction = 1)
```
Data loading
```
myObj.loadData(reUseTrain=3)
```

The next step is to choose the model which you want to use. For example, if we use the resnet34 model
```
model_ft = models.resnet34(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)
model_ft = model_ft.to(myObj.device)

myObj.setModel(model = model_ft,modeltag="resnet34mod")
```
Then we can just train our network with your choise of the criterion, optimizer and schdeduler
```
myObj.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs=21)
```
## Results
The confusion matrix can be obtained by
```
myObj.evaluate()
```
For resnet34, we get
