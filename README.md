# StarryNight: Predicting Arstists of Different Paitings
Final project for the Erdos Institute 2022 boot camp. 

Team members: 
- Xiaozhou Feng
- Xiaoyu Liu
- Estefany Nunez
- Bryan Reynolds
- Kai Wei
# Project Descriptionton
When we see a paiting, we may not be able to tell its artist immediately. If we were in an art exhibition, then the labels can tell us it, but it may be not very convenient to go to exhibitions frequently to find the artist of a painting which we encounter in life. It would be nice if we can have an app to do this quickly. Convolutional neural networks (CNNs) provide us with an opportunity to fix this problem. Basically, a classifier using a pre-trained convolutional neural network consists of multiple layers of nodes with parameters (weights) trained by an existing dataset. When the input (a paiting in our problem) is provided, it automatically generates a prediction of the artist of the painting as an output. Here we use a dataset of paintings of six artists: Monet, van Gogh, da Vinci, Rembrandt, Picasso, and Dali and use this dataset to train several different neural network models. Our test resutls show that our models can provide very accurate predictions of artist of paintings. A website is established to deploy our classifier [ArtitstClassifier](https://huggingface.co/spaces/czkaiweb/StarryNight). Also, a style transfer model is added based on a CNN model trained on famous paintings from each artist [StyleTransfer](https://huggingface.co/spaces/breynolds1247/StarryNight_StyleTransfer).
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

This repository can be easily used. We provide a colab template which people can use to train your dataset on Google Colab's free GPUs and choose your favoriate model. 


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
To set up a very basic data transformer, try
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

To choose the sizes of your training, validation and test sets, as well as the fraction of data to run over (Here we use valitation set: 10%, test set: 70%, and the full dataset)
```
myObj.splitData(val_size=0.1,test_size = 0.7,fraction = 1)
```
Data loading (the reUseTrain parameter is used for data augmentation, the default value is 1 if you only want to use each training image once)
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
Then we can just train our network with your choice of the criterion, optimizer and scheduler
```
myObj.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs=21)
```
## Test the Trained Model
The confusion matrix can be obtained by
```
myObj.evaluate()
```
For resnet34, we get

![confusionmatrix](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/evaluation_resnet34.png)
It can be seen that for some artists the results are good, but for others the results are less accurate. You can try different models to find the best prediction.

We can also track the history of accuracy and loss function after each epoch with
```
myObj.drawHistory()
```
![accuracy_history](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/accuracy_history_resnet34.png)

![loss_history](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/loss_history_resnet34.png)

## Save the Trained Weights

To save and download your trained weights, 
```
torch.save(myObj.Model.state_dict(), 'model_weights.pth')

from google.colab import files
files.download("model_weights.pth")
```

## Voter

After training several different models, we can combine these results into a voter to get a more accurate prediction. The comparison between the base models (0: VGG, 1: EfficientNet, 2: MobileNet, 3:ResNet34, 4: ConvNext), the voter (like random forests) and hard voting is shown as following
![accuracy_scores](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/accuracy_scores.png)

It can be seen that the prediction is slightly improved and the hard voting gives the best result.

The confusion matrix of the voter

![voter_confusionmatrix](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/voter_confusionmatrix.png)

The confusion matrix of hard voting

![hardvoting_confusionmatrix](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/hardvoting_confusionmatrix.png)
# Applications of Our Training Results

We build an artist classifier model based on our trained model and a style transfer model based on CNN transfer learning techniques. 
## Artist Classifier
For the artist classifier, people can use the link https://huggingface.co/spaces/czkaiweb/StarryNight to check the artist of a painting. 

For example
![artist_check](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/artist_check.png)

A paiting of Monet is used to test the classifier and it gives a great result!

## Style Transfer

Another interesting application of CNN models is style transfer, which makes your picture have a similar style as the paintings of the famous artists. The style transfer application is available at this link https://huggingface.co/spaces/breynolds1247/StarryNight_StyleTransfer.

Here we use a picture of the CMS detector. By using the style transfer, now it has Picasso's painting style!

![style_transfer](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/style_transfer.png)

If we then upload this figure to our classifier, and it is recognized as Picasso's painting!

![style_transfer_check](https://github.com/czkaiweb/vanGogh-and-Other-Artist/blob/main/style_transfer_check.png)
