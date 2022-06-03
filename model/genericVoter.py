from preprocessing.ImageTranform import *
from preprocessing.CustomizedDataset import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import torch
import time
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import seaborn as sns

import xgboost as xgb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class genericVoter():
    def __init__(self):
        self.metaData = None
        self.metaDF = None
        self.trainDF = None
        self.valDF = None
        self.datasetChecked = False
        self.artistMap = {}

        self.UseNormalized = True
        self.StatGot = False
        self.trainTransform = None
        self.valTransform = None
        self.trainMean = None
        self.trainStd = None

        self.NetWork = None
        self.Model = None
        self.ModelTag = None

        self.Dataset = None
        self.valSize = 0.2
        self.testSize = 0.1
        self.reUseTrain = 1
        self.trainIndices = []
        self.trainDataset = None
        self.valDataset = None
        self.testDataset = None
        self.trainDataLoader = None
        self.valDataLoader = None
        self.testDataLoader = None
        self.datasetSize = {}

        self.trainLoss = []
        self.trainAccu = []
        self.valLoss = []
        self.valAccu = []
        self.skipNorm = False
        self.votingResults = []
        self.votingLabels = []

        self.batch_size = 5
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter()



        # For baggings:
        self.Voter = None
        self.baggingModels = []
        self.baggingTransformer = []
        self.baggingWeights = []
        self.baggingtestDatasetLoader = []
        self.baggingValDatasetLoader = []

        self.baseModelpreds = []
        self.labels = []
        self.voter_input = []
        self.voter_label = []
        self.classifierMap = {}
        self.bestVoter = None
        self.leastCrossEntropy = None
        self.highaccuracy = None
        self.votingInputs = []
    
    def setDataset(self, metadata, path="../data/imgs"):
        self.datasetChecked = False
        self.metaData = metadata
        if ".csv" in metadata:
            try:
                self.metaDF = pd.read_csv(metadata)
                self.metaDF = self.metaDF
            except Exception as err:
                print("Failed to read .csv file: {}".format(err))
        self.dataPath = path

        uniqueArtist = self.metaDF["Artist"].unique().tolist()
        artistCode = [i for i in range(len(uniqueArtist))]
        for i in artistCode:
            self.artistMap[i] = uniqueArtist[i]

        self.metaDF["Artist"].replace(uniqueArtist, artistCode, inplace=True)
        
    def splitData(self, val_size = 0.2, test_size = 0.1, shuffle = True, random_seed = 42, fraction = 1):
        self.valSize = val_size
        self.testSize = test_size
        dataset_size = len(self.metaDF)
        indices = list(range(dataset_size))
        split_test = int(np.floor(self.testSize * dataset_size))
        split_val = int(np.floor(self.valSize * dataset_size))

        if shuffle :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[split_test+split_val:], indices[split_test:split_test+split_val],indices[:split_test]
        self.trainDF = self.metaDF.iloc[train_indices[:int(len(train_indices)*fraction)]]
        self.valDF = self.metaDF.iloc[val_indices[:int(len(val_indices)*fraction)]]
        self.testDF = self.metaDF.iloc[test_indices[:int(len(test_indices)*fraction)]]

    def getStat(self):
        # Get the statistic for train set:
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        if self.trainDataset == None:
            print("There is no training set")
            self.trainMean = None
            self.trainStd = None

        for data in self.trainDataLoader:
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(data["image"], dim=[0,2,3])
            channels_squared_sum += torch.mean(data["image"]**2, dim=[0,2,3])
            num_batches += 1
    
        self.trainMean = channels_sum / num_batches
        # std = sqrt(E[X^2] - (E[X])^2)
        self.trainStd = (channels_squared_sum / num_batches - self.trainMean ** 2) ** 0.5

    def UseNormalizedTransformer(self, normalize = True):
        self.UseNormalized = normalize

        if type(self.trainDF) != type(None):
            self.loadData()

    def setTransformer(self, transform):
        self.trainTransform = transform["train"]
        self.valTransform = transform["val"]

    def loadData(self, reUseTrain = 1):
        if type(self.trainDF) != type(None):
            self.trainDataset = CustomizedDataset(self.trainDF, self.dataPath, transform=self.trainTransform)
            self.valDataset = CustomizedDataset(self.valDF, self.dataPath, transform=self.valTransform)
            self.testDataset = CustomizedDataset(self.testDF, self.dataPath, transform= self.valTransform)

        if type(self.trainDF) != type(None):
            self.trainDataset = CustomizedDataset(self.trainDF, self.dataPath, transform=self.trainTransform, reUse = self.reUseTrain)
            self.valDataset = CustomizedDataset(self.valDF, self.dataPath, transform=self.valTransform)
            self.testDataset = CustomizedDataset(self.testDF, self.dataPath, transform= self.valTransform)

            classCounter = []
            for i in self.artistMap.keys():
                classCounter.append(len(self.trainDF[self.trainDF["Artist"]==i]))
            totalCounter = sum(classCounter)
            classWeight = [1]*len(classCounter)
            for i in self.artistMap.keys():
                classWeight[i] = totalCounter/classCounter[i]

            labels = self.trainDF["Artist"].values
            weightList = [classWeight[i] for i in labels]
            weightList *= self.reUseTrain
            #print(weightList)
            self.trainSampler = WeightedRandomSampler(weightList, num_samples = len(weightList) , replacement=True)
            #self.valSampler = SubsetRandomSampler(val_indices)
            self.datasetSize["train"] = len(self.trainDF)
            self.datasetSize["val"] = len(self.valDF)
            self.datasetSize["test"] = len(self.testDF)

            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, sampler = self.trainSampler)

        

    def checkDataset(self,size = (3,224,224)):
        reLoadFlag = False

        hardCodedList = [
            "53e872e09c40a912e36d53daf6920243",
            "53e872e09c40a912e36d53daf6920243",
            "53e872e09c40a912e36d53daf6920243",
            "0641ceb25ff823cf52802ea8c07558b1",
            "ee7c58e67dabd4b7459bddf9f669f707",
            "84e717a6cf749d681ffbb2557758bd13",
            "7dc6002a966d35a49f263376c2a56256",
            "8c9f5633d217a54f78ed5f2146324b38",
            "632fc73d4e2de07f28d85429ff390476",
            "b96a4a83299a54b4be4ff78c0d92c170",
            "a2bbef5813e42924f2168bef12a26478",
            "9d694f7ee70d5f694dda849a0a132bcb",
            "d7999f4ec0a7efcc004c733e813ddb39",
            "0be2cd049d2ecf361c6ffcb1c5054de5",
            "e2ee5d0fd45f95074b7e5e8665ab22de",
            "bcbf898f298ba2a00e088c50e3e27134",
            "e5f0411ee34ba439116a0d8d8cfe64c3",
            "042faaee15402a071df98949a2fe4298",
            "521cde1dc6bf903b92698cce5fc45cc9",
            "19527a307f8328d218c0d592b0e62eb9",
            "04e84f850d81a2e0a7440e13b46744e4",
            "66855fd1af6fcc7eb4ed461d5c82a810",
            "f5ef2960e0f5227289fb2e126ce0b4c5",
            "a2e7131ada2d8c94340e87dde1fbee69",
            "c9250c6bebf980301c38dc5735024fba",
            "db1d700c1a5c2bd9fdede96e65032757",
            "b808f9e26122a39d1f22814ca8f21db6",
        ]
        
        for index, data in enumerate(self.trainDataset):
                inputs = data["image"]
                labels = data["artist"]
                if tuple(inputs.shape) != size:
                    reLoadFlag = True
                    print(data["hash"],inputs.shape)
                    self.trainDF = self.trainDF.drop(self.trainDF[self.trainDF["hash"]==data["hash"]].index)

        for index, data in enumerate(self.valDataset):
                inputs = data["image"]
                labels = data["artist"]
                if tuple(inputs.shape) != size:
                    reLoadFlag = True
                    print(data["hash"],inputs.shape)
                    self.valDF = self.valDF.drop(self.valDF[self.valDF["hash"]==data["hash"]].index)

        for index, data in enumerate(self.testDataset):
                inputs = data["image"]
                labels = data["artist"]
                if tuple(inputs.shape) != size:
                    reLoadFlag = True
                    print(data["hash"],inputs.shape)
                    self.testDF = self.testDF.drop(self.testDF[self.testDF["hash"]==data["hash"]].index)  

        self.loadData()


    def setBagging(self, models, transformers, weights):
        self.baggingModels = models
        self.baggingTransformer = transformers
        self.baggingWeights = weights

        for index,model in enumerate(self.baggingModels):
            model.load_state_dict(torch.load(self.baggingWeights[index],map_location=self.device),strict=False)
    
    def prepareInputForVoter(self, skipNorm = False):
        self.skipNorm = skipNorm
        self.voter_input = []
        self.voter_label = []
        self.baseModelpreds = []
        self.labels = []
        for modelid in range(len(self.baggingModels)):
            model = self.baggingModels[modelid]
            #model.load_state_dict(torch.load(self.baggingWeights[modelid],map_location=self.device),strict=False)
            #transformer = self.baggingTransformer[modelid]['val']
            transformer = copy.deepcopy(self.baggingTransformer[modelid]['val'])
            if not self.skipNorm:
                transformer.transforms.append(transforms.Normalize(mean = tuple(self.trainMean.tolist()), std=tuple(self.trainStd.tolist())) )
            predMap = {}
            labelMap = {}
            with torch.no_grad():
                model.eval()
                for index in tqdm(self.valDF.index):
                    imgFile = self.valDF['hash'][index]
                    labal = self.valDF['Artist'][index]
                    img_name = os.path.join(self.dataPath,imgFile+'.jpg')
                    image = Image.open(img_name)
                    if image.mode == 'P':
                        image = image.convert('RGB')
                    image = transformer(image)

                    inputs = image.to(self.device)
                    inputs = torch.unsqueeze(inputs, dim=0)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    predMap[imgFile] = preds.tolist()
                    labelMap[imgFile] = labal

            self.baseModelpreds.append(predMap)
            self.labels.append(labelMap)

        for modelId in range(len(self.baseModelpreds)):
            predicts = self.baseModelpreds[modelId]
            labels = self.labels[modelId]

            nCorrect = 0
            nTotal = 0
            for key in predicts.keys():
                if predicts[key] == labels[key]:
                    nCorrect += 1
                nTotal += 1
            print("Sanity check for model:{}, accuracy:{}".format(modelId,nCorrect/nTotal))

        for index in self.valDF.index:
            hashId = self.valDF['hash'][index]
            votes = []
            for baseMap in self.baseModelpreds:
                votes += baseMap[hashId]
            self.voter_input.append(votes)
            self.voter_label.append(self.labels[0][hashId])

    def setVoterClassifier(self, classifierMap):
        self.classifierMap = classifierMap

    
    def fitVoter(self, useKFold = True, nKFold = 5):
        X = pd.DataFrame(self.voter_input)
        y = pd.DataFrame(self.voter_label)

        for key in self.classifierMap.keys():
            voter = self.classifierMap[key]

            if useKFold:
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)

                scores = []

                for train_index, test_index in kfold.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model = voter
                    model.fit(X_train.copy(), y_train[0].values)
    
                    y_pred = model.predict(X_test)
    
                    accuracy = accuracy_score(y_test[0].values, y_pred)
                    print(key, "accuracy:", accuracy)
                    if self.highaccuracy == None or accuracy > self.highaccuracy:
                        self.highaccuracy = accuracy
                        self.bestVoter = model

            else:

                model = voter
                model.fit(X.copy(), y[0].values)
    
                y_pred = model.predict(X)
    
                accuracy = accuracy_score(y[0].values, y_pred)
                print(key, "accuracy:", accuracy)
                if self.highaccuracy == None or accuracy > self.highaccuracy:
                    self.highaccuracy = accuracy
                    self.bestVoter = model

    
    def evaluateVoter(self, num = -1):
        self.votingResults = []
        self.votingLabels = []
        self.votingInputs = []
        if num > 0:
            testDF = self.testDF.head(num)
        else:
            testDF = self.testDF
        for index in tqdm(testDF.index):
            imgFile = testDF['hash'][index]
            labal = testDF['Artist'][index]
            img_name = os.path.join(self.dataPath,imgFile+'.jpg')
            image = Image.open(img_name)
            votingVector = []
            if image.mode == 'P':
                image = image.convert('RGB')

            for modelid in range(len(self.baggingModels)):
                model = self.baggingModels[modelid]
                #model.load_state_dict(torch.load(self.baggingWeights[modelid],map_location=self.device),strict=False)
                with torch.no_grad():
                    model.eval()
                    
                    #transformer = self.baggingTransformer[modelid]['val']
                    transformer = copy.deepcopy(self.baggingTransformer[modelid]['val'])
                    if not self.skipNorm:
                        transformer.transforms.append(transforms.Normalize(mean = tuple(self.trainMean.tolist()), std=tuple(self.trainStd.tolist())) )
                    imageTensor = transformer(image)

                    inputs = imageTensor.to(self.device)
                    inputs = torch.unsqueeze(inputs, dim=0)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    votingVector += preds.tolist()
            self.votingInputs.append(votingVector)

            votingPredict = self.bestVoter.predict([votingVector])

            self.votingResults.append(votingPredict)
            self.votingLabels.append(labal)
        accuracy = accuracy_score(self.votingLabels, self.votingResults)
        print("voter accuracy:", accuracy)

        artistList = [self.artistMap[i] for i in range(6)]
        cfMatrix = confusion_matrix(self.votingLabels, self.votingResults, normalize = 'true')
        dfcfMatrix = pd.DataFrame(cfMatrix, index=artistList,
                         columns=artistList)
        plt.figure(figsize=(12, 7))    
        sns.heatmap(dfcfMatrix, annot=True).get_figure()
        plt.savefig("VoterConfusionMatrix.jpg")

        for i in range(len(self.votingInputs[0])):
            accuracy = accuracy_score(np.array(self.votingInputs)[:,i], self.votingLabels)
            print("base model: {} accu: {}".format(i,accuracy))

        copyOfInput = copy.deepcopy(self.votingInputs)
        mostFreqVoting = pd.DataFrame(copyOfInput).mode(axis=1)[0]
        accuracy = accuracy_score(mostFreqVoting.values,self.votingLabels)
        print("hard voting accu: {}".format(accuracy))

        cfMatrix = confusion_matrix(self.votingLabels, mostFreqVoting.values, normalize = 'true')
        dfcfMatrix = pd.DataFrame(cfMatrix, index=artistList,
                         columns=artistList)
        plt.figure(figsize=(12, 7))    
        sns.heatmap(dfcfMatrix, annot=True).get_figure()
        plt.savefig("HardVotingConfusionMatrix.jpg")



if __name__ == "__main__":
    myObj = genericCNN()


