from preprocessing.ImageTranform import *
from preprocessing.CustomizedDataset import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch
import time
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

class genericCNN():
    def __init__(self):
        self.metaData = None
        self.metaDF = None
        self.trainDF = None
        self.valDF = None
        self.artistMap = {}


        self.trainTransform = None
        self.valTransform = None
        self.NetWork = None
        self.Model = None
        self.ModelTag = None

        self.Dataset = None
        self.valSize = 0.2
        self.trainDataset = None
        self.valDataset = None
        self.trainDataLoader = None
        self.valDataLoader = None
        self.datasetSize = {}

        self.batch_size = 5
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def setTransformer(self, transform):
        self.trainTransform = transform["train"]
        self.valTransform = transform["val"]

        if self.trainDataLoader != None:
            self.trainDataset = CustomizedDataset(self.trainDF, self.dataPath, transform=self.trainTransform)
            self.valDataset = CustomizedDataset(self.valDF, self.dataPath, transform=self.valTransform)
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size)
            self.valDataLoader = DataLoader(self.valDataset, batch_size=self.batch_size)


    
    def setBatchSize(batch_size = 5):
        self.batch_size = batch_size
        if self.trainDataLoader != None:
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size)
            self.valDataLoader = DataLoader(self.valDataset, batch_size=self.batch_size)
    
    def setDataset(self, metadata, path="../data/imgs"):
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
        
    def splitData(self, val_size = 0.2, shuffle = True, random_seed = 42, fraction = 1):
        self.valSize = val_size
        dataset_size = len(self.metaDF)
        indices = list(range(dataset_size))
        split = int(np.floor(self.valSize * dataset_size))
        if shuffle :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        self.trainDF = self.metaDF.iloc[train_indices[:int(len(train_indices)*fraction)]]
        self.valDF = self.metaDF.iloc[val_indices[:int(len(val_indices)*fraction)]]

    def loadData(self):
        if type(self.trainDF) != type(None):
            self.trainDataset = CustomizedDataset(self.trainDF, self.dataPath, transform=self.trainTransform)
            self.valDataset = CustomizedDataset(self.valDF, self.dataPath, transform=self.valTransform)

            #self.trainSampler = SubsetRandomSampler(train_indices)
            #self.valSampler = SubsetRandomSampler(val_indices)
            self.datasetSize["train"] = len(self.trainDF)
            self.datasetSize["val"] = len(self.valDF)
        
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size)
            self.valDataLoader = DataLoader(self.valDataset, batch_size=self.batch_size)
        elif type(self.metaDF) != type(None):
            print("No data split assigned, only train dataset will be used")
            self.trainDataset = CustomizedDataset(self.metaDF, self.dataPath, transform=self.trainTransform)
        else:
            print("No data found")

    def showDatasetBatch(self, tag = "train" ):
        if tag == "train":
            dataloader = self.trainDataLoader
        elif tag == "validation":
            dataloader = self.valDataLoader
        else:
            dataloader = self.trainDataLoader

        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch, sample_batched['image'].size(),
            sample_batched['artist'])

            # observe 4th batch and stop.
            if i_batch == 0:
                plt.figure()
                self.displayDatasetBatch(sample_batched)
                plt.axis('off')
                plt.ioff()
                plt.show()
                break

    def displayDatasetBatch(self, imageBatch):
            """Show image with landmarks for a batch of samples."""
            images_batch, artist_batch = imageBatch['image'], imageBatch['artist']
            print(images_batch.shape)
            batch_size = len(images_batch)
            im_size = images_batch.size(2)
            grid_border_size = 2

            f = plt.figure(figsize=(12, 12), dpi=80)
            for i in range(batch_size):
                # Debug, plot figure
                f.add_subplot(1, batch_size, i + 1)
                pilImg = transforms.ToPILImage()(images_batch[i])
                plt.imshow(np.asarray(pilImg))
                plt.title(artist_batch[i])

    def setModel(self,model, modeltag):
        self.ModelTag = modeltag
        self.Model = model

    def checkDataset(self,size = (3,224,224)):
        reLoadFlag = False
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
        if reLoadFlag:
            self.loadData()

    def train_model(self, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(self.Model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.Model.train()  # Set self.Mto training mode
                else:
                    self.Model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                if phase == 'train':
                    dataloader = self.trainDataLoader
                else:
                    dataloader = self.valDataLoader

                # Iterate over data.
                with tqdm(dataloader, unit="batch") as tepoch:
                    for sample_batched in tepoch:
                        tepoch.set_description(f"Epoch {epoch} Phase {phase}")
                        inputs = sample_batched["image"]
                        labels = sample_batched["artist"]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.Model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                            optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.datasetSize[phase]
                epoch_acc = running_corrects.double() / self.datasetSize[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.Model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.Model.load_state_dict(best_model_wts)
        return self.Model


    def evaluate(self):
        pass

if __name__ == "__main__":
    myObj = genericCNN()


