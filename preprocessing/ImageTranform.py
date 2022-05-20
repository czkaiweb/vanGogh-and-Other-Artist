import torch
from PIL import Image
from torchvision import transforms

class ImageTransformer():
    def __init__(self,output_shape):
        if type(output_shape) != tuple or type(output_shape) != tuple:
            pass
        self.H,self.W = output_shape
        self.transform = None

    def initTransform(self):
        self.transform ={
            "train":  transforms.Compose([
                transforms.Resize(260),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            "val": 
                transforms.Compose([
                transforms.Resize(260),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        }
        
    def getTransformer(self):
        return self.transform


if __name__ == "__main__":
    transformer = ImageTransformer()