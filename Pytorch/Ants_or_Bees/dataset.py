from lib import *
from image_transform import ImageTransform
from utils import make_datapath_list


resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]
        
        if label == "nude":
            label = 0
        elif label == "sexs":
            label = 1
        
        return img_transformed, label

train_list = make_datapath_list("train")
val_list = make_datapath_list("val")


train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
print(len(train_dataset))