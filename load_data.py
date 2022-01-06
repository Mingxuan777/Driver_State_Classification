# coding=utf-8
# load data file

from torchvision import transforms as T
from PIL import Image
from torch.utils import data
import os, random

random.seed(1) # make the code result reproducible 

# return directories of all files
def get_filepath(dir_root):
    # get root path of the files and store them in list file_paths
    file_paths = [] # filepath list, which is used to store filepath string.
    for root, dirs, files in os.walk(dir_root): # walk through all files in the root directory and get their file directory
        for file in files:
            file_paths.append(os.path.join(root, file))
            
    return file_paths

# load data class
class LoadDataset(data.Dataset):
    def __init__(self, data_root, args, transforms=None, train=True):
        self.train = train # accept training status
        imgs_in = get_filepath(data_root) # get image file
        random.shuffle(imgs_in) # shuffle images
        imgs_num = len(imgs_in) # use special method __len__ to return the length of images.

        # build transform (to transform the data to tensor) using Compose method
        if transforms is None and self.train: # for training
            self.transforms = T.Compose([T.RandomHorizontalFlip(), T.RandomResizedCrop(224), T.ToTensor()])
        else: # for validation
            self.transforms = T.Compose([T.Resize(size=(224, 224)), T.ToTensor()])

        # select training and validation dataset
        if self.train is True: # train set
            self.imgs = imgs_in[:int(args.traindata_size * imgs_num)] # list operation to cut training data 
            # print('test') # for debug
        elif self.train is False:   # validation set
            self.imgs = imgs_in[int(args.valdata_size * imgs_num):] # list operation to cut validation data
            # print('test') # for debug

    # special method (faster)
    def __len__(self):
        # return the length of image list
        return len(self.imgs)

    # special method (faster)
    def __getitem__(self, index):
        img_path = self.imgs[index] # use index to select image path
        label = int(img_path.split('/')[-2][1]) # use int(img_path.split('\\')[-2][1]) if you are using windows # get label, 0, 1, 2, ... 9
        data = Image.open(img_path) # read image
        data = self.transforms(data) # transform data to tensor

        return data, label # return data and label
