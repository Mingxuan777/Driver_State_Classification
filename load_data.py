
from torchvision import transforms as T
from PIL import Image
from torch.utils import data
import os, random

random.seed(1)

def get_filepath(dir_root):
    # get root path of the files and store them in list file_paths
    file_paths = []
    for root, dirs, files in os.walk(dir_root):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class DriverDataset(data.Dataset):

    def __init__(self, data_root, args, transforms=None, train=True):
        self.train = train
        imgs_in = get_filepath(data_root)
        random.shuffle(imgs_in)
        imgs_num = len(imgs_in)

        if transforms is None and self.train:
            self.transforms = T.Compose([T.RandomHorizontalFlip(),
                                         T.RandomResizedCrop(224),
                                         T.ToTensor()])
        else:
            self.transforms = T.Compose([T.Resize(size=(224, 224)),
                                         T.ToTensor()])
        if self.train is True: # train set
            self.imgs = imgs_in[:int(args.traindata_size * imgs_num)]
            # print('test') # for debug.
        elif self.train is False:          # validation set
            self.imgs = imgs_in[int(args.valdata_size * imgs_num):]
            # print('test') # for debug

    def __getitem__(self, index):
        img_path = self.imgs[index]

        label = int(img_path.split('/')[-2][1])
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)