"""
source: 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://github.com/utkuozbulak/pytorch-custom-dataset-examples#custom-dataset-fundamentals
https://github.com/pytorch/tutorials/blob/45e4f2a1ca2f7b58a4f3f800fc817b1921bc45ff/recipes_source/recipes/custom_dataset_transforms_loader.py#L111
https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7

"""
from __future__ import print_function, division
import json
import os
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from skimage import io, transform

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ignore warnings
import warnings

from torchvision.transforms.transforms import RandomVerticalFlip
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BDDDataset(Dataset):
    def __init__(self, image_path, drivable_path, transform=None):
        """
        Args:
            label_path (string): pathto json file with annotations
            image_dir (string):   Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_path
        self.drivable_dir = drivable_path
        self.images = os.listdir(self.image_dir)
        self.drivable_label = os.listdir(self.drivable_dir)

        self.data_len = len(self.images)
        """
        #Read in image_names
        for i in range (len(self.labels)):
            self.image_names = str(self.labels[i].get("name"))
            self.

        self.labels = self.read_json_file(self.label_path)
        """

        #Transformations
        self.rescale = transforms.Resize((288, 512))
        self.vertical_flip = transforms.RandomVerticalFlip()
        # add_noise GaussianBlur
        #RandomCrop + get_params: i, j, h, w = transforms.RandomCrop.get_params(input, (100, 100)) input = F.crop(input, i, j, h, w) target = F.crop(target, i, j, h, w)
        #RandomErasing, adjust_brightness, adjust_contrast, gamme, hue
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.2783222,  0.29209027, 0.28937113], 
                                                std=[0.24401996, 0.2618693,  0.2725748 ])



    def __len__(self):
        return self.data_len
    
    def load_image(self, index):
        image_path = self.images[index]
        img = Image.open(image_path)
        return img
    
    def __getitem__(self, idx):

        single_image = self.images[idx]
        img_name = os.path.join(self.image_dir ,single_image)
        img_as_img = Image.open(os.path.join(self.image_dir ,single_image))
        img_as_tensor = self.rescale(img_as_img)
        img_as_tensor = self.to_tensor(img_as_tensor)
        img_as_tensor = self.normalize(img_as_tensor)

        # numpy image: H x W x C
        # torch image: C X H X W
        #print(idx)
        single_label = self.drivable_label[idx]
        label_as_img = Image.open(os.path.join(self.drivable_dir ,single_label))
        label_as_tensor = self.rescale(label_as_img)
        #label_as_tensor = self.to_tensor(label_as_tensor)
        label_as_tensor = torch.from_numpy(np.array(label_as_tensor)).float()
        label_as_tensor = label_as_tensor.unsqueeze(0)
        
        #numpy operatoren

        sample = {'image': img_as_tensor, 'label': label_as_tensor, 'image_name': img_name}
        #add new parameters maybe like this for tensorboard?:
        #sample["wheater"] = "clouded"
        return sample

    """
    #Functional Transforms - pytorch
    def my_vertical_flip(input, target):
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.vlip(image)
            segmentation = TF.vflip(segmentation)
        # more transforms ...
        return image, segmentation
    """

    """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.labels[idx, 0])
        image = io.imread(img_name)
        label = self.labels[idx, 1:]
        label = np.array([label])
        label = label.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def read_json_file(self, label_path):
        with open(label_path, "r") as read_file:
            self.labels = json.load(read_file)

    """

def mean_std(train_loader):
    #Calculates mean and std for dataset
    #Results may vary through batch sizes
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    #teilen durch 255
    for index, data in enumerate(train_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data['image'].numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

        if index % 100 == 0:
            print(index, "/", len(train_loader))

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    print("mean: ", pop_mean, "std0: ", pop_std0, "std1: ",pop_std1)

if __name__ == "__main__":
    # Call dataset
    working_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch"
    bdd100k_train_img_path = working_dir + "/dataset/images/train/"
    bdd100k_train_dl_path = working_dir + "/dataset/lanes_seg/train/"

    bdd100kdataset = BDDDataset(image_path=bdd100k_train_img_path, drivable_path = bdd100k_train_dl_path)
    train_loader = DataLoader(dataset=bdd100kdataset, batch_size=64, shuffle=True, num_workers=6)


    import sys


    def show_images(image, label,image_name):
        """Show image with landmarks"""
        np.set_printoptions(threshold=sys.maxsize)

        print(label[0,:100,:100].numpy())

        #plt.imshow(image.permute(1, 2, 0))
        #plt.pause(5)

        #print (label[0,120:200,260:360]numpy())
        #plt.imshow(label.view(288,512))
        #label = np.array(label)
        #print(label.shape)
       # print(label[150:230,450:512].numpy())
        #plt.imshow(label[150:230,450:512].data.numpy())
        plt.imshow(label.squeeze(0))

        #plt.scatter(label[:, 0], label[:, 1], s=10, marker='.', c='r')
        plt.pause(2)  # pause a bit so that plots are updated

    plt.ion()
    fig = plt.figure()

    for i in range(len(bdd100kdataset)):
        sample = bdd100kdataset[i+8]
        img = sample['image']
        segLabel = sample['label']
        #print(i, sample['image'].shape, sample['label'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_images(**sample)

        if i == 3:
            plt.show()
            break
        #target = torch.argmax(seg_gt, dim=1)
        #print(target.shape, target.dim())
       #print(target.view(target.size(0), -1).unique(dim=1))