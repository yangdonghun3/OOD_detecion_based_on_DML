import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        #test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        #test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    return testsetout

class SiameseDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, target_dataset):
        # # of channel check
        if len(target_dataset.data.shape)==4:
            self.channel3_flag = True
        elif len(target_dataset.data.shape)==3:
            self.channel3_flag = False
        
        self.target_dataset = target_dataset
        try:
            self.train = self.target_dataset.train
            self.init_like_mnist()
        except:
            self.init_like_svhn()
            
    def init_like_svhn(self):    
        if self.target_dataset.split=='train':
            self.train = True
        else:
            self.train = False
        self.transform = self.target_dataset.transform

        if self.train:
            self.train_labels = torch.from_numpy(self.target_dataset.labels)
            self.train_data = torch.from_numpy(self.target_dataset.data)
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = torch.from_numpy(self.target_dataset.labels)
            self.test_data = torch.from_numpy(self.target_dataset.data)
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.test_labels[i].item()]), 1] for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i, random_state.choice(self.label_to_indices[ np.random.choice(list(self.labels_set - set([self.test_labels[i].item()]))) ]), 0]  
                                         for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
            
    def init_like_mnist(self):    
        self.train = self.target_dataset.train
        self.transform = self.target_dataset.transform

        if self.train:
            try:
                self.train_labels = self.target_dataset.train_labels
                self.train_data = self.target_dataset.train_data
            except:
                self.train_labels = torch.FloatTensor(self.target_dataset.targets)
                self.train_data = torch.FloatTensor(self.target_dataset.data)
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            try:
                self.test_labels = self.target_dataset.test_labels
                self.test_data = self.target_dataset.test_data
            except:
                self.test_labels = torch.FloatTensor(self.target_dataset.targets)
                self.test_data = torch.FloatTensor(self.target_dataset.data)
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.test_labels[i].item()]), 1] for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i, random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([self.test_labels[i].item()])))]), 0]  
                                         for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        if self.channel3_flag==True:
            img1 = Image.fromarray(img1.numpy(), mode='RGB')
            img2 = Image.fromarray(img2.numpy(), mode='RGB')
        else:
            img1 = Image.fromarray(img1.numpy(), mode='L')
            img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), target

    def __len__(self):
        return len(self.target_dataset)


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, target_dataset):
        # # of channel check
        if len(target_dataset.data.shape)==4:
            self.channel3_flag = True
        elif len(target_dataset.data.shape)==3:
            self.channel3_flag = False
        
        self.target_dataset = target_dataset
        try:
            self.train = self.target_dataset.train
            self.init_like_mnist()
        except:
            self.init_like_svhn()
    
    def init_like_svhn(self):
        if self.target_dataset.split=='train':
            self.train = True
        else:
            self.train = False
        self.transform = self.target_dataset.transform

        if self.train:
            self.train_labels = torch.from_numpy(self.target_dataset.labels)
            self.train_data = torch.from_numpy(self.target_dataset.data)
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = torch.from_numpy(self.target_dataset.labels)
            self.test_data = torch.from_numpy(self.target_dataset.data)
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets
    
    def init_like_mnist(self):
        self.train = self.target_dataset.train
        self.transform = self.target_dataset.transform

        if self.train:
            try:
                self.train_labels = self.target_dataset.train_labels
                self.train_data = self.target_dataset.train_data
            except:
                self.train_labels = torch.FloatTensor(self.target_dataset.targets)
                self.train_data = torch.FloatTensor(self.target_dataset.data)
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            try:
                self.test_labels = self.target_dataset.test_labels
                self.test_data = self.target_dataset.test_data
            except:
                self.test_labels = torch.FloatTensor(self.target_dataset.targets)
                self.test_data = torch.FloatTensor(self.target_dataset.data)
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        if self.channel3_flag==True:
            img1 = Image.fromarray(img1.numpy(), mode='RGB')
            img2 = Image.fromarray(img2.numpy(), mode='RGB')
            img3 = Image.fromarray(img3.numpy(), mode='RGB')
        else:
            img1 = Image.fromarray(img1.numpy(), mode='L')
            img2 = Image.fromarray(img2.numpy(), mode='L')
            img3 = Image.fromarray(img3.numpy(), mode='L')
            
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.target_dataset)
    
    
def CrossEntropyDataset(Dataset):
    return Dataset
        