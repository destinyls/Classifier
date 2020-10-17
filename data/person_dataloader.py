import os
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data.person_dataset import PersonDataset

IMAGE_FOLDER = 'dataset/'
def get_image_path(filename):
    return (IMAGE_FOLDER + filename)

def build_dataloader(data_root, batch_size=1):    
    test = pd.read_csv(os.path.join(data_root, 'test.csv'))
    test_paths = test['image_path'].apply(get_image_path)
    test_dataset = PersonDataset(test_paths, train = False, test = True)
    testloader = Data.DataLoader(test_dataset, shuffle=False, batch_size = batch_size, num_workers = 0)

    train = pd.read_csv(os.path.join(data_root, 'train_plus_labeled_test.csv'))
    train_paths = train['image_path'].apply(get_image_path)
    train_labels = train.loc[:, 'smoking_images':'normal_images']
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size = 0.2, random_state=23, stratify = train_labels)
    train_paths.reset_index(drop=True,inplace=True)
    train_labels.reset_index(drop=True,inplace=True)
    valid_paths.reset_index(drop=True,inplace=True)
    valid_labels.reset_index(drop=True,inplace=True)
    
    train_dataset = PersonDataset(train_paths, train_labels)
    trainloader = Data.DataLoader(train_dataset, shuffle=True, batch_size = batch_size, num_workers = 4)
    valid_dataset = PersonDataset(valid_paths, valid_labels)
    validloader = Data.DataLoader(valid_dataset, shuffle=False, batch_size = batch_size, num_workers = 4)
    return trainloader, validloader, testloader

    
    
    
