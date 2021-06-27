
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, BatchSampler
import pandas as pd
import numpy as np
from sklearn import preprocessing

def payload_to_tensor(payload):
    data = np.empty([1,3000],dtype = float)
    # for index in enumerate(payload):
    #     data[index] = float(int(num,16))
    for i in range(0,len(payload)-2,2):
        j = i // 2
        data[0][j] = (float(int(payload[i],16) * 16 + int(payload[i+1], 16)))
    return torch.from_numpy(data)

def normalize(payload):
    data = np.empty([1, 3000], dtype=float)
    for i in range(0,3000,2):
        data[i] = int(payload[i],16) * 16 + int(payload[i+1],16)
    data_norm = preprocessing.normalize(data)
    return data_norm

class EmbeddingTriDataSet(Dataset):
    def __init__(self,train_path,test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        self.train_labels = self.train['label']
        self.train_data = self.train['payload']
        self.test_labels = self.test['label']
        self.test_data = self.test['payload']
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label : np.where(self.train_labels == label)[0]
                                for label in self.labels_set}

    def __getitem__(self, index):
        payload1, label1 = self.train_data[index], self.train_labels[index]
        payload1 = payload_to_tensor(payload1)
        return payload1,label1


    def __len__(self):
        return len(self.train)   # need add test

class EmbeddingTest(Dataset):
    def __init__(self,train_path,test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        self.train_labels = self.train['label']
        self.train_data = self.train['payload']
        self.test_labels = self.test['label']
        self.test_data = self.test['payload']
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label : np.where(self.test_labels == label)[0]
                                for label in self.labels_set}

    def __getitem__(self, index):
        payload1, label1 = self.test_data[index], self.test_labels[index]
        payload1 = payload_to_tensor(payload1)
        return payload1,label1


    def __len__(self):
        return len(self.test)   # need add test



class TrainTriDataSet(Dataset):
    def __init__(self,train_path,test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        self.train_labels = self.train['label']
        self.train_data = self.train['payload']
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label : np.where(self.train_labels == label)[0]
                                for label in self.labels_set}

    def __getitem__(self, index):
        payload1, label1 = self.train_data[index], self.train_labels[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        payload2 = self.train_data[positive_index]
        payload3 = self.train_data[negative_index]
        payload1 = payload_to_tensor(payload1)
        payload2 = payload_to_tensor(payload2)
        payload3 = payload_to_tensor(payload3)
        return (payload1,payload2,payload3), []


    def __len__(self):
        return len(self.train)   # need add test

class TestDataSet(Dataset):
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        self.train_labels = self.train['label']
        self.train_data = self.train['payload']
        self.labels_set = set(self.train_labels)

        self.test_labels = self.test['label']
        self.test_data = self.test['payload']
        self.test_set = set(self.test_labels)
        self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                 for label in self.test_set}
        random_state = np.random.RandomState(29)
        triplets = [[i,
                     random_state.choice(self.label_to_indices[self.test_labels[i]]),
                     random_state.choice(self.label_to_indices[
                                             np.random.choice(
                                                 list(self.labels_set - set([self.test_labels[i]]))
                                             )
                                         ])
                     ]
                    for i in range(len(self.test_data))]
        self.test_triplets = triplets

    def __getitem__(self, index):
        payload1 = self.test_data[self.test_triplets[index][0]]
        payload2 = self.test_data[self.test_triplets[index][1]]
        payload3 = self.test_data[self.test_triplets[index][2]]
        payload1 = payload_to_tensor(payload1)
        payload2 = payload_to_tensor(payload2)
        payload3 = payload_to_tensor(payload3)
        return (payload1, payload2, payload3), []

    def __len__(self):
        return len(self.test)



class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

