import sys

from dataset import TrainTriDataSet, TestDataSet, EmbeddingTriDataSet,EmbeddingTest
from trainer import fit
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

train_dataset = EmbeddingTriDataSet(r'D:\data\tri\all\train1.csv', r'D:\data\tri\all\test2.csv')
test_dataset = EmbeddingTest(r'D:\data\tri\all\train1.csv', r'D:\data\tri\all\test2.csv')
triplet_train_dataset = TrainTriDataSet(r'D:\data\tri\all\train1.csv', r'D:\data\tri\all\test2.csv')
triplet_test_dataset = TestDataSet(r'D:\data\tri\all\train1.csv', r'D:\data\tri\all\test2.csv')
n_classes = 8

mnist_classes = ['1', '2', '3', '4', '5', '6', '7', '8']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
          ]

res = np.zeros((n_classes,8))

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(8):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.show()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 8))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for payloads, target in dataloader:
            if cuda:
                payloads = payloads.cuda()
            embeddings[k:k+len(payloads)] = model.get_embedding(payloads).data.cpu().numpy()
            labels[k:k+len(payloads)] = np.array(target)
            for idx in np.array(target):
                res[idx] += model.get_embedding(payloads).data.cpu().numpy()[idx]
            # labels[k:k + len(payloads)] = np.array(target)
            k += len(payloads)
    for i in range(8):
        res[i] = res[i] / len(dataloader.dataset)
    return embeddings, labels

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

def accuracy(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 8))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for payloads, target in dataloader:
            if cuda:
                payloads = payloads.cuda()
            # embeddings[k:k + len(payloads)] = model.get_embedding(payloads).data.cpu().numpy()
            # labels[k:k + len(payloads)] = np.array(target)
            tar_arr = np.array(target).tolist()
            idx = 0
            for y_hat in tar_arr:
                sim = sys.maxsize
                y = -1
                out = model.get_embedding(payloads).data.cpu().numpy()[idx]
                for i in range(8):
                    cos_sim = cosine_distance(out,res[i])
                    if(cos_sim < sim):
                        sim = cos_sim
                        y = i
                print(str(y) + " " + str(y_hat))
            # labels[k:k + len(payloads)] = np.array(target)
                idx += 1
    return embeddings, labels
# Set up data loaders
batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric

# embedding_net = EmbeddingNet()
# model = ClassificationNet(embedding_net, n_classes=n_classes)
# if cuda:
#     model.cuda()
# loss_fn = torch.nn.NLLLoss()
# lr = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 50
#
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
# train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_baseline, train_labels_baseline)
# val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_baseline, val_labels_baseline)
#
#
#
#
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
#
# # Set up the network and training parameters
# from networks import EmbeddingNet, TripletNet
# from losses import TripletLoss
#
# margin = 1.
# embedding_net = EmbeddingNet()
# model = TripletNet(embedding_net)
# if cuda:
#     model.cuda()
# loss_fn = TripletLoss(margin)
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 100
#
#
# fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,metrics=[AccumulatedAccuracyMetric()])
#
# #torch.save(model,r'D:\data\tri')
# train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_tl, train_labels_tl)
# val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_tl, val_labels_tl)

from datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=8, n_samples=2)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=8, n_samples=2)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50
fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])
train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_otl, train_labels_otl)
accuracy(train_loader,model)
val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_otl, val_labels_otl)



# display_emb_online, display_emb, display_label_online, display_label = val_embeddings_otl, val_embeddings_tl, val_labels_otl, val_labels_tl
# x_lim = (np.min(display_emb_online[:,0]), np.max(display_emb_online[:,0]))
# y_lim = (np.min(display_emb_online[:,1]), np.max(display_emb_online[:,1]))
# plot_embeddings(display_emb, display_label, x_lim, y_lim)
# plot_embeddings(display_emb_online, display_label_online, x_lim, y_lim)