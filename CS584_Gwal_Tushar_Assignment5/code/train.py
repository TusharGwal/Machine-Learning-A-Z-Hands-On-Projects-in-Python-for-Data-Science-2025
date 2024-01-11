from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch,idx_train,idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
    return [epoch+1,loss_train.item(),acc_train.item(),loss_val.item(),acc_val.item(),time.time()-t]


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return [loss_test.item(),acc_test.item()]

# Train model
t_total = time.time()
resultsO = []
testResult = []
for i in [60, 120, 180,240,300]:

    results = []
    for epoch in range(args.epochs):
        results.append(train(epoch,idx_train=range(i),idx_val=range(i, 500)))

    results = np.asarray(results)
    resultsO.append(results)
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    testResult.append(test())
    titles = [
        "epoch vs Training loss",
        "epoch vs Training accuracy",
        "epoch vs Validation loss",
        "epoch vs validation accuracy",
        "epoch vs time"

    ]
    f = plt.figure(figsize=(20, 10))
    plt.suptitle("Results for training data in range of "+str(i)+" and validation data in range of "+str(i)+" to "+str(500))
    for j in range(5):
        plt.subplot(2, 3, j + 1)
        plt.plot(results[:,0],results[:,j+1])
        plt.title(titles[j])
    # plt.show()

    plt.savefig('/Users/tushargwal/Desktop/data'+str(i)+".jpg")


resultsO = np.asarray(resultsO)
testResult = np.asarray(testResult)
f2 = plt.figure(figsize=(20, 10))
# plt.suptitle("Results for training data in range of "+str(i)+" and validation data in range of "+str(i)+" to "+str(500))
plt.suptitle("Training and validation set results")
colors = ["green","red","yellow","blue","black"]
labels = [60, 120, 180,240,300]
for j in range(5):
    plt.subplot(2, 3, j + 1)
    plt.title(titles[j])
    for i in range(resultsO.shape[0]):
        plt.plot(resultsO[i][:,0],resultsO[i][:,j+1],color=colors[i],label="labeled data"+str(labels[i]))

plt.legend(scatterpoints=1, shadow=False, loc="upper right")
plt.savefig('/Users/tushargwal/Desktop/combined.jpg')

f3 = plt.figure(figsize=(10, 5))
plt.suptitle("Testing set results")
plt.subplot(1, 2, 1)
plt.plot(labels,testResult[:,0])
plt.title("labeled data vs testing loss")

plt.subplot(1, 2, 2)
plt.plot(labels,testResult[:,1])
plt.title("labeled data vs testing accuracy")

plt.savefig('/Users/tushargwal/Desktop/test.jpg')