import argparse
import numpy as np
import pickle
import random
import torch
from model import LogisticRegression
from torch.autograd import Variable
from utils import node_num_map

def train(feat_data, labels, train_idx, model, criterion, optimizer, epochs):
    '''
    train a logistic regression classifier with L2 regularization
    '''
    for epoch in range(epochs):
        random.shuffle(train_idx)
        correct = 0
        for idx in train_idx:
            feature = feat_data[idx:idx+1, :]
            label = torch.LongTensor(labels[idx])
            outputs = model(Variable(torch.FloatTensor(feature)))
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum()
        print("Epoch {epoch}, Accuracy: {accuracy}".format(
            epoch=epoch,
            accuracy=100 * correct / len(train_idx)
        ))

def evaluate(feat_data, labels, test_idx, model, criterion):
    '''
    evaluate the logistic regression classifier
    '''
    correct = 0

    for idx in test_idx:
        feature = feat_data[idx:idx+1, :]
        label = torch.LongTensor(labels[idx])
        outputs = model(Variable(torch.FloatTensor(feature)))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum()

    accuracy = 100 * correct / len(test_idx)
    print("Accuracy: {}".format(accuracy))

    return accuracy

def load_dataset(dataset, embed_dim):
    '''
    load features, labels and create train/test indices
    '''
    num_nodes = node_num_map[dataset]
    feat_data = np.zeros((num_nodes, embed_dim))
    labels = np.empty((num_nodes,1), dtype=np.int64)

    node_map = {}
    label_map = {}
    label_node_list_map = {}
    train_idx = []
    test_idx = []

    with open("{dataset}-airports/labels-{dataset}-airports.txt".format(
        dataset=dataset
    )) as fp:
        fp.readline()
        for i, line in enumerate(fp):
            info = line.strip().split() 
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
                label_node_list_map[len(label_map)-1] = []
            labels[i] = label_map[info[-1]]
            label_node_list_map[labels[i][0]].append(i)

    for label, node_list in label_node_list_map.items():
        node_list_size = len(node_list)
        anchor = int(node_list_size * 0.8)
        random.shuffle(node_list)
        train_idx.extend(node_list[:anchor])
        test_idx.extend(node_list[anchor:])

    with open("../struc2vec/emb/{dataset}-airports.emb".format(
        dataset=dataset
    )) as fp:
        fp.readline()
        for i, line in enumerate(fp):
            info = line.strip().split() 
            node_idx = node_map[info[0]]
            feat_data[node_idx, :] = np.array([float(i) for i in info[1:]]) 
    
    return feat_data, labels, train_idx, test_idx

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="usa",
                        help="airport dataset to run experiment on")

    parser.add_argument("--embed_dim", type=int, default=128,
                        help="struc2vec output embedding dimesnion")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")

    parser.add_argument("--l2", type=float, default=0.1,
                        help="L2 regularization")

    args = parser.parse_args()
    dataset = args.dataset
    embed_dim = args.embed_dim
    epochs = args.epochs
    lr_rate = args.lr
    l2 = args.l2
    num_classes = 4
    
    test_accuracy = []

    for i in range(10):
        print("Experiment {}".format(i))
        model = LogisticRegression(embed_dim, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, weight_decay=l2)

        feat_data, labels, train_idx, test_idx = load_dataset(dataset, embed_dim)

        train(feat_data, labels, train_idx, model, criterion, optimizer, epochs)
        test_acc = evaluate(feat_data, labels, test_idx, model, criterion)

        test_accuracy.append(test_acc)

    print("Average performance: {}".format(np.average(test_accuracy)))

if __name__ == "__main__":
    main()