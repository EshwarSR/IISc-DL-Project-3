from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time
import scikitplot as skplt
import matplotlib.pyplot as plt
import os
import glob

import torch
import torch.nn as nn

from torchtext import data
from torchtext import datasets


def get_train_valid_test_data(word_vectors, batch_size, device):
    inputs = data.Field(lower=True, tokenize='spacy')
    answers = data.Field(sequential=False)

    train, valid, test = datasets.SNLI.splits(inputs, answers)

    inputs.build_vocab(train, valid, test)
    inputs.vocab.load_vectors(word_vectors)
    answers.build_vocab(train)

    train_data, validation_data, test_data = data.BucketIterator.splits(
        (train, valid, test), batch_size=batch_size, device=device)
    return train_data, validation_data, test_data, inputs, answers


def plot_confusion_matrices(model_name, epoch_num, train_y_truth, train_y_pred, validate_y_truth, validate_y_pred):
    plot_folder = "plots/models/" + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    # Train CM
    skplt.metrics.plot_confusion_matrix(
        train_y_truth, train_y_pred, normalize=True)
    plt.savefig(plot_folder + epoch_num + '_train_cm.png')
    plt.title(model_name+"_train_cm")
    plt.clf()
    plt.close()

    # Validation CM
    skplt.metrics.plot_confusion_matrix(
        validate_y_truth, validate_y_pred, normalize=True)
    plt.savefig(plot_folder + epoch_num + '_valid_cm.png')
    plt.title(model_name+"_valid_cm")
    plt.clf()
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def evaluate(model, data, loss_fn):
    data.init_epoch()
    all_y_truth = []
    all_y_pred = []
    loss_meter = AverageMeter("loss")
    for batch_idx, batch in enumerate(data):
        preds = model(batch)
        _, pred_classes = torch.max(preds, 1)
        loss = loss_fn(preds, batch.label).item()
        all_y_pred.extend(pred_classes.tolist())
        all_y_truth.extend(batch.label.tolist())
        loss_meter.update(loss, batch.label.shape[0])
    acc = accuracy_score(all_y_truth, all_y_pred)
    return acc, loss_meter.avg, all_y_pred, all_y_truth


def train_evaluate_model(model_name, model, optimizer, num_epochs, train_data, validation_data, evaluate_every, log_every_iterations):

    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    epoch_ticks = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_st = time.time()
        train_data.init_epoch()
        model.train()
        for batch_idx, batch in enumerate(train_data):
            # switch model to training mode, clear gradient accumulators
            optimizer.zero_grad()

            # forward pass
            answer = model(batch)

            # calculate loss of the network output with respect to training labels
            loss = loss_fn(answer, batch.label)

            # backpropagate and update optimizer learning rate
            loss.backward()
            optimizer.step()

            # print("Batch time:", time.time() - batch_st)
            if batch_idx % log_every_iterations == 0:
                # print progress message
                print("Time elapsed:", time.time() - start_time, "|",
                      "Epoch:", str(epoch + 1) + "/" + str(num_epochs), "|",
                      "Batch:", str(batch_idx+1) + "/" +
                      str(len(train_data)), "|",
                      "Batch Loss:", loss.item())

        if epoch % evaluate_every == (evaluate_every - 1):
            # switch model to evaluation mode
            model.eval()

            # calculate accuracy on validation set
            with torch.no_grad():
                print("Starting Evaluation for epoch:", epoch+1)
                st = time.time()
                # Train data set evaluation
                train_acc, train_loss, train_y_pred, train_y_truth = evaluate(
                    model, train_data, loss_fn)
                train_accuracies.append(train_acc)
                train_losses.append(train_loss)
                et = time.time()
                print("Time taken for train evaluation", et - st)

                # Validation data set evaluation
                st = time.time()
                validation_acc, validation_loss, validate_y_pred, validate_y_truth = evaluate(
                    model, validation_data, loss_fn)
                validation_accuracies.append(validation_acc)
                validation_losses.append(validation_loss)
                et = time.time()
                print("Time taken for validation evaluation", et - st)

                # Writing Confusion matrices
                plot_confusion_matrices(
                    model_name, str(epoch+1), train_y_truth, train_y_pred, validate_y_truth, validate_y_pred)

                epoch_ticks.append(epoch+1)

            # Printing Epoch Metrics
            print("\n\n"+"="*25)
            print("Time elapsed:", time.time() - start_time)
            print("Epoch:", str(epoch + 1) + "/" + str(num_epochs))
            print("Epoch time:", time.time() - epoch_st)
            print("Train Acc:", train_acc)
            print("Validation Acc:", validation_acc)
            print("Train Loss:", train_loss)
            print("Validation Loss:", validation_loss)
            print("="*25 + "\n\n")

            # Putting the model back to train mode
            model.train()

    # Saving the model
    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(model.state_dict(), model_dir + model_name + ".pt")

    return model, train_losses, validation_losses, train_accuracies, validation_accuracies, epoch_ticks
