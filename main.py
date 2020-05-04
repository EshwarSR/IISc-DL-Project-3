import time
import torch
import RNN_models
import os
import torch.optim as optim
import torch.nn as nn
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from torchtext import data
from torchtext import datasets
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def load_data_bert(filepath):
    inputs = []
    labels = []
    token_type_ids = []
    max_len = 0
    with open(filepath) as f:
        for sample in f:
            tti = []
            jsn = json.loads(sample)
            encoded_sent1 = tokenizer.encode(
                jsn["sentence1"],
                add_special_tokens=True)
            tti.extend([0]*len(encoded_sent1))
            encoded_sent2 = tokenizer.encode(
                jsn["sentence2"],
                add_special_tokens=True)
            tti.extend([1]*(len(encoded_sent2) - 1))

            encoded_sents = encoded_sent1 + encoded_sent2[1:]
            if len(encoded_sents) > max_len:
                max_len = len(encoded_sents)
            inputs.append(encoded_sents)
            labels.append(CLASSES[jsn["annotator_labels"][0]])
            token_type_ids.append(tti)

    return inputs, labels, token_type_ids, max_len


def evaluate_bert(model, data):
    all_y_truth = []
    all_y_pred = []
    loss_meter = AverageMeter("loss")
    for batch in data:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

        loss, logits = model(b_input_ids,
                             token_type_ids=b_token_type_ids,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        _, pred_classes = torch.max(logits, 1)
        all_y_pred.extend(pred_classes.tolist())
        all_y_truth.extend(b_labels.tolist())
        loss_meter.update(loss, b_labels.shape[0])
    acc = accuracy_score(all_y_truth, all_y_pred)
    return acc, loss_meter.avg, all_y_pred, all_y_truth


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


def rnn_evaluate(model, data, loss_fn):
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


def load_data(filepath):
    inputs = []
    labels = []
    with open(filepath) as f:
        for sample in f:
            jsn = json.loads(sample)
            inputs.append(jsn["sentence1"] + " " + jsn["sentence2"])
            labels.append(CLASSES[jsn["annotator_labels"][0]])

    return inputs, labels


def evaluate(X, y, count_model, tfidf_model, lr_model):
    X_tfidf = tfidf_model.transform(count_model.transform(X))
    y_pred = lr_model.predict(X_tfidf)
    accuracy = (y_pred == y).mean()
    return accuracy, y_pred


def get_train_valid_test_data(batch_size, device):
    inputs = data.Field(lower=True, tokenize='spacy')
    answers = data.Field(sequential=False)

    train, valid, test = datasets.SNLI.splits(inputs, answers)
    inputs.build_vocab(train, valid, test)
    answers.build_vocab(train)

    train_data, validation_data, test_data = data.BucketIterator.splits(
        (train, valid, test), batch_size=batch_size, device=device)
    return test_data, inputs, answers


def get_attention_masks(X):
    attention_masks = []
    for sample in X:
        att_mask = [int(token_id > 0) for token_id in sample]
        attention_masks.append(att_mask)

    return attention_masks

###########
# Configs #
###########


batch_size = 128

device = torch.device('cpu')
print("Running on the device", device)

CLASSES = {"entailment": 0, "contradiction": 1, "neutral": 2}
INV_CLASSES = {0: "entailment", 1: "contradiction", 2: "neutral"}
MAX_LEN = 128

############################
# Loading Dataset & Models #
############################
data_st = time.time()
test_data, inputs, answers = get_train_valid_test_data(batch_size, device)
"""
vocab_length = len(inputs.vocab)
num_classes = len(answers.vocab)

rnn_model = RNN_models.RNN_3L_400_notfixed_notprojected_bi_3L(
    vocab_length, num_classes)

rnn_model.load_state_dict(torch.load(
    "models/" + rnn_model.__class__.__name__+".pt", map_location=device))
rnn_model.eval()

loss_fn = nn.CrossEntropyLoss()
"""
# TF-IDF Model
X_test, y_test = load_data(".data/snli/snli_1.0/snli_1.0_test.jsonl")

MODEL_NAME = 'LR_unprocessed'
model_folder = "models/" + MODEL_NAME + "/"
count_model = pickle.load(open(model_folder+"count_model.pkl", "rb"))
tfidf_model = pickle.load(open(model_folder+"tfidf_model.pkl", "rb"))
lr_model = pickle.load(open(model_folder+"lr_model.pkl", "rb"))

#------ BERT ------#
# Model
output_dir = "./bert_model/"
bert_model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Data
X_test_b, y_test_b, token_type_ids_test, max_len_test = load_data_bert(
    ".data/snli/snli_1.0/snli_1.0_test.jsonl")

X_test_b = pad_sequences(X_test_b, maxlen=MAX_LEN, dtype="long",
                         value=0, truncating="post", padding="post")
token_type_ids_test = pad_sequences(token_type_ids_test, maxlen=MAX_LEN, dtype="long",
                                    value=1, truncating="post", padding="post")

att_masks_test = get_attention_masks(X_test_b)

X_test_b = torch.tensor(X_test_b)
y_test_b = torch.tensor(y_test_b)
att_masks_test = torch.tensor(att_masks_test)
token_type_ids_test = torch.tensor(token_type_ids_test)


test_data_b = TensorDataset(
    X_test_b, att_masks_test, token_type_ids_test, y_test_b)
test_sampler = SequentialSampler(test_data_b)
test_dataloader = DataLoader(
    test_data_b, sampler=test_sampler, batch_size=batch_size)


####################
# Tesinging Models #
####################
"""
acc, loss, all_y_pred, all_y_truth = rnn_evaluate(
    rnn_model, test_data, loss_fn)
print("Deep Model Accuracy:", round(acc, 3))
# print("Loss:", loss)

class_labels = answers.vocab.itos
with open("deep_model.txt", "w") as f:
    pred_labels = [class_labels[idx] for idx in all_y_pred]
    f.write("\n".join(pred_labels))
"""
# TF-IDF Model
acc, y_pred = evaluate(X_test, y_test, count_model, tfidf_model, lr_model)
print("TF-IDF LR Model Accuracy:", round(acc, 3))
with open("tfidf.txt", "w") as f:
    pred_labels = [INV_CLASSES[item] for item in y_pred]
    f.write("\n".join(pred_labels))

# BERT Model
test_acc, test_loss, test_y_pred, test_y_truth = evaluate_bert(
    bert_model, test_dataloader)
print("BERT Model Accuracy:", round(acc, 3))
with open("deep_model.txt", "w") as f:
    pred_labels = [INV_CLASSES[item] for item in test_y_pred]
    f.write("\n".join(pred_labels))
