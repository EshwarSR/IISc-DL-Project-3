import json
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import spacy
import scikitplot as skplt
import matplotlib.pyplot as plt
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


PoolExecutor = ThreadPoolExecutor

start_time = time.time()


def load_data(filepath):
    inputs = []
    labels = []
    with open(filepath) as f:
        for sample in f:
            jsn = json.loads(sample)
            inputs.append(jsn["sentence1"] + " " + jsn["sentence2"])
            labels.append(CLASSES[jsn["annotator_labels"][0]])

    return inputs, labels


def spacy_preprocess(text):
    # nlp = spacy.load("en")
    doc = nlp(text.lower())
    return doc


def preprocess_data(inputs, remove_punct, remove_stop, lemmatize):
    processed_inputs = []
    with PoolExecutor() as executor:
        for doc in executor.map(spacy_preprocess, inputs):
            # doc = nlp(sentences)
            temp_list = []
            for tok in doc:
                if remove_punct:
                    if tok.is_punct:
                        continue
                if remove_stop:
                    if tok.is_stop:
                        continue
                if lemmatize:
                    temp_list.append(tok.lemma_)
                else:
                    temp_list.append(tok.text)
            processed_inputs.append(" ".join(temp_list))
    return processed_inputs


def evaluate(X, y, count_model, tfidf_model, lr_model):
    X_tfidf = tfidf_model.transform(count_model.transform(X))
    print("\nEvaluating the model")
    y_pred = lr_model.predict(X_tfidf)
    accuracy = (y_pred == y).mean()
    return accuracy, y_pred


def plot_confusion_matrices(model_name, train_y_truth, train_y_pred, validate_y_truth, validate_y_pred):
    plot_folder = "plots/models/" + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    # Train CM
    skplt.metrics.plot_confusion_matrix(
        train_y_truth, train_y_pred, normalize=True)
    plt.savefig(plot_folder + 'train_cm.png')
    plt.title(model_name+"train_cm")
    plt.clf()
    plt.close()

    # Validation CM
    skplt.metrics.plot_confusion_matrix(
        validate_y_truth, validate_y_pred, normalize=True)
    plt.savefig(plot_folder + 'valid_cm.png')
    plt.title(model_name+"_valid_cm")
    plt.clf()
    plt.close()


def save_models(count_model, tfidf_model, lr_model):
    model_folder = "models/"+MODEL_NAME+"/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Count Model
    pickle.dump(count_model, open(model_folder+"count_model.pkl", "wb"))

    # TFIDF Model
    pickle.dump(tfidf_model, open(model_folder+"tfidf_model.pkl", "wb"))

    # LR Model
    pickle.dump(lr_model, open(model_folder+"lr_model.pkl", "wb"))


def train(X_train, y_train):
    print("\nTraining CountVectorizer")
    count_model = CountVectorizer(binary=False)
    count_model.fit(X_train)

    print("\nTraining TFIDF")
    tfidf_model = TfidfTransformer()
    tfidf_model.fit(count_model.transform(X_train))
    X_train_tfidf = tfidf_model.transform(count_model.transform(X_train))

    print("\nTraining Logistic Regression")
    lr_model = LogisticRegression(
        max_iter=10000, multi_class='multinomial', penalty='elasticnet', solver='saga', l1_ratio=0.5)
    lr_model.fit(X_train_tfidf, y_train)

    return count_model, tfidf_model, lr_model


###########
# Configs #
###########
MODEL_NAME = "LR_unprocessed_elasticnet"
print("Training the model", MODEL_NAME)

CLASSES = {"entailment": 0, "contradiction": 1, "neutral": 2}

################
# Loading Data #
################

nlp = spacy.load("en")

print("Loading Dataset")
X_train, y_train = load_data(".data/snli/snli_1.0/snli_1.0_train.jsonl")
X_valid, y_valid = load_data(".data/snli/snli_1.0/snli_1.0_dev.jsonl")
X_test, y_test = load_data(".data/snli/snli_1.0/snli_1.0_test.jsonl")
print("Loaded the dataset")

"""
# Preprocessing
pp_st = time.time()
remove_punct = True
remove_stop = True
lemmatize = True

X_train = preprocess_data(X_train, remove_punct, remove_stop, lemmatize)
X_valid = preprocess_data(X_valid, remove_punct, remove_stop, lemmatize)
X_test = preprocess_data(X_test, remove_punct, remove_stop, lemmatize)

print("Time for Preprocessing", time.time() - pp_st)

# Saving Processed Dataset
pickle.dump(X_train, open(".data/X_train_lower_processed.pkl", "wb"))
pickle.dump(X_valid, open(".data/X_valid_lower_processed.pkl", "wb"))
pickle.dump(X_test, open(".data/X_test_lower_processed.pkl", "wb"))
"""

# X_train = pickle.load(open(".data/X_train_lower_processed.pkl", "rb"))
# X_valid = pickle.load(open(".data/X_valid_lower_processed.pkl", "rb"))
# X_test = pickle.load(open(".data/X_test_lower_processed.pkl", "rb"))


######################
# Training the model #
######################

count_model, tfidf_model, lr_model = train(X_train, y_train)

########################
# Evaluating the model #
########################

train_accuracy, train_pred = evaluate(
    X_train, y_train, count_model, tfidf_model, lr_model)
valid_accuracy, valid_pred = evaluate(
    X_valid, y_valid, count_model, tfidf_model, lr_model)
test_accuracy, test_pred = evaluate(
    X_test, y_test, count_model, tfidf_model, lr_model)


train_accuracy = round(train_accuracy, 3)
valid_accuracy = round(valid_accuracy, 3)
test_accuracy = round(test_accuracy, 3)

print("Accuracies:", train_accuracy, valid_accuracy, test_accuracy)


###################################
# Plotting the Confusion matrices #
###################################
plot_confusion_matrices(MODEL_NAME, y_train, train_pred, y_valid, valid_pred)

####################
# Saving the model #
####################
save_models(count_model, tfidf_model, lr_model)

print("\nTime taken for full run", time.time() - start_time, "\n")
