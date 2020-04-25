import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import torch
import json
import utils
from keras.preprocessing.sequence import pad_sequences
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_data(filepath):
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


def get_attention_masks(X):
    attention_masks = []
    for sample in X:
        att_mask = [int(token_id > 0) for token_id in sample]
        attention_masks.append(att_mask)

    return attention_masks

###########
# Configs #
###########


GPU = 0
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
CLASSES = {"entailment": 0, "contradiction": 1, "neutral": 2}


torch.cuda.set_device(GPU)
device = torch.device('cuda:'+str(GPU))
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)


# temp = utils.get_train_valid_test_data("glove.6B.100d", 128, device)

###########
# Dataset #
###########
st = time.time()
X_train, y_train, token_type_ids_train, max_len_train = load_data(
    ".data/snli/snli_1.0/snli_1.0_train.jsonl")
X_valid, y_valid, token_type_ids_valid, max_len_valid = load_data(
    ".data/snli/snli_1.0/snli_1.0_dev.jsonl")

print("Time for loading and tokenizing data:", time.time() - st)
print("Lengths:", max_len_train, max_len_valid)
print(X_train[0])

#################
# Preprocessing #
#################

st = time.time()

# Padding
print('\nPadding/truncating all sentences to length', MAX_LEN)

X_train = pad_sequences(X_train, maxlen=MAX_LEN, dtype="long",
                        value=0, truncating="post", padding="post")
X_valid = pad_sequences(X_valid, maxlen=MAX_LEN, dtype="long",
                        value=0, truncating="post", padding="post")

token_type_ids_train = pad_sequences(token_type_ids_train, maxlen=MAX_LEN, dtype="long",
                                     value=1, truncating="post", padding="post")
token_type_ids_valid = pad_sequences(token_type_ids_valid, maxlen=MAX_LEN, dtype="long",
                                     value=1, truncating="post", padding="post")

# Attention masks
att_masks_train = get_attention_masks(X_train)
att_masks_valid = get_attention_masks(X_valid)


print("Time for pre processing:", time.time() - st)
print(X_train[0])
print(att_masks_train[0])
print(token_type_ids_train[0])

# Data Loaders
X_train = torch.tensor(X_train)
X_valid = torch.tensor(X_valid)
y_train = torch.tensor(y_train)
y_valid = torch.tensor(y_valid)
att_masks_train = torch.tensor(att_masks_train)
att_masks_valid = torch.tensor(att_masks_valid)
token_type_ids_train = torch.tensor(token_type_ids_train)
token_type_ids_valid = torch.tensor(token_type_ids_valid)

train_data = TensorDataset(X_train, att_masks_train,
                           token_type_ids_train, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(
    X_valid, att_masks_valid, token_type_ids_valid, y_valid)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(
    validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

#######################
# Model and optimizer #
#######################

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Total number of training steps is number of batches * number of EPOCHS.
total_steps = len(train_dataloader) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


############
# Training #
############


loss_values = []

for epoch_i in range(0, EPOCHS):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_token_type_ids = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=b_token_type_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]

        total_loss += loss.item()

        loss.backward()

        # Prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch

        with torch.no_grad():

            outputs = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")


############
# Plotting #
############


# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

# Plot the learning curve.
plt.plot(loss_values, 'b-o')

# Label the plot.
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("bert.png")
