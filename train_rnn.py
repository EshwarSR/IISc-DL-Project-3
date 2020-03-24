import utils
import time
import torch
import matplotlib.pyplot as plt
import scikitplot as skplt
import RNN_models
import os
import torch.optim as optim
import sys

###########
# Configs #
###########
CURRENT_EXPERIMENT = sys.argv[1]  # "EXP4"
GPU = int(sys.argv[2])  # 4


num_epochs = 20
evaluate_every = 1
log_every_iterations = 1000
learning_rate = 0.0005
batch_size = 128

word_vectors = "glove.6B.100d"

if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
    device = torch.device('cuda:'+str(GPU))
else:
    device = torch.device('cpu')

print("\n\nCURRENT EXPERIMENT:", CURRENT_EXPERIMENT)
print("Running on the device", device)

###################
# Loading Dataset #
###################
print("Loading Dataset")
data_st = time.time()
train_data, validation_data, test_data, inputs, answers = utils.get_train_valid_test_data(
    word_vectors, batch_size, device)
print("Finished loading Dataset", time.time() - data_st)

#####################
# Model Definitions #
#####################

vocab_length = len(inputs.vocab)
num_classes = len(answers.vocab)

# Experiments
experiments = {
    "EXP1": {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    },
    "EXP2": {
        "adam_0005": 0.0005,
        "adam_001": 0.001,
        "adam_005": 0.005
    },
    "EXP3": {
        "RNN_1L_fixed_projected_single_2L": RNN_models.RNN_1L_fixed_projected_single_2L(vocab_length, num_classes),
        "RNN_1L_notfixed_projected_single_2L": RNN_models.RNN_1L_notfixed_projected_single_2L(vocab_length, num_classes),
        "RNN_1L_fixed_notprojected_single_2L": RNN_models.RNN_1L_fixed_notprojected_single_2L(vocab_length, num_classes),
        "RNN_1L_notfixed_notprojected_single_2L": RNN_models.RNN_1L_notfixed_notprojected_single_2L(vocab_length, num_classes)
    },
    "EXP4":
    {
        "RNN_1L_fixed_projected_bi_2L": RNN_models.RNN_1L_fixed_projected_bi_2L(vocab_length, num_classes),
        "RNN_1L_notfixed_projected_bi_2L": RNN_models.RNN_1L_notfixed_projected_bi_2L(vocab_length, num_classes),
        "RNN_1L_fixed_notprojected_bi_2L": RNN_models.RNN_1L_fixed_notprojected_bi_2L(vocab_length, num_classes),
        "RNN_1L_notfixed_notprojected_bi_2L": RNN_models.RNN_1L_notfixed_notprojected_bi_2L(vocab_length, num_classes)
    },
    "EXP5": {
        "RNN_2L_fixed_projected_single_2L": RNN_models.RNN_2L_fixed_projected_single_2L(vocab_length, num_classes),
        "RNN_2L_notfixed_projected_single_2L": RNN_models.RNN_2L_notfixed_projected_single_2L(vocab_length, num_classes),
        "RNN_2L_fixed_notprojected_single_2L": RNN_models.RNN_2L_fixed_notprojected_single_2L(vocab_length, num_classes),
        "RNN_2L_notfixed_notprojected_single_2L": RNN_models.RNN_2L_notfixed_notprojected_single_2L(vocab_length, num_classes)
    },
    "EXP6":
    {
        "RNN_2L_fixed_projected_bi_2L": RNN_models.RNN_2L_fixed_projected_bi_2L(vocab_length, num_classes),
        "RNN_2L_notfixed_projected_bi_2L": RNN_models.RNN_2L_notfixed_projected_bi_2L(vocab_length, num_classes),
        "RNN_2L_fixed_notprojected_bi_2L": RNN_models.RNN_2L_fixed_notprojected_bi_2L(vocab_length, num_classes),
        "RNN_2L_notfixed_notprojected_bi_2L": RNN_models.RNN_2L_notfixed_notprojected_bi_2L(vocab_length, num_classes)
    },
    "EXP7": {
        "RNN_3L_fixed_projected_single_2L": RNN_models.RNN_3L_fixed_projected_single_2L(vocab_length, num_classes),
        "RNN_3L_notfixed_projected_single_2L": RNN_models.RNN_3L_notfixed_projected_single_2L(vocab_length, num_classes),
        "RNN_3L_fixed_notprojected_single_2L": RNN_models.RNN_3L_fixed_notprojected_single_2L(vocab_length, num_classes),
        "RNN_3L_notfixed_notprojected_single_2L": RNN_models.RNN_3L_notfixed_notprojected_single_2L(vocab_length, num_classes)
    },
    "EXP8":
    {
        "RNN_3L_fixed_projected_bi_2L": RNN_models.RNN_3L_fixed_projected_bi_2L(vocab_length, num_classes),
        "RNN_3L_notfixed_projected_bi_2L": RNN_models.RNN_3L_notfixed_projected_bi_2L(vocab_length, num_classes),
        "RNN_3L_fixed_notprojected_bi_2L": RNN_models.RNN_3L_fixed_notprojected_bi_2L(vocab_length, num_classes),
        "RNN_3L_notfixed_notprojected_bi_2L": RNN_models.RNN_3L_notfixed_notprojected_bi_2L(vocab_length, num_classes)
    },
    "EXP9":
    {
        "RNN_3L_300_notfixed_notprojected_bi_2L": RNN_models.RNN_3L_300_notfixed_notprojected_bi_2L(vocab_length, num_classes),
        "RNN_3L_400_notfixed_notprojected_bi_2L": RNN_models.RNN_3L_400_notfixed_notprojected_bi_2L(vocab_length, num_classes),
        "RNN_3L_300_notfixed_notprojected_bi_3L": RNN_models.RNN_3L_300_notfixed_notprojected_bi_3L(vocab_length, num_classes),
        "RNN_3L_400_notfixed_notprojected_bi_3L": RNN_models.RNN_3L_400_notfixed_notprojected_bi_3L(vocab_length, num_classes)
    }
}

################################
# Training & Evaluating Models #
################################
metrics = {
    "train_loss": {},
    "validation_loss": {},
    "train_accuracies": {},
    "validation_accuracies": {}
}

# for model_name, model in models.items():
for item1, item2 in experiments[CURRENT_EXPERIMENT].items():
    if CURRENT_EXPERIMENT == "EXP1":
        model_name = item1
        optimizer_fn = item2
        model = RNN_models.SNLIClassifier(vocab_length,  num_classes)
        optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    elif CURRENT_EXPERIMENT == "EXP2":
        model_name = item1
        learning_rate = item2
        model = RNN_models.SNLIClassifier(vocab_length,  num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        model_name = item1
        model = item2
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Started training model:", model_name)
    start = time.time()

    # Moving the model to device
    model.embed.weight.data.copy_(inputs.vocab.vectors)
    model.to(device)

    model, train_losses, validation_losses, train_accuracies, validation_accuracies, epoch_ticks = utils.train_evaluate_model(
        model_name, model, optimizer, num_epochs, train_data, validation_data, evaluate_every, log_every_iterations)
    metrics["train_loss"][model_name] = train_losses
    metrics["validation_loss"][model_name] = validation_losses
    metrics["train_accuracies"][model_name] = train_accuracies
    metrics["validation_accuracies"][model_name] = validation_accuracies

    end = time.time()
    print("Starting Loss:", train_losses[0], validation_losses[0])
    print("Ending Loss:", train_losses[-1], validation_losses[-1])
    print("Starting Accuracy:", train_accuracies[0], validation_accuracies[0])
    print("Ending Accuracy:", train_accuracies[-1], validation_accuracies[-1])
    print("Time taken:", end-start)
    print("\n\n")


###############################
# Plotting the Metrics Graphs #
###############################

def plot_metrics(metrics_data, metrics_name):
    for model_name in experiments[CURRENT_EXPERIMENT].keys():
        plt.plot(epoch_ticks, metrics_data[model_name], label=model_name)
        plt.legend(loc='best')
        plt.title(metrics_name)

    plot_folder = "plots/" + CURRENT_EXPERIMENT + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(plot_folder + metrics_name + '.png')
    plt.clf()


plot_metrics(metrics["train_loss"], "train_loss")
plot_metrics(metrics["validation_loss"], "validation_loss")
plot_metrics(metrics["train_accuracies"], "train_accuracies")
plot_metrics(metrics["validation_accuracies"], "validation_accuracies")

print("Done")
