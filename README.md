## Project 3 - Textual Entailment of SNLI Data using TF-IDF based Logistic Regression and RNNs

**Aim:** 

Train two models for identifying the Textual Entailment of SNLI Data as entailment/contradiction/neutral. 
1. TF-IDF based Logistic regression model.
2. Deep Model based on RNNs.


**Dataset** 
Dataset used in this project is SNLI.

**Folder description:**
- `utils.py` contains helper code used in other files.

- `train_lr.py` trains, validates and saves the logistic regression models. It also generates the required plots. Used for running multiple experiments.

- `RNN_models.py` defines many RNN network architectures used in this project.

- `train_rnn.py` trains, validates and saves the CNN models. It also generates the required plots. Used for running multiple experiments.

- `main.py` can be used to run inferences on new data. It loads the best models and writes 2 files `tfidf.txt` (output from best logistic regression model) and `deep_model.txt` (output from the best RNN model).

- `requirements.txt` contains the list of python packages used.

- `models` folder contains all the trained models.

- `plots` folder contains plots of all experiments.

**NOTE:** Run all the python files only from the current folder.