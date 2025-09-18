#!/usr/bin/env python
# coding: utf-8

from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import evaluate

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 


#  <font color="red">NOTE</font> This code is outdated and a new model shall be created. Culture_Coding.xlsx files have been changed to _Altogether_Dataset_RACoded.xlsx

from huggingface_hub import notebook_login
# If below code does not work, copy and paste this code in the terminal: huggingface-cli login 
# then paste this read token (you will have to construct it)


notebook_login()


# ## Inference

from transformers import pipeline, AutoTokenizer

# set up the pipeline
classifier = pipeline("text-classification", top_k=None, model="Chantland/Hraf_MultiLabel", use_auth_token="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", tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"))


# This is the code you should currently use. Just copy and paste the text in here and will will spit out a demo answer.

# sample inference ENTER TEXT IN HERE.
text = '''
“Drinking-tubes made of the leg-bones of swans (Fig. 109) are 190 also used chiefly as a measure of precaution against diseases ‘subject to shunning.’....”
'''

tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
# reveal
prediction = classifier(text, **tokenizer_kwargs)
prediction


prediction[0][0]['label']


# ### Load datset <font color="red">(do not run more than 1 cell)</font>

# #### Old Dataset (dataset this was trained on)

import pandas as pd
import numpy as np

# load the dataset (THIS IS WHERE YOU WOULD ENTER IN THE DATA YOU WANTED TO TEST!)
df = pd.read_excel("../../RA_Cleaning/Culture_Coding_old.xlsx", header=[0,1], index_col=0)
# subdivide into just passage and outcome
df_small = pd.DataFrame()
df_small[["ID","passage","EVENT", "CAUSE", "ACTION"]] = df[[('CULTURE', "Passage Number"), ('CULTURE', "Passage"), ('EVENT', "No_Info"), ('CAUSE', "No_Info"), ('ACTION', "No_Info")]]
# Flip the label of "no_info"
df_small[["EVENT", "CAUSE", "ACTION"]] = df_small[["EVENT", "CAUSE", "ACTION"]].replace({0:1, 1:0})

# Remove certain passages which should not be in training or inference (these are duplicates that had to be manually found by a human)
values_to_remove = [3252, 33681, 6758, 10104]
df_small = df_small[~df_small['ID'].isin(values_to_remove)]
df_small

# Create an NLP friendly dataset
Hraf = Dataset.from_dict(df_small.to_dict(orient= 'list'))
Hraf


# #### Version 2 Codings: (Current Dataset - MINUS - Old Dataset)

# 

# load OLD and new datasets (THIS IS WHERE YOU WOULD ENTER IN THE DATA YOU WANTED TO TEST!)
df_old = pd.read_excel("../../RA_Cleaning/Culture_Coding_old.xlsx", header=[0,1], index_col=0)
df_current = pd.read_excel("../../RA_Cleaning/Culture_Coding.xlsx", header=[0,1], index_col=0)

# Remove the runs not of the first from the datasets (this will be made superfluous later but nonetheless is an extra assuredness step)
# df_old = df_old.loc[df_old[("CODER","Run_Number")]==1] #if this had "Run_Number" column, you would uncomment and run this line
df_current = df_current.loc[df_current[("CODER","Run_Number")]==1]

# only get new rows that have NOT been trained/tested on before
df_new = pd.concat([df_current, df_old])
df_new = df_new[~df_new.duplicated(subset=("CULTURE","Passage Number"), keep=False)]


# subdivide into just passage and outcome
df_small = pd.DataFrame()
df_small[["ID","passage","EVENT", "CAUSE", "ACTION"]] = df_new[[('CULTURE', "Passage Number"), ('CULTURE', "Passage"), ('EVENT', "No_Info"), ('CAUSE', "No_Info"), ('ACTION', "No_Info")]]
# Flip the label of "no_info"
df_small[["EVENT", "CAUSE", "ACTION"]] = df_small[["EVENT", "CAUSE", "ACTION"]].replace({0:1, 1:0})

# Remove certain passages which should not be in training or inference (these are duplicates that had to be manually found by a human)
values_to_remove = [3252, 33681, 6758, 10104]
df_small = df_small[~df_small['ID'].isin(values_to_remove)]
df_small

# Create an NLP friendly dataset
Hraf = Dataset.from_dict(df_small.to_dict(orient= 'list'))
Hraf


Hraf_dict = Hraf.to_dict()


import json
Hraf_dict = Hraf.to_dict()
with open(f"test_dataset.json", "w") as outfile:
    json.dump(Hraf_dict, outfile)


# #### Test Dataset codings:

import json
f = open("Datasets/test_dataset.json")
data = json.load(f)
Hraf = Dataset.from_dict(data)
Hraf


# ### Predict The Dataset

# get label names
labels = [label for label in Hraf.features.keys() if label not in ['ID', 'passage']]
labels


# load a list of passages and predict them (will take about .25 seconds per passage for me so beware the wait)

HrafOutput = []
for text in Hraf:
    # get actual labels
    actual_labels = [text[label] for label in labels]
    prediction = classifier(text['passage'], **tokenizer_kwargs)

    # get predicted labels
    scores = {item['label']:item['score'] for item in prediction[0]} #turn prediction into a dictionary
    pred_labels = [1 if scores[label] >= 0.5 else 0 for label in labels]


    output_dict = dict()
    output_dict["pred_labels"] = pred_labels
    output_dict["actual_labels"] = actual_labels
    output_dict["passage"] = text['passage']
    output_dict["ID"] = text['ID']


    # score[0][("actual_label", 'passage')] = text['passage'], text['label']
    HrafOutput.append(output_dict)


HrafOutput


# ### Calculate "Correctness" Metrics

# actual_labels
place= 0
print(sum(np.array(actual_labels)[:,place]))
print(sum(np.array(pred_labels)[:,place]))


HrafOutput


# 

from sklearn.metrics import f1_score, accuracy_score

actual_labels = [x['actual_labels'] for x in HrafOutput]
pred_labels = [x['pred_labels'] for x in HrafOutput]
for index, label in enumerate(labels):
    f1 = round(f1_score(y_true=np.array(actual_labels)[:,index], y_pred=np.array(pred_labels)[:,index]),3)
    print(f"{label}: {(6 - len(label)) *' '}{f1}")

print("\n")

f1_micro = round(f1_score(y_true=actual_labels, y_pred=pred_labels, average='micro'),3)
f1_macro = round(f1_score(y_true=actual_labels, y_pred=pred_labels, average='macro'),3)
print(f'F1 score (micro) {f1_micro}\nF1 score (macro) {f1_macro}')


# #### Create Correctness Metric Myself (note, unfinished for multilabel)
# This was made before finding out we could just load preconstructed, but it is great for the confusion matrix! <br>
# To use, change the "label_index" to run one label at a time.
# 

# set up confusion matrix for calculating precision myself
confusionMatrix_dict = {"TruePos":0, "FalsePos":0, "FalseNeg":0, "TrueNeg":0}
label_index = 2
for text in HrafOutput:
    if text['actual_labels'][label_index] == 1:
        if text['pred_labels'][label_index] == 1:
            confusionMatrix_dict["TruePos"] += 1
        elif text['pred_labels'][label_index] == 0:
            confusionMatrix_dict["FalseNeg"] += 1
        else:
            raise Exception("ERROR pos")
    elif text['actual_labels'][label_index] == 0:
        if text['pred_labels'][label_index] == 1:
            confusionMatrix_dict["FalsePos"] += 1
        elif text['pred_labels'][label_index] == 0:
            confusionMatrix_dict["TrueNeg"] += 1
        else:
            raise Exception("ERROR neg")
    else:
        raise Exception("ERROR actual")



confusionMatrix_dict


TP = confusionMatrix_dict['TruePos']
TN = confusionMatrix_dict['TrueNeg']
FP = confusionMatrix_dict['FalsePos']
FN = confusionMatrix_dict['FalseNeg']


#### Delete
import numpy as np
TP = 810
TN = 170
FP = 330
FN = 90


def pred(TP, TN, FP, FN):
    # get scores 
    precision = TP/ (TP + FP)
    recall = TP/ (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    F1_test = TP/ (TP + .5*(FP + FN)) #done to double check work
    MCC_num = (TP * TN) - (FP - FN)
    MCC_denom = np.sqrt((TP + FN) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = MCC_num / MCC_denom #mathews correlation coefficient

    assert round(F1,5) == round(F1_test,5), ValueError
    print(f'Precision:   {round(precision,3)}\nRecall:      {round(recall,3)}\nF1:          {round(F1,3)}\nMCC:         {round(MCC,3)}')

print("Normal")
pred(TP, TN, FP, FN)

print("\n\nFalse Flipping")
pred(TP, TN, FN, FP)

print("\n\nFull Flipped")
pred(TN, TP, FN, FP)


# get scores 
precision = TP/ (TP + FP)
recall = TP/ (TP + FN)
F1 = (2 * precision * recall) / (precision + recall)
F1_test = TP/ (TP + .5*(FP + FN)) #done to double check work
MCC_num = (TP * TN) - (FP - FN)
MCC_denom = np.sqrt((TP + FN) * (TP + FN) * (TN + FP) * (TN + FN))
MCC = MCC_num / MCC_denom #mathews correlation coefficient

assert round(F1,5) == round(F1_test,5), ValueError
print(f'Precision:   {round(precision,3)}\nRecall:      {round(recall,3)}\nF1:          {round(F1,3)}\nMCC:         {round(MCC,3)}')


# ## Optional File save

HrafOutput


# optionally save the file to json
from transformers import AutoTokenizer
import copy

HrafOutput_dummy = copy.deepcopy(HrafOutput)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["passage"], truncation=True)

tokenized_Hraf = Hraf.map(preprocess_function, batched=True)

for index, passage in enumerate(HrafOutput_dummy):
    assert passage['passage'] == tokenized_Hraf[index]['passage']
    passage['pred_labels'] = {key:passage['pred_labels'][index] for index, key in enumerate(labels)}
    passage['actual_labels'] = {key:passage['actual_labels'][index] for index, key in enumerate(labels)}
    passage['input_ids'] = tokenized_Hraf[index]['input_ids']


import json
# Save to unformatted json (uncomment)
with open(f"Datasets/tokenized_inputs.json", "w") as outfile:
    json.dump(HrafOutput_dummy, outfile)


# # Save to Dataset (uncomment)
# HrafOutput_dummy_dataset = Dataset.from_list(HrafOutput_dummy)
# Dataset.to_json(HrafOutput_dummy_dataset, f"../Tokenized_Datasets/tokenized_Hraf_{training_label}_Dataset")


# 

# ## CHi Square

from scipy.stats import chi2_contingency

ct_EVENT_CAUSE = pd.crosstab(df[('EVENT','No_Info')], df[('CAUSE','No_Info')], rownames=['ACTION'], colnames=['CAUSE'])
ct_EVENT_CAUSE


def chi_square_calc(row, col):
    cross_tab = pd.crosstab(df[(row,'No_Info')], df[(col,'No_Info')], rownames=[row], colnames=[col])
    stat, p, dof, expected = chi2_contingency(cross_tab)
    results = f"{row} by {col}:\nchi: {round(stat,1)}\np:   {round(p,3)}\n\n"
    return results

group_list = [('EVENT', 'CAUSE'), ('EVENT', 'ACTION'), ('ACTION', 'CAUSE')]
for row, col in group_list:
    print(chi_square_calc(row, col))


def chi_sqr(obs):
    size_x = obs.shape
    chi_mat = np.zeros(size_x)
    for row in range(size_x[0]):
        for col in range(size_x[1]):
            exp = np.sum(x[row]) * np.sum(x[:,col]) / np.sum(x)
            chi_mat[row, col] = np.sum((obs[row, col] - exp)**2 / exp)
    return chi_mat

print(np.sum(chi_sqr(x)))


# ## Prediction
