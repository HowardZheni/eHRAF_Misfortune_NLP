#!/usr/bin/env python
# coding: utf-8

from datasets.dataset_dict import DatasetDict
from datasets import Dataset, concatenate_datasets
import evaluate
import os
import re
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 


# OPTIONAL IF YOU WANT TO PUSH TO THE HUB!
from huggingface_hub import notebook_login
# IF RUNNING THIS CELL DOES NOT WORK:
# copy and paste this code in the terminal: huggingface-cli login 
# then paste your token


notebook_login()


# ## Import the dataset

path = "../../../eHRAF_Scraper-Analysis-and-Prep/Data/"
dataFolder = r"(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_local_officials_OR_luck_and_chance_OR_magicians_and_diviners_OR_mortuary_specialists_OR_nuclear_family_OR_priesthood_OR_prophet/"
# dataFolder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'

#load df (only load one of these commented out lines)
# df = pd.read_excel(f"{path}{dataFolder}/_Altogether_Dataset_RACoded.xlsx", header=[0,1], index_col=0) # Fall 2023 sickness + non-sickness
df = pd.read_excel(f"{path}{dataFolder}/_Altogether_Dataset_RACoded_Combined.xlsx", header=[0,1], index_col=0) # Spring 2023 - Spring 2024  sickness + nonsickness dataset
df.head(3)


df["CODER"][["Run_Number", "Dataset"]].value_counts(sort=False, dropna=False)


# subdivide into just passage and outcome
df_small = pd.DataFrame()
df_small[["ID","passage","EVENT","CAUSE","ACTION"]] = df[[('CULTURE', "Passage Number"), ('CULTURE', "Passage"), ('EVENT', "No_Info"), ('CAUSE', "No_Info"), ('ACTION', "No_Info")]]
# Flip the lable of "no_info"
df_small[["EVENT","CAUSE","ACTION"]]  = df_small[["EVENT","CAUSE","ACTION"]].replace({0:1, 1:0})

# Remove certain passages which should not be in training or inference (these are duplicates that had to be manually found by a human)
values_to_remove = [3252, 33681, 6758, 10104]
df_small = df_small[~df_small['ID'].isin(values_to_remove)]

df_small

# # create train and validation/test sets
# train_val, test = train_test_split(df_small, test_size=0.2, random_state=10)

# create train and validation/test sets
train_val, test = train_test_split(df_small, test_size=0.2, random_state=10)
# # do it again to get the test and validation sets (15% = 50% * 30%)
# test, validation = train_test_split(test_val, test_size=0.5, random_state=10)




# Create an NLP friendly dataset
Hraf = DatasetDict(
    {'train':Dataset.from_dict(train_val.to_dict(orient= 'list')),
     'test':Dataset.from_dict(test.to_dict(orient= 'list'))})
Hraf


# Make sure the training set is as biased as our groups (we want to train on as or less biased data as the groups they come from) <br>
# We are shooting for equivelent biases across test, train, and validation (if it exists at this step)
# 

# extract the total proportion
def totalProportion(df, col, present=1):
    value_counts = df[col].value_counts()
    percentage = round(value_counts[present]/len(df)*100,2)
    return percentage

# extracts percentages per datafaframe
def colProportion(Hraf, col):
    percentage_list = []
    for dataframe in Hraf.keys():
        percentage_list += [round(sum(Hraf[dataframe][col]) / (len(Hraf[dataframe]))*100,2)]
    return percentage_list



# print bias per label
dataframe_keys= Hraf.keys()
labels = [label for label in Hraf['train'].features.keys() if label not in ['ID', 'passage']]
header = "                                TOTAL"
for key in dataframe_keys:
    header += f"     {key}"
print(header)
print('_'*(len(header)+4))
for col in labels:
    totalPercentage =  totalProportion(df_small, col)
    percentage_list =  colProportion(Hraf, col)
    spacing = 10
    percentage_str = f"{totalPercentage}{' '* (spacing-len(str(totalPercentage)))}"
    for index, key in enumerate(dataframe_keys):
        percentage_str += f"{(len(key)-5)*' '}{percentage_list[index]}{' '* (spacing-len(str(percentage_list[index])))}"
    print(f"{col}:{' ' * (30- len(col))} {percentage_str}")


# ## Preprocess

# Create labels for training and preprocessing

labels = [label for label in Hraf['train'].features.keys() if label not in ['ID', 'passage']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
id2label


# load a DistilBERT tokenizer to preprocess the text field: <br>

# Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length:<br>
# Guidelines were followed from NielsRogge found <a href= "https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb"> here </a>

from transformers import AutoTokenizer
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["passage"]
  # encode them
  encoding = tokenizer(text, max_length=512, truncation=True) #max length for BERT is 512
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding


# To apply the preprocessing function over the entire dataset, use 🤗 Datasets map function. You can speed up map by setting batched=True to process multiple elements of the dataset at once:

# Tokenize data, remove all columns and give new ones
tokenized_Hraf = Hraf.map(preprocess_data, batched=True, remove_columns=Hraf['train'].column_names)
tokenized_Hraf


# sample decoding
example = tokenized_Hraf['train'][1]
print(example.keys())
print(tokenizer.decode(example['input_ids']))


print(example['labels'])
[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]


# Number of passages longer than 512 tokens (and therefore truncated)
sequence_i = []
for i, tx in enumerate(tokenized_Hraf['train']):
    if len(tx['input_ids']) >= 512:
        sequence_i.append(i)
print('Number Truncated: ', len(sequence_i))
print(f'Percentage Truncated: {round(len(sequence_i)/len(tokenized_Hraf["train"])*100,1)}%')
print(sequence_i)


# ### Create Splits

#  Stratification using multilabels is a difficult process as the number of unique bins of stratification increases exponentially by the number of labels (see more info and potential ways to conduct multilabel sttratification sampling <a href="https://dl.acm.org/doi/10.5555/2034161.2034172"> HERE  </a>). We will currently disregard focusing on stratification of all the labels/classifications and just use a single label for stratification. Currently, this is still giving decent splits that do not deviate far from the true proportion or between n_splits. Still, one should check the proportional deviation of each label to make sure

#  Splitting
from sklearn.model_selection import StratifiedKFold
# folds = StratifiedKFold(n_splits=5)
folds = StratifiedKFold(n_splits=5, shuffle= True, random_state=10)
splits = folds.split(np.zeros(Hraf['train'].num_rows), Hraf['train']['ACTION'])


train_list = []
val_list = []

for fold, (train_idxs, val_idxs) in enumerate(splits, start=1):
    train_list += [train_idxs]
    val_list += [val_idxs]
    print("Fold:",fold)
    print(f"EVENT:  {np.mean(Hraf['train'][train_idxs]['EVENT'])}\nCAUSE:  {np.mean(Hraf['train'][train_idxs]['CAUSE'])}\nACTION: {np.mean(Hraf['train'][train_idxs]['ACTION'])}\n")

# print(train_list,"\n", val_list)
# print(train_idxs)


# Now create a batch of examples using <a href="https://huggingface.co/docs/transformers/v4.29.0/en/main_classes/data_collator#transformers.DataCollatorWithPadding"> DataCollatorWithPadding</a>. It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Set tokenized passages to PyTorch Tensor

tokenized_Hraf.set_format("torch")


# ## Evaluate

# Obtain F1 score for evaluation

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction, TrainerCallback
import torch

# Get Metric performance
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

# Compute evaluation
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


# # Retrieving best model
# class BestCheckpointCallback(TrainerCallback):
#     def __init__(self):
#         self.best_checkpoint = None

#     def on_save(self, args, state, control, **kwargs):
#         # Update the best_checkpoint variable when a new best checkpoint is saved
#         self.best_checkpoint = control.value
# # Initialize the callback
# best_checkpoint_callback = BestCheckpointCallback()


# 
# ## Train
# Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    problem_type='multi_label_classification',
    num_labels = len(labels), 
    id2label=id2label, 
    label2id=label2id
)
# Get initial state (this is for later kfolds loops which appear to have data leakage)
initial_model_state = {name: param.data.clone() for name, param in model.named_parameters()}


#forward pass (NOT IMPLEMENTED YET, JUST A TEST)
outputs = model(input_ids=tokenized_Hraf['train']['input_ids'][0].unsqueeze(0), labels=tokenized_Hraf['train'][0]['labels'].unsqueeze(0))
outputs


# ### Initialize Training 

# ### Optional Paramters

# ### OPTIONAL If it crashes on a fold, you may skip the fold by specifying the start.
# ### Set to 1 after a successful run
# # start_fold = 1
# # # # Set it to true if you want the model to start from a checkpoint, 
# # # # You can also specify a specific checkpoint. Otherwise choose False, normally, this may be good to set False unless a crash occurs
# # resume_bool = '/checkpoint-434' 
# # #set True if you want to start at the beginnning
# # overwrite_training = False 


# # Create Eval_dataset (be aware this may raise a prompt you must fill in)
# # only create a new eval_df if one does not exist (this step is useful in case of a crash)
# if 'eval_df' not in locals(): 
#     eval_df = pd.DataFrame()
# else:
#     eval_inputAppend = input("eval_df found, would you like to append to there? (y/n)")
#     if eval_inputAppend.lower() =='y':
#         print("Appending to old eval dataframe, this is useful if you must restart training")
#         # fix issues with starting fold
#         if max(eval_df['fold']) >= start_fold: 
#             Start_fold_input = input(f"Your starting fold ({start_fold}) is not greater than the largest fold in the eval_df ({max(eval_df['fold'])}), this may mean redoing folds. Is this what you want (y/n)")
#             if Start_fold_input.lower() != 'y':
#                 raise Exception('Quitting run, please redo your start_fold parameter')
#     else:
#         eval_df = pd.DataFrame()


# ### Training Versions (run only 1)

# #### Normal training

training_args = TrainingArguments(
    output_dir="HRAF_Model_MultiLabel_ThreeLargeClasses_kfoldsDEMO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    push_to_hub=False,
)



# # Train the model
assert (start_fold >0) and (start_fold <= len(train_list)), f"Incorrect Starting fold, must be greater than or equal to 1 and less than or equal to {len(train_list)}"
for fold, (train_idxs, val_idxs) in enumerate(zip(train_list, val_list), start=1): # K-fold loop

    #Skip folds if desired
    if start_fold >fold:
        print('\033[93m'+ f"Skipping Fold {fold}"+ '\033[0m')
        continue

    print(f"------Fold {fold}--------\n")
    train_ds = tokenized_Hraf["train"].select(train_idxs)
    val_ds = tokenized_Hraf["train"].select(val_idxs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )
    try:
        trainer.train() 
    except:
        print('\033[93m'+ f"A crash occurred, restarting fold from checkpoint"+ '\033[0m')
        trainer.train(resume_from_checkpoint=True) #This is the same thing above but often restarting can make all the difference so let's try it

    # Evaluate and then concatinate results to a dataframe
    eval_dict = trainer.evaluate()
    eval_df_line = pd.DataFrame([eval_dict])
    eval_df_line["fold"] = fold
    eval_df_line["train_count"] = len(train_ds)
    eval_df_line["val_count"] = len(val_ds)
    eval_df_line["total_count"] = eval_df_line["val_count"] + eval_df_line["train_count"]
    eval_df = pd.concat([eval_df, eval_df_line])



# Save the model to disk
trainer.save_model()


# Push to hub (I have not gotten this to work so alternatively you can manually add in the best checkpoint by uploading the checkpoint into your hugging face account)
trainer.push_to_hub()


# #### Weight Decay Checking

# The following code attempts to investigate the best weight decay for the model by initiallizing training for each weight decay.

def eval_save(eval_df, dataset=Hraf, overwrite_training=True):
    # Augment Evaluation File 
    from datetime import date

    today = date.today()
    date_tm = today.strftime("%y/%m/%d")

    #reorganize columns
    cols = list(eval_df.columns.values) 
    remove_list = ["fold", "epoch","weight_decay", "learning_rate"]
    for removal in remove_list:
        cols.remove(removal)
    cols = remove_list+cols
    eval_df = eval_df[cols]


    trainingStatus = 'Initial Training' if overwrite_training == True else 'Continue Training'

    info_df  = pd.DataFrame({"Date":len(eval_df)*[date_tm],"Train_status":len(eval_df)*[trainingStatus]})
    eval_df = eval_df.reset_index(drop=True)
    eval_df = pd.concat([info_df, eval_df], axis=1)



    # import evaluation if it exists
    if os.path.exists("Evaluation.xlsx"):
        old_eval = pd.read_excel("Evaluation.xlsx", sheet_name="Sheet1", index_col=0)
        eval_df = pd.concat([old_eval, eval_df])

    eval_df.to_excel('Evaluation.xlsx')
# Run this if your model crashes and you want to save after the fact
# eval_save(dataset=Hraf, eval_df=eval_df)


### DELETE

# eval_df = pd.DataFrame()
# fold_f1s = []
# fold_f1 = eval_dict['eval_f1']
# fold_f1s.append(fold_f1)
# print(f"Fold {fold} Accuracy: {fold_f1}")

# eval_df_line = pd.DataFrame([eval_dict])
# eval_df_line["fold"] = fold
# eval_df_line["weight_decay"] = weight_decay
# eval_df_line["learning_rate"] = learning_rate
# eval_df_line["train_count"] = len(train_ds)
# eval_df_line["val_count"] = len(val_ds)
# eval_df_line["total_count"] = eval_df_line["val_count"] + eval_df_line["train_count"]
# eval_df = pd.concat([eval_df, eval_df_line])
# eval_df


# Note: If you are re-running the model, you must respecify the model. For reasons that I have yet to determine, Huggingface uses a cached model (I believe) every loop and thus causes a data leak that makes folds not indpendent of each other. This is a bad thing and can make the data overfitted and a less reliable estimate of model performance.

# torch.cuda.is_available()
# device = torch.device('mps')
# device


import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7


model_name = "HRAF_Model_MultiLabel_ThreeLargeClasses_kfoldsDEMO_WeightInvestigation"


# Define a range of weight decay values to test
weight_decay_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
# weight_decay_values = [1e-2]
learning_rate = 2e-5
# Track the average validation accuracy for each weight decay value
avg_validation_f1s = []



for weight_decay in weight_decay_values:
    fold_f1s = []


    eval_df = pd.DataFrame()
    # # Train the model
    # assert (start_fold >0) and (start_fold <= len(train_list)), f"Incorrect Starting fold, must be greater than or equal to 1 and less than or equal to {len(train_list)}"
    for fold, (train_idxs, val_idxs) in enumerate(zip(train_list, val_list), start=1): # K-fold loop


        output_dir = f"{model_name}/output_dir_{weight_decay}_fold_{fold}"

        resume_bool = False


        #Skip folds already completed
        if os.path.exists(f"{output_dir}"):
            if os.path.exists(f"{output_dir}/finished.txt"):
                print('\033[93m'+ f"Skipping {output_dir} as it is indicated as finished" + '\033[0m')
                continue
            else:
                print('\033[93m'+ f"Starting from last checkpoint {output_dir}"+ '\033[0m')
                resume_bool = True # resume from the last checkpoint if there is an output folder but it is not finished.





        print(f"------Fold {fold}/{len(train_list)}--------\n")

        #reinitialize the model (since it appears to be dataleaking over loops)
        model.load_state_dict(initial_model_state)
        for name, param in model.named_parameters():
                try:
                    assert (np.array(initial_model_state[name]) == np.array(param.data)).all(), "Parameters differ from original model"
                except:
                    print(name, "Differs from initial model")
        model.load_state_dict(initial_model_state)

        train_ds = tokenized_Hraf["train"].select(train_idxs)
        val_ds = tokenized_Hraf["train"].select(val_idxs)


        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=8,  # should be multiples of 8
            per_device_eval_batch_size=8, # should be multiples of 8
            num_train_epochs=5,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model='f1',
            push_to_hub=False,
            logging_dir=f"{model_name}/logs_{weight_decay}_fold_{fold}",
            logging_steps=100,
            use_cpu=True, # set True or False depending on if you want ot use the GPU, which is faster but has been unreliable on Macs
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[best_checkpoint_callback], 
            compute_metrics=compute_metrics,

        )
        try:
            trainer.train(resume_from_checkpoint = resume_bool) 
        except:
            print('\033[91m'+ f"A crash occurred, restarting fold from checkpoint"+ '\033[0m')
            trainer.train(resume_from_checkpoint=True) #This is the same thing above but often restarting can make all the difference so let's try it

        # Evaluate and then concatinate results to a dataframe

        # Evaluate on validation set for this fold
        eval_dict = trainer.evaluate(val_ds)
        fold_f1 = eval_dict['eval_f1']
        fold_f1s.append(fold_f1)
        print(f"Fold {fold} F1: {fold_f1}")

        eval_df_line = pd.DataFrame([eval_dict])
        eval_df_line["fold"] = fold
        eval_df_line["weight_decay"] = weight_decay
        eval_df_line["learning_rate"] = learning_rate
        eval_df_line["fold_f1"] = fold_f1
        eval_df_line["train_count"] = len(train_ds)
        eval_df_line["val_count"] = len(val_ds)
        eval_df_line["total_count"] = eval_df_line["val_count"] + eval_df_line["train_count"]
        eval_df = pd.concat([eval_df, eval_df_line])

        # # Get best model and then finish
        # best_checkpoint = best_checkpoint_callback.best_checkpoint
        # print("Best Checkpoint:", best_checkpoint)

        # Save Best model
        f = open(f"{output_dir}/finished.txt", "w")
        f.write(f"Best Model: TBD code incomplete")
        f.close()


    # Calculate average accuracy for this weight decay value
    if len(fold_f1s) == 0:
        print('\033[93m'+ f"No F1's in list, this likely means all the folds were skipped" + '\033[0m')
        continue
    elif len(fold_f1s) < len(train_list):
        print('\033[93m'+ f"Warning less F1's than expected, likely some folds were skipped and thus the mean f1 may be off" + '\033[0m')
    else:
        pass

    avg_f1 = np.mean(fold_f1s)
    avg_validation_f1s.append(avg_f1)
    print(f"Average Accuracy for Weight Decay {weight_decay}: {avg_f1}")
    #Save evaluation File
    eval_save(dataset=Hraf, eval_df=eval_df)




# Choose the weight decay with the highest average validation accuracy
best_weight_decay = weight_decay_values[np.argmax(avg_validation_f1s)]
print(f"Best Weight Decay: {best_weight_decay}")


# Save the model to disk
# trainer.save_model()


# ### Save Evaluation Dataset columns

# Augment Evaluation File 

from datetime import date

today = date.today()
date_tm = today.strftime("%y/%m/%d")

#reorganize columns
cols = list(eval_df.columns.values) 
remove_list = ["fold", "epoch"]
for removal in remove_list:
    cols.remove(removal)
cols = ["fold","epoch"]+cols
eval_df = eval_df[cols]

numrows = sum(Hraf.num_rows.values())

trainingStatus = 'Initial Training' if overwrite_training == True else 'Continue Training'

info_df  = pd.DataFrame({"Date":len(eval_df)*[date_tm],"Train_status":len(eval_df)*[trainingStatus]})
eval_df = eval_df.reset_index(drop=True)
eval_df = pd.concat([info_df, eval_df], axis=1)
eval_df




# import evaluation if it exists
if os.path.exists("Evaluation.xlsx"):
    old_eval = pd.read_excel("Evaluation.xlsx", index_col=0)
    eval_df = pd.concat([old_eval, eval_df])

eval_df.to_excel('Evaluation.xlsx')


# ### Save Partitioned Dataset 

def make_dir(path):
    import os
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path)

# make folder if it does not exist yet
path = os.getcwd() + '/Datasets'
make_dir(path)
# save to Json
for key in Hraf.keys():
    Hraf_dict = Hraf[key].to_dict()
    file_path = f"{path}/{key}_dataset.json"
    with open(file_path, "w") as outfile:
        json.dump(Hraf_dict, outfile)
        print(len(Hraf_dict['ID']), f"Rows for \'{key}\' succesfully saved to {file_path}")


# ## Continue Training
# 

# This code is meant for if you want to continue training from where you left off. It will for the most part have extremely similar code to the initial run code and even ask you to run some cells above which have needed functions. If you do not have an active dataset or model yet, this is not for you!

import json
import pandas as pd

path = "../../../eHRAF_Scraper-Analysis-and-Prep/Data/"
dataFolder = r"(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_local_officials_OR_luck_and_chance_OR_magicians_and_diviners_OR_mortuary_specialists_OR_nuclear_family_OR_priesthood_OR_prophet/"
# dataFolder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'

#load df (only load one of these commented out lines)
# df = pd.read_excel(f"{path}{dataFolder}/_Altogether_Dataset_RACoded.xlsx", header=[0,1], index_col=0) # Fall 2023 sickness + non-sickness
df = pd.read_excel(f"{path}{dataFolder}/_Altogether_Dataset_RACoded_Combined.xlsx", header=[0,1], index_col=0) # Spring 2023 - Spring 2024  sickness + nonsickness dataset
# df.head(3)


# Remove duplicates
# Ideally, we want no duplicates. If there is a duplicate, prefer run 3 over run 1.
#Take only the run number 1 and 3
df = df.loc[(df[("CODER","Run_Number")]==1) | (df[("CODER","Run_Number")]==3)]
dup1 = df[("CULTURE","Passage Number")].duplicated(keep=False) #passage number duplicates
print("Passage Number Duplicates before filtering:", sum(dup1))
dup2 = df[("CODER","Run_Number")] != 3 # select all that are not run 3 (as we want to use run 3)
df = df[~(dup1 & dup2)]
print("Passage Number Duplicates after filtering:", sum(df[("CULTURE","Passage Number")].duplicated(keep=False)))
print("Passage Duplicates after filtering:", sum(df[("CULTURE","Passage")].duplicated(keep=False)),"\n")

# subdivide into just passage and outcome
df_small = pd.DataFrame()
df_small[["ID","passage","EVENT","CAUSE","ACTION"]] = df[[('CULTURE', "Passage Number"), ('CULTURE', "Passage"), ('EVENT', "No_Info"), ('CAUSE', "No_Info"), ('ACTION', "No_Info")]]
# Flip the lable of "no_info"
df_small[["EVENT","CAUSE","ACTION"]]  = df_small[["EVENT","CAUSE","ACTION"]].replace({0:1, 1:0})

# Remove certain passages which should not be in training or inference (these are duplicates that had to be manually found by a human)
values_to_remove = [3252, 33681, 6758, 10104]
df_small = df_small[~df_small['ID'].isin(values_to_remove)]
df_small.head(3)



# ### Load and compare old dataset from JSON datasets

loc = ""

Hraf_prev = DatasetDict()

dataset_names = ['train', 'test']

df_prev = pd.DataFrame([])
for name in dataset_names:
    f = open(loc+f"Datasets/{name}_dataset.json")
    data = json.load(f)
    df_prev = pd.concat([df_prev, pd.DataFrame(data)])
    Hraf_prev[name] = Dataset.from_dict(data) # load to hugging face dataset dict
    # Closing file
    f.close()
# f = open(loc+"Datasets/train_dataset.json")
# # f = open("../HRAF_MultiLabel_ThreeLargeClasses/Datasets/test_dataset.json") #load old threemain class (comment this out unless you specifically are using it)
# data = json.load(f)
# Hraf_train = pd.DataFrame(data)
df_prev.head(3)


# #### Check difference between the two datasets

# Make sure all the rows in the original/previous dataset appear in the new one

assert len(set(df_small.columns) - set(df_small.columns)) == 0, "Dataframe columns do not match"
diff_count = len(df_small) - len(df_prev)
# df_small['ID'].isin(df_prev['ID'])

dif_df = df_prev[~df_prev['ID'].isin(df_small['ID'])] #get all ID's which are in the original dataset but not the new one
if len(dif_df) != 0:
    print('\033[93m'+ "WARNING, Not all rows of original dataset are within new dataset." + '\033[0m')
    print("IDs:\n",dif_df[['ID','passage']])
    print('\033[93m'+ "Including unknown extra rows to new dataset, stop here if this is not desired." + '\033[0m')
    df_small = pd.concat([df_small, dif_df])
    diff_count = len(df_small) - len(df_prev)


# extract only the new rows which do not appear in the original dataset
df_new = df_small[~df_small['ID'].isin(df_prev['ID'])]


# Divide them and turn them into HRAF
# create train and validation/test sets
train_val, test = train_test_split(df_new, test_size=0.2, random_state=10)
# # do it again to get the test and validation sets (15% = 50% * 30%)
# test, validation = train_test_split(test_val, test_size=0.5, random_state=10)




# Create an NLP friendly dataset
Hraf = DatasetDict(
    {'train':Dataset.from_dict(train_val.to_dict(orient= 'list')),
     'test':Dataset.from_dict(test.to_dict(orient= 'list'))})
Hraf


# ### Conduct filtering much like original training (copied and pasted or ran above cells when they are functions)

# Get labels
labels = [label for label in Hraf['train'].features.keys() if label not in ['ID', 'passage']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
id2label


from transformers import pipeline, AutoTokenizer

# CHANGE Model name
model = "MultiLabel_ThreeLargeClasses_kfoldsDEMO"

# set up the pipeline from local
import os
path =os.path.abspath(f"HRAF_Model_{model}")
# classifier = pipeline("text-classification", model=path, top_k=None)


# Get tokenizer from old model
model = "MultiLabel_ThreeLargeClasses_kfoldsDEMO"
# set up the pipeline from local
import os
path =os.path.abspath(f"HRAF_Model_{model}") #the need to specify checkpoints may not be needed now with setting the load best checkpoint at the end, regardless, consider specifying
# Note for above, the last accurate model was checkpoint-1176 so consider adding that into the path if you want to assure it uses it. Although I believe it should automatically load the best model!
tokenizer = AutoTokenizer.from_pretrained(path)


# Tokenize data, remove all columns and give new ones (GET FUNCTION FROM ABOVE)
tokenized_Hraf = Hraf.map(preprocess_data, batched=True, remove_columns=Hraf['train'].column_names)
tokenized_Hraf


# ### Create Splits

#  Splitting
from sklearn.model_selection import StratifiedKFold
# folds = StratifiedKFold(n_splits=5)
folds = StratifiedKFold(n_splits=5, shuffle= True, random_state=10)
splits = folds.split(np.zeros(Hraf['train'].num_rows), Hraf['train']['ACTION'])


train_list = []
val_list = []

for train_idxs, val_idxs in splits:
    train_list += [train_idxs]
    val_list += [val_idxs]
    print(f"EVENT:  {np.mean(Hraf['train'][train_idxs]['EVENT'])}\nCAUSE:  {np.mean(Hraf['train'][train_idxs]['CAUSE'])}\nACTION: {np.mean(Hraf['train'][train_idxs]['ACTION'])}\n")


# ### Collate and create torch

from transformers import DataCollatorWithPadding
# Pad data
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# set to torch
tokenized_Hraf.set_format("torch")


# ### Load model then Train

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    path, 
    problem_type='multi_label_classification',
    num_labels = len(labels), 
    id2label=id2label, 
    label2id=label2id
)


# Make sure you run the functions above like the "evaluate" function

### Optional Paramters

### OPTIONAL If it crashes on a fold, you may skip the fold by specifying the start.
### Set to 1 after a successful run
start_fold = 1
# # Set it to true if you want the model to start from a checkpoint, 
# # You can also specify a specific checkpoint. Otherwise choose False, normally, this may be good to set False unless a crash occurs
resume_bool = False
#set True if you want to start at the beginnning
overwrite_training = False 


# Create Eval_dataset (be aware this may raise a prompt you must fill in)
# only create a new eval_df if one does not exist (this step is useful in case of a crash)
if 'eval_df' not in locals(): 
    eval_df = pd.DataFrame()
else:
    eval_inputAppend = input("eval_df found, would you like to append to there? (y/n)")
    if eval_inputAppend.lower() =='y':
        print("Appending to old eval dataframe, this is useful if you must restart training")
        # fix issues with starting fold
        if max(eval_df['fold']) >= start_fold: 
            Start_fold_input = input(f"Your starting fold ({start_fold}) is not greater than the largest fold in the eval_df ({max(eval_df['fold'])}), this may mean redoing folds. Is this what you want (y/n)")
            if Start_fold_input.lower() != 'y':
                raise Exception('Quitting run, please redo your start_fold parameter')
    else:
        eval_df = pd.DataFrame()


training_args = TrainingArguments(
    output_dir="HRAF_Model_MultiLabel_ThreeLargeClasses_kfoldsDEMO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    push_to_hub=False,
)



# # Train the model
assert (start_fold >0) and (start_fold <= len(train_list)), f"Incorrect Starting fold, must be greater than or equal to 1 and less than or equal to {len(train_list)}"
for fold, (train_idxs, val_idxs) in enumerate(zip(train_list, val_list), start=1): # K-fold loop

    #Skip folds if desired
    if start_fold >fold:
        print('\033[93m'+ f"Skipping Fold {fold}"+ '\033[0m')
        continue

    print(f"------Fold {fold}--------\n")
    train_ds = tokenized_Hraf["train"].select(train_idxs)
    val_ds = tokenized_Hraf["train"].select(val_idxs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )
    try:
        trainer.train() 
    except:
        print('\033[93m'+ f"A crash occurred, restarting fold from checkpoint"+ '\033[0m')
        trainer.train(resume_from_checkpoint=True) #This is the same thing above but often restarting can make all the difference so let's try it

    # Evaluate and then concatinate results to a dataframe
    eval_dict = trainer.evaluate()
    eval_df_line = pd.DataFrame([eval_dict])
    eval_df_line["fold"] = fold
    eval_df_line["train_count"] = len(train_ds)
    eval_df_line["val_count"] = len(val_ds)
    eval_df_line["total_count"] = eval_df_line["val_count"] + eval_df_line["train_count"]
    eval_df = pd.concat([eval_df, eval_df_line])



# Save the model to disk
trainer.save_model()


# def train_loop(train_list=train_list, val_list=val_list, model=model, training_args=training_args, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics, start_fold=1):
#     assert (start_fold >0) and (start_fold <= len(train_list)), f"Incorrect Starting fold, must be no lower than 1 and no higher than {len(train_list)}"

# train_loop(start_fold=1)


# ### Save Evaluation

# eval_df = eval_df.drop(columns=['train_Count','val_Count',])


# Augment Evaluation File 

from datetime import date

today = date.today()
date_tm = today.strftime("%y/%m/%d")

#reorganize columns
cols = list(eval_df.columns.values) 
remove_list = ["fold", "epoch"]
for removal in remove_list:
    cols.remove(removal)
cols = ["fold","epoch"]+cols
eval_df = eval_df[cols]

numrows = sum(Hraf.num_rows.values())

trainingStatus = 'Initial Training' if overwrite_training == True else 'Continue Training'

info_df  = pd.DataFrame({"Date":len(eval_df)*[date_tm],"Train_status":len(eval_df)*[trainingStatus]})
eval_df = eval_df.reset_index(drop=True)
eval_df = pd.concat([info_df, eval_df], axis=1)
eval_df


# import evalutaion if it exists
if os.path.exists("Evaluation.xlsx"):
    old_eval = pd.read_excel("Evaluation.xlsx", index_col=0)
    eval_df = pd.concat([old_eval, eval_df])

eval_df.to_excel('Evaluation.xlsx')


# ### Save Paritioned Datasets

def make_dir(path):
    import os
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path)


# make folder if it does not exist yet
path = os.getcwd() + '/Datasets'
make_dir(path)
# save to Json
for key in Hraf.keys():
    Hraf_dict = Hraf[key]
    Hraf_dict = concatenate_datasets([Hraf_dict, Hraf_prev[key]])
    Hraf_dict = Hraf_dict.to_dict()
    file_path = f"{path}/{key}_dataset.json"
    with open(file_path, "w") as outfile:
        json.dump(Hraf_dict, outfile)
        print(len(Hraf_dict['ID']), f"Rows for \'{key}\' succesfully saved to {file_path}")


Hraf


Hraf_prev


Hraf_dummy = Hraf['train']
Hraf_dummy = concatenate_datasets([Hraf_dummy, Hraf_prev['train']])
Hraf_dummy


Hraf_dummy = Hraf
Hraf_dummy = concatenate_datasets([Hraf_dummy, Hraf_prev])
Hraf_dummy

