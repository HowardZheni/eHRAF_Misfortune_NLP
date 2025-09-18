#!/usr/bin/env python
# coding: utf-8

from datasets.dataset_dict import DatasetDict
from datasets import Dataset, concatenate_datasets
import evaluate
import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 


# OPTIONAL - FOR UPLOADING TO HUGGINGFACE
# from huggingface_hub import notebook_login
# # # copy and paste this code in the terminal: huggingface-cli login 
# # then paste this read token (you will have to construct it)


# notebook_login()


# ## Import the dataset

# Dropbox Path (uncomment if using dropbox)
# df_path = "../../../eHRAF_Scraper-Analysis-and-Prep/Data/"
# dataFolder = r"(subjects-(contracts_OR_disabilities_OR_disasters_OR_friendships_OR_gift_giving_OR_infant_feeding_OR_lineages_OR_local_officials_OR_luck_and_chance_OR_magicians_and_diviners_OR_mortuary_specialists_OR_nuclear_family_OR_priesthood_OR_prophet/"
# # dataFolder = r'subjects-(sickness)_FILTERS-culture_level_samples(PSF)'

# GitHub Path (uncomment if using file hierarchy from Github)
df_path = "../../Coding and Dataset/Coded Dataset"
dataFolder = r""



# Get model and centralized path if relevent
model_name = "HRAF_MultiLabel_SubClasses_Kfolds"
path = f"" #Path to centralized file locations (leave blank if centralized location is here)



# load df (only load one of these commented out lines)
# df = pd.read_excel(f"{df_path}{dataFolder}/_Altogether_Dataset_RACoded.xlsx", header=[0,1], index_col=0) # Fall 2023 sickness + non-sickness
df = pd.read_excel(f"{df_path}{dataFolder}/_Altogether_Dataset_RACoded_Combined.xlsx", header=[0,1], index_col=0) # Spring 2023 - Spring 2024  sickness + nonsickness dataset
df.head(3)


# ## Preprocess

# ### Remove Duplicates

# There were multiple iterations of Research Assistants labeling the data. <br>
# We will just use run number 1 and 3 with preference to 3 when there are duplicates (as it is the most recent and robust)
# 
# 

# Show run number and dataset
df["CODER"][["Run_Number", "Dataset"]].value_counts(sort=False, dropna=False)


# useRuns = [1,3] #Only include these runs (NOTE THIS IS COMMENTED OUT IN ORDER TO NOT RUIN THE SUBLABEL DATASET BUT EVENTUALLY YOU SHOULD USE THIS CODE)
# df = df.loc[df[("CODER","Run_Number")].isin(useRuns)]

mask_NotDuplicate = ~(df.duplicated(("CULTURE","Passage"), keep=False)) 
mask_Dataset2 = df[("CODER","Run_Number")]==3

print("Duplicate Passages Before:",sum((df.duplicated(("CULTURE","Passage")))))


df = df[(mask_NotDuplicate) |  (mask_Dataset2)]




# Remove certain passages which should not be in training or inference (these are duplicates that had to be manually found by a human)
values_to_remove = [3252, 33681, 6758, 10104]
df = df[~df[('CULTURE','Passage Number')].isin(values_to_remove)]

print("Duplicate Passages After:",sum((df.duplicated(("CULTURE","Passage")))))
df[("CODER","Run_Number")].value_counts()


df


# ### Set up Dataset
# 

#Construct col list
cols = list(df.columns)
id_index = cols.index(('CULTURE', "Passage Number"))
passage_index = cols.index(('CULTURE', "Passage"))
event_index =  cols.index(('EVENT', "No_Info"))
cause_index = cols.index(('CAUSE', "No_Info"))
action_index = cols.index(('ACTION', "No_Info"))
# get a list of all the multi-indexed column names we want to evaluate
# col_list = [cols[id_index]] + [cols[passage_index]] + cols[event_index:event_index+4] + cols[cause_index:cause_index+7] + cols[action_index:action_index+7] #to include all columns including No_info
col_list = [cols[id_index]] + [cols[passage_index]] + cols[event_index+1:event_index+4] + cols[cause_index+1:cause_index+7] + cols[action_index+1:action_index+7] # to include al columns BUT No_info


## Remove the following columns from the dataset. Based on the results of previous models ran and the bias of the categories, remove the following columns from the dataset
remv_cols = [("CAUSE","Just_Happens"),("CAUSE","Other"),("ACTION","Other")]
for remv in remv_cols:
    col_list.remove(remv)



# get column names to ascribe to the new data frame
colNames = ["ID","passage"]
for category, sub_cat in col_list:
    # skip passage and id which have already been added
    # print(category, sub_cat)
    if category == "CULTURE":
        continue
    if sub_cat == "No_Info":
        colNames += [category]#this to include main classes, we will hold off on that
        pass
    else:
        colNames += [f'{category}_{sub_cat}']

print("Columns excluded:\n", set(cols)-set(col_list),"\n")
# for col in col_list
print("Columns included:")
for col in colNames:
    print(col)
# colNames


# ### Create Huggingface Dataset and do splits

# subdivide into just passage and outcome
df_small = pd.DataFrame()
df_small[colNames] = df[col_list]
# Flip the lable of "no_info"
# df_small[["EVENT","CAUSE","ACTION"]]  = df_small[["EVENT","CAUSE","ACTION"]].replace({0:1, 1:0})


# create train and validation/test sets
train_val, test = train_test_split(df_small, test_size=0.2, random_state=10)


# Create an NLP friendly dataset
Hraf = DatasetDict(
    {'train':Dataset.from_dict(train_val.to_dict(orient= 'list')),
     'test':Dataset.from_dict(test.to_dict(orient= 'list'))})
Hraf


# # # Delete, simply for quick filing of cleaned RA code for other analysis

# #Construct col list
# cols = list(df.columns)
# id_index = cols.index(('CULTURE', "Passage Number"))
# culture_index = cols.index(('CULTURE', "Culture"))
# passage_index = cols.index(('CULTURE', "Passage"))
# event_index =  cols.index(('EVENT', "No_Info"))
# cause_index = cols.index(('CAUSE', "No_Info"))
# action_index = cols.index(('ACTION', "No_Info"))
# # get a list of all the multi-indexed column names we want to evaluate
# # col_list = [cols[id_index]] + [cols[passage_index]] + cols[event_index:event_index+4] + cols[cause_index:cause_index+7] + cols[action_index:action_index+7] #to include all columns including No_info
# col_list = [cols[id_index]] + [cols[culture_index]] +  [cols[passage_index]] + cols[event_index+1:event_index+4] + cols[cause_index+1:cause_index+7] + cols[action_index+1:action_index+7] # to include al columns BUT No_info


# # get column names to ascribe to the new data frame
# colNames = ["ID","passage","Culture"]
# for category, sub_cat in col_list:
#     # skip passage and id which have already been added
#     # print(category, sub_cat)
#     if category == "CULTURE":
#         continue
#     if sub_cat == "No_Info":
#         colNames += [category]#this to include main classes, we will hold off on that
#         pass
#     else:
#         colNames += [f'{category}_{sub_cat}']

# # subdivide into just passage and outcome
# df_small = pd.DataFrame()
# df_small[colNames] = df[col_list]

# df_path = "../../NLP Predictions Analysis"
# df_path = f"{df_path}/_CleanedRACode.xlsx"
# df_small.to_excel(df_path)


# #### Show class bias
# 

# "Raw" is the actual percentage of times a label was selected (label_N / TOTAL) <br>
# "Adj." is the proportional percentage of times within a category (EVENT, CAUSE, ACTION) that a label was selected (label_N / Category_TOTAL).<br>
# Note that since multiple labels even within a category can be selected, the total Adj. amount for a category will be >=100%

multiCol = list(df.columns)
valuecountCol = []
for col in multiCol:
    if col[0] in ["CULTURE", "OTHER", "CODER"] or col[1] in ["Description", "Local_terms", "Local_Terms"]:
        continue
    else:
        valuecountCol.append(col)

# set up dataframe for easy saving
df_biases = pd.DataFrame(columns=["Class","Raw_Bias","Adj__Bias"])

# Get proportions and show table. 'raw' is just number of present divided by total while 'adj' is within main category proportion present divided by total main class present
print("BIAS FOR ANSWERING \'PRESENT\'")
print("Passage Count: ", len(df))
print(f"{' '*39}Raw{' '*10}Adj.") 
print(f"{'_'*60}")
for col in valuecountCol:
    if col[1] == "No_Info":
        proportion = 1-np.mean(df[col])
        mainCat_proportion = proportion
        proportion = round(proportion,2)
        df_biases = pd.concat([df_biases, pd.DataFrame({"Class":[col[0]],"Raw_Bias":[proportion]})])
        print(f"\n{col[0]}:{(38-len(col[0]))*' '}{proportion}")
    else:
        proportion = np.mean(df[col])
        adj_proportion = round(proportion / mainCat_proportion,2) # get adjusted proportion within category
        proportion = round(proportion,2)
        df_biases = pd.concat([df_biases, pd.DataFrame({"Class":[col[1]],"Raw_Bias":[proportion], "Adj__Bias":[adj_proportion]})])
        print(f"\t{col[1]}:{(30-len(col[1]))*' '}{proportion}{' '*(12-len(str(proportion)))}{adj_proportion}")




# #### Make sure the train, validation, and test sets are as biased as our total input data (we want each to match more or less with the total) <br>

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


# ### Labeling
# 

# Create labels for training and preprocessing

labels = [label for label in Hraf['train'].features.keys() if label not in ['ID', 'passage']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
id2label


# load a tokenizer to preprocess the text field: <br>

# Create a preprocessing function to tokenize text and truncate sequences to be no longer than DistilBERT’s maximum input length:<br>
# Guidelines were followed from NielsRogge found <a href= "https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb"> here </a>

from transformers import AutoTokenizer
import numpy as np


# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

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


# Set tokenized passages to PyTorch Tensor
tokenized_Hraf.set_format("torch")
tokenized_Hraf


example = tokenized_Hraf['train'][1]
print(example.keys())
print(tokenizer.decode(example['input_ids']))
print(example['labels'])
[id2label[idx] for idx, label in enumerate(example['labels']) if label == 1.0]


# Show number of passages longer than 512 tokens (and therefore truncated)
sequence_i = []
for i, tx in enumerate(tokenized_Hraf['train']):
    if len(tx['input_ids']) == 512:
        sequence_i.append(i)
print('Number Truncated: ', len(sequence_i))
print(f'Percentage Truncated: {round(len(sequence_i)/len(tokenized_Hraf["train"])*100,1)}%')
print(sequence_i)


# Now create a batch of examples using <a href="https://huggingface.co/docs/transformers/v4.29.0/en/main_classes/data_collator#transformers.DataCollatorWithPadding"> DataCollatorWithPadding</a>. It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

# ### Create Splits

#  Stratification using multilabels is a difficult process as the number of unique bins of stratification increases exponentially by the number of labels (see more info and potential ways to conduct multilabel sttratification sampling <a href="https://dl.acm.org/doi/10.5555/2034161.2034172"> HERE  </a>). We will currently disregard focusing on stratification of all the labels/classifications and just use a single label for stratification. Currently, this is still giving decent splits that do not deviate far from the true proportion or between n_splits. Still, one should check the proportional deviation of each label to make sure

#  Splitting
from sklearn.model_selection import StratifiedKFold
fold_n =5

# folds = StratifiedKFold(n_splits=5)
folds = StratifiedKFold(n_splits=fold_n, shuffle= True, random_state=10)
cols = Hraf['train'].column_names
splits = folds.split(np.zeros(Hraf['train'].num_rows), Hraf['train'][cols[-1]])
# preconstruct dataframe to show
fold_str = ["Fold "+str(x) for x in range(1,fold_n+1)]
df_foldPerc = pd.DataFrame(data=np.zeros((fold_n,len(labels))),columns=labels, index=fold_str)

train_list = []
val_list = []

for fold, (train_idxs, val_idxs) in enumerate(splits, start=1):
    train_list += [train_idxs]
    val_list += [val_idxs]
    train_hub = Hraf['train'][train_idxs]
    # print(len(Hraf['train'][train_idxs]["EVENT_Illness"]))
    df_foldPerc.iloc[fold-1] = [np.round(np.mean(train_hub[col]),2) for col in cols[2:]]

df_foldPerc


# ### Save Paritioned Datasets

# # Save datasets for later inference (SKIP IF YOU DO NOT WANT TO OVERWRITE DATASET FILES)

# def make_dir(path):
#     import os
#     # Check whether the specified path exists or not
#     isExist = os.path.exists(path)
#     if not isExist:
#     # Create a new directory because it does not exist
#         os.makedirs(path)

# # make folder if it does not exist yet
# path_datasets = os.getcwd() + '/Datasets'
# make_dir(path_datasets)
# # save to Json
# for key in Hraf.keys():
#     Hraf_dict = Hraf[key].to_dict()
#     file_path = f"{path_datasets}/{key}_dataset.json"
#     with open(file_path, "w") as outfile:
#         json.dump(Hraf_dict, outfile)
#         print(len(Hraf_dict['ID']), f"Rows for \'{key}\' succesfully saved to {file_path}")


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



# ## Train
# 

# Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:<br>
# You may need to log into huggingface to load transformer models

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DistilBertForSequenceClassification, RobertaModel
# (you may ned to log into Hugging face at the top to load models)

# # DistilBert
# model = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased", 
#     problem_type='multi_label_classification',
#     num_labels = len(labels), 
#     id2label=id2label, 
#     label2id=label2id
# )

# Roberta 
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base', 
    problem_type='multi_label_classification',
    num_labels = len(labels), 
    id2label=id2label, 
    label2id=label2id
)




from transformers import DataCollatorWithPadding
# Set up data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Get initial state (this is for kfolds loops which appear to have data leakage)
initial_model_state = {name: param.data.clone() for name, param in model.named_parameters()}


#forward pass (NOT IMPLEMENTED YET, JUST A TEST)
outputs = model(input_ids=tokenized_Hraf['train']['input_ids'][0].unsqueeze(0), labels=tokenized_Hraf['train'][0]['labels'].unsqueeze(0))
outputs


# ### Training

# Save evaluation dataframe
def eval_save(eval_df, directory="", overwrite_training=True):
    # Augment Evaluation File 
    from datetime import date

    file_path = f"{directory}/Evaluation.xlsx"

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
    if os.path.exists(file_path):
        old_eval = pd.read_excel(file_path, sheet_name="Sheet1", index_col=0)
        eval_df = pd.concat([old_eval, eval_df])

    eval_df.to_excel(file_path)

# combine the history output the model gives into a more digestable list format within a dictionary for val and train
def epochDictCreator(history_list:list) -> dict:
    epochHistory = dict() 

    #get train and Val and put them in a dictionary
    for train_or_val in ["train","val"]:
        if train_or_val== 'val':
            epochHistory_list = [x for x in history_list if (x.get("eval_loss", 'a') !='a')] # get only validation epochs (will have eval_loss)
        elif train_or_val == 'train':
            epochHistory_list = [x for x in history_list if (x.get("loss", 'a') !='a')] # get only training epochs (will have loss and typically the epoch will be a non-integer)
        else:
            raise Exception("Must enter train or val")
        # Create dictionary of values for val or train
        epochHistory_dict = dict()
        count=0
        for epoch in epochHistory_list:
            if count == 0:
                epochHistory_dict = {key:[val] for key, val in epoch.items()}
            else:
                for key, val in epoch.items():
                    try:
                        epochHistory_dict[key].append(val)
                    except:
                        print("train or Val:",train_or_val,"Count:", count)
            count +=1
        epochHistory[train_or_val] = epochHistory_dict
    return epochHistory

# save epoch dicts to a file (append if already exists)
def epochDictSave(directory, History_list:list):
    import json
    import os
    directory = f"{directory}/HistoryLog.json"
    if os.path.exists(directory):
        with open(directory, 'r') as openfile:
            oldHistory = json.load(openfile)
            History_list = oldHistory+History_list

    with open(directory, "w") as outfile:
        json.dump(History_list, outfile, indent=4)
        # outfile.write(json_object)

# Create Parameter JSON file
def saveParam(output_dir, param_dict:dict):
    import json
    import os
    # Check if param file already exists, if so, warn the user.
    if os.path.exists(f"{output_dir}/Model_Params.json"):
        print('\033[91m'+ "WARNING model parameter file overwritten" + '\033[0m')
        with open(f"{output_dir}/Model_Params.json", "r") as openfile:
            oldParam = json.load(openfile)
            flag = 0
            for key, val in param_dict.items():
                try:
                    if oldParam[key] != val:
                        flag = 1
                        print(f"{key}: {oldParam[key]} != {val}")
                except:
                    print(f"{key} not found in old file")
            if flag == 0:
                print("All parameters match")
    # save file
    with open(f"{output_dir}/Model_Params.json", "w") as outfile:
        json.dump(param_dict, outfile)
# create descending folders, give list of folders from parent to lowest child
def createFolders(paths:list):
    current_path = ""
    for path in paths:
        if len(path) == 0 or path == '/': # Skip blank paths (here to work with both google colab and vscode local notebooks)
            continue
        current_path += f"{path}/"
        os.mkdir(current_path) if not os.path.exists(current_path) else None


from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # L2 Regularization
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=1000
)


# #### Run training

# Kfolds trainer
#CHANGE for inputs
weight_decay = .01
learning_rate= .00002
dropout_list = []
batch_size = 8 # should be multiples of 8
epochs = 10


model_folder = "Model_00_TEST_DELETE" #this is where the output folders will be created and where a centralized model is made.
maxFolds = 1 # Set to 1 if you do not want to use kfolds, otherwise, set to any other fold number, including ones which you might want to stop early.



# model_path = f"{path}/{model_folder}"
model_path = f"{model_folder}"
eval_df = pd.DataFrame()

# # Train the model
for fold, (train_idxs, val_idxs) in enumerate(zip(train_list, val_list), start=1): # K-fold loop


    # skip folds if desired
    if fold > maxFolds:
        continue
    output_folder = f"Hierarchy_test_fold_{fold}/"
    output_dir = f"{model_path}/{output_folder}"
    resume_bool = False

    #Skip folds already completed
    if os.path.exists(f"{output_dir}"):
        if os.path.exists(f"{output_dir}/finished.txt"):
            print('\033[93m'+ f"Skipping {output_dir} as it is indicated as finished" + '\033[0m')
            continue
        else:
            print('\033[93m'+ f"Starting from last checkpoint {output_dir}"+ '\033[0m')
            resume_bool = True # resume from the last checkpoint if there is an output folder but it is not finished.

    # Create Parameter JSON file
    param_dict = {
        "learning_rate":learning_rate,
        "weight_decay":weight_decay,
        "batch_size":batch_size,
        "epochs":epochs,
        "Fold":fold,
        "model_name": model_folder,
    }
    createFolders(paths=[path, model_folder, output_folder])
    saveParam(output_dir=output_dir, param_dict=param_dict)



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
        per_device_train_batch_size=batch_size,  # should be multiples of 8
        per_device_eval_batch_size=batch_size, # should be multiples of 8
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model='f1',
        push_to_hub=False,
        save_total_limit=3, #Save only three checkpoints
        load_best_model_at_end = True, # retain the best model regardless of if it is beyond the save limit
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        use_cpu=False, # set True or False depending on if you want ot use the GPU, which is faster but has been unreliable on Macs
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        # callbacks=[best_checkpoint_callback], 
        compute_metrics=compute_metrics,

    )
    try:
        train_result = trainer.train(resume_from_checkpoint = resume_bool) 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print('\033[91m'+ f"A crash occurred, restarting fold from checkpoint"+ '\033[0m')
        train_result = trainer.train(resume_from_checkpoint=True) #This is the same thing above but often restarting can make all the difference so let's try it
    finally:
        #save logging
        epochDictSave(directory=output_dir, History_list=trainer.state.log_history)




    # Evaluate and then concatinate results to a dataframe

    # Evaluate on validation set for this fold
    eval_dict = trainer.evaluate(val_ds)
    fold_f1 = eval_dict['eval_f1']
    # fold_f1s.append(fold_f1)
    print(f"Fold {fold} F1: {fold_f1}")

    eval_df_line = pd.DataFrame([eval_dict])
    eval_df_line["model_name"] = output_dir
    eval_df_line["fold"] = fold
    eval_df_line["weight_decay"] = weight_decay
    eval_df_line["learning_rate"] = learning_rate
    eval_df_line["fold_f1"] = fold_f1
    eval_df_line["train_count"] = len(train_ds)
    eval_df_line["val_count"] = len(val_ds)
    eval_df_line["total_count"] = eval_df_line["val_count"] + eval_df_line["train_count"]
    #Save evaluation File
    eval_save(eval_df=eval_df_line, directory=f"{model_folder}")

    # Have a centralized eval_df For manual investiagtion when loops are done
    eval_df = pd.concat([eval_df, eval_df_line])

    # # Get best model and then finish
    # best_checkpoint = best_checkpoint_callback.best_checkpoint
    # print("Best Checkpoint:", best_checkpoint)

    # Save Best model
    trainer.save_model()
    f = open(f"{output_dir}/finished.txt", "w")
    f.write(f"Best Model: {trainer.state.best_model_checkpoint}")
    f.close()

    print("Best Model Checkpoint", trainer.state.best_model_checkpoint)










#Evaluate on test set (uncomment if you want to evaluate on test set after kfolds training)
# trainer.evaluate()


# ## Explore model

# #### (OPTIONAL) Correct google collab output

# This code below is meant to extract a table from google colab eroneously not saved. Likely, if things go correctly, this code will never be needed again and can be deleted. DO NOT RUN THIS CODE IF THE ABOVE TRAINING MODEL WORKED NORMALLY AND YOU HAVE A EPOCH OUTPUT SAVED

import pandas as pd
#Load File
codiedTable_path = "HRAF_Model_MultiLabel_SubClasses/Copied_colab_table.xlsx"
df_colab = pd.read_excel(codiedTable_path)
df_colab.head(3)
#Rename column headers


#initialize epoch History
epochHistory = {"train":dict(), "val":dict()}
epochHistory['train'] = {'epoch':list(df_colab['Epoch']), 'loss':list(df_colab['Training Loss'])}
epochHistory['val'] = {'epoch':list(df_colab['Epoch']), 'eval_loss':list(df_colab['Validation Loss']), 'eval_f1':list(df_colab['F1']), 'eval_accuracy':list(df_colab['Accuracy'])}


# ### Graph Plots

import json
import matplotlib.pyplot as plt
import os
import re


# #### Single Model Investgation
# 

# ##### Get History

# Optional, Reinitialize pathways shown above (copied and pasted, you must CHANGE if they do not match)
model_name = "HRAF_MultiLabel_SubClasses_Kfolds"
path = f"" #Path to centralized file locations
model_folder = "Model_3_LearningRates" #this is where the output folders will be created and where a centralized model is made.
model_path = f"{model_folder}"
outputFolder= "Learning_Rate_2e-05_fold_1"


history_path = f"{model_path}/{outputFolder}"
#Load history from file
f = open(history_path+"/HistoryLog.json")
history_list = json.load(f)
f.close()
epochHistory= epochDictCreator(history_list)


# ##### Load from loaded trainer

# #initialize epochHistory from loaded trainer
# history_list = trainer.state.log_history
# epochHistory = epochDictCreator(history_list)


# ##### Graph

# Graph Training Loss
train_loss = epochHistory['train']['loss']
train_epoch = epochHistory['train']['epoch']
val_loss = epochHistory['val']['eval_loss']
val_epoch = epochHistory['val']['epoch']



# "ro" is for "red dot"
plt.plot(train_epoch, train_loss, 'ro', label='Training loss')
# b is for "solid blue line"
plt.plot(val_epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# train_loss = epochHistory['train']['loss']
# train_epoch = epochHistory['train']['epoch']
val_acc = epochHistory['val']['eval_accuracy']
val_epoch = epochHistory['val']['epoch']


# b is for "solid blue line"
plt.plot(val_epoch, val_acc, 'b', label='Validation accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# train_loss = epochHistory['train']['loss']
# train_epoch = epochHistory['train']['epoch']
val_f1 = epochHistory['val']['eval_f1']
val_epoch = epochHistory['val']['epoch']



# b is for "solid blue line"
plt.plot(val_epoch, val_f1, 'b', label='Validation F1')
plt.title('Validation F1')
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.legend()

plt.show()


epochHistory['val'].keys()
import re

val_f1 = epochHistory['val']['eval_f1']
val_epoch = epochHistory['val']['epoch']
plt.plot(val_epoch, val_f1, 'b', linewidth=3, label='Overall Validation F1')

for key, value in epochHistory['val'].items():
  if not key.endswith('metrics'):
    continue
  title = re.findall(r'eval_(.*)_',key)[0]
  #get values into a single dict
  concatenated_dict = {}
  for dictionary in epochHistory['val'][key]:
      for key, value in dictionary.items():
          if key not in concatenated_dict:
              concatenated_dict[key] = []
          concatenated_dict[key].append(value)
  valCategories_f1 = concatenated_dict['f1']
  plt.plot(val_epoch, valCategories_f1, label=f'{title} F1')


plt.title('Validation F1 by Categories')
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.legend()

plt.show()


# #### Multiple Model Investigation

# Optional, Reinitialize pathways shown above (copied and pasted, you must CHANGE if they do not match)
model_name = "HRAF_MultiLabel_SubClasses_Kfolds"
path = f"" #Path to centralized file locations
model_folder = "Model_4_WeightByLearning" #this is where the output folders will be created and where a centralized model is made.
model_path = f"{path}{model_folder}"



# Get list of models through common directory
files_dir = [
    f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))
]

### OPTIONAL CHANGE
# Redo the text for the file names to better fit graphs (feel free to comment out)
def lowNum(text):
    startText =  'Learning_Rate_'
    regx = fr'{startText}([a-zA-Z0-9\.\-]+)'
    num = re.findall(regx,text)[0]
    return float(num)
files_dir.sort(reverse=True, key=lowNum)



epochHistory_dict = {}
for file in files_dir:

    history_path = f"{model_path}/{file}"
    f = open(history_path+"/HistoryLog.json")
    history_list = json.load(f)
    f.close()
    epochHistory= epochDictCreator(history_list)
    epochHistory_dict[file] = epochHistory



# Show Validation error rate
for key, epochHistory in epochHistory_dict.items():
    val_loss = epochHistory['val']['eval_loss']
    val_epoch = epochHistory['val']['epoch']
    plt.plot(val_epoch, val_loss, label=key)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Show F1 score improvements
for key, epochHistory in epochHistory_dict.items():
    val_f1 = epochHistory['val']['eval_f1']
    val_epoch = epochHistory['val']['epoch']
    plt.plot(val_epoch, val_f1, label=key)
plt.title('Validation F1')
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.legend()
plt.show()




