import sys

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


import nltk
import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
from transformers import MT5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)    
    
    decoded_preds = tokenizer.batch_decode(predictions[0].argmax(-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    tlens = np.array([len(dl.split()) for dl in decoded_labels])
    plens = np.array([len(dp.split()) for dp in decoded_preds])
    
    lenerr = np.abs(tlens - plens).mean()
    
    output = {}
    output['lenerr'] = lenerr
    
    return output    
   
def tokenize_function_inout(examples):        
    itexts = examples['Input']   
    otexts = examples['Output']   

    newtexts = [] # Prepended by output length
    for i in range(len(itexts)):
        it = itexts[i]
        ot = otexts[i]              
        
        wlen = len(ot.split())        
        prefix = '{}: '.format(wlen)         
        
        newtexts.append(prefix + it)                
    
    orgoutput = tokenizer(newtexts)    
    newoutput = tokenizer(otexts)
    
    orgoutput['labels'] = newoutput['input_ids']   
    
    return orgoutput


dpath = 'data/train/'

model_id = sys.argv[1] 

mylr = float(sys.argv[2])
myep = int(sys.argv[3])
mywd = float(sys.argv[4])

gpu = int(sys.argv[5])


os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu)

assert model_id in ['t5-small', 'google/t5-v1_1-small', 'facebook/bart-base', 'google/pegasus-large']

tl = 15
row = 1.0
rs = 0.0
rd = 0.0

numsteps = 500 if 'pegasus' not in model_id else 300

trnfile = 't5_tl{}_row{}_rs{}_rd{}.es.pretrn'.format(tl, row, rs, rd)

datasets = load_dataset('csv', data_files={'train': dpath+trnfile, 'vld':dpath+'pretrain.vld'})

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenized_datasets = datasets.map(tokenize_function_inout, batched=True, 
                                  num_proc=8, remove_columns=["Input", "Output"]) 

seq_datasets = tokenized_datasets

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

model_name = '{}-pretrained_lr{}_ep{}_wd{}_3times_apnd/'.format(model_id.split('/')[-1], mylr, myep, mywd)

batch_size = 64
args = Seq2SeqTrainingArguments(
    model_name,       
    learning_rate=mylr, # 1e-8 for bart,  1e-7 for t5
    num_train_epochs=myep,
    per_device_train_batch_size=batch_size,    
    per_device_eval_batch_size=batch_size,    
    weight_decay=mywd, # 1e-2 for bart, 1e-3 for t5
    save_total_limit=1,    
    load_best_model_at_end=True,
    group_by_length=True,
    dataloader_num_workers=1,    

    do_eval=True,  
    evaluation_strategy="steps",
    eval_steps=numsteps,  
    save_steps=numsteps,  
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=seq_datasets["train"],    
    eval_dataset=seq_datasets["vld"],   
    data_collator=data_collator,
    compute_metrics=compute_metrics,    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

