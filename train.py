# coding=utf-8
import sys 

import wandb

import gc
import pdb
import warnings
import time
import copy
import math
import string
import argparse
import glob
import os
import pickle
import random
import re
import shutil
import pandas as pd
from functools import partial
from typing import Dict, List, Tuple

from itertools import chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from rouge import Rouge

from datasets import load_dataset

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn.utils.rnn import pad_sequence

from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration, AutoModelForSeq2SeqLM

from Evaluator import Evaluator

from transformers import logging as translogging

translogging.set_verbosity_warning()

from SemanticSimilarity import SemanticSimilarity
from LanguageModel import LanguageModel

class Tester(): # Evaluation code
    
    def _build_dataset(self, tl, dpath, args):
        arts = [i.strip() for i in open(dpath + '/input.txt'.format())]  

        if args.numvlddata == 0:
            print('No validation data')
        else:
            arts = arts[:args.numvlddata]

        # Prepend length prompts
        if tl >= 1: # integer target lengths such as 8, 10, 13
            prefix = '{}: '.format(tl)

            controlled_inputs = [prefix+i for i in arts]

            target_lens = [tl] * len(arts)
        else: # ratio of target lengths
            controlled_inputs = []
            target_lens = [] 
            for at in arts:
                cl = int(cal_len(at) * tl)
                prefix = '{}: '.format(cl)
                controlled_inputs.append(prefix + at)
                target_lens.append(cl)  

            # To enhance the generation efficiency
            sortidx = np.array(target_lens).argsort()
            sorted_controlled_inputs = np.array(controlled_inputs)[sortidx]
            controlled_inputs = sorted_controlled_inputs.tolist()
            target_lens = np.array(target_lens)[sortidx].tolist()
   
        inputs = self.tokenizer(controlled_inputs, return_tensors='pt', padding=True, 
                       add_special_tokens=True).input_ids.cuda()
        batched_inputs = torch.split(inputs, 64)        

        # Load human-written references            
        refers = []
        for fn in os.listdir(dpath):
            if 'ref' in fn:                 
                each_ref = [r.strip() for r in open(dpath+'/{}'.format(fn))]

                if args.numvlddata != 0:
                    each_ref = each_ref[:args.numvlddata]

                refers.append(each_ref)    

        if tl < 1: # For ratio case, find corresponding references and input articles       
            sorted_refers = np.array(refers[0])[sortidx] # Gigaword contains a single reference set
            refers = [sorted_refers.tolist()]

            sorted_arts = np.array(arts)[sortidx]
            arts = sorted_arts.tolist()       

        return [batched_inputs, arts, refers, target_lens]
    
    def __init__(self, args, tokenizer, ss_model, lm_model, target_len, testdata='Giga'):         
        dpath = 'data/vld/'
        
        self.tokenizer = tokenizer
        
        self.vld_datasets = []
        self.isrange = args.isrange
        self.scl = args.scl

        if args.islength == True:
            target_len = [8, 10, 13] # Fixed lengths
        else: # ratio case
            target_len = [0.5] # Ratio-based length

        for tl in target_len:
            vdataset = self._build_dataset(tl, dpath, args)               
            self.vld_datasets.append(vdataset)

        self.evaluator = Evaluator(ss_model, lm_model, args.scf)
        self.repeat_penalty = args.repeat_penalty
        self.no_repeat_ngram = args.no_repeat_ngram
        # self.maxlens = [args.max_len_s, args.max_len_m, args.max_len_l]
        
    def get_score(self, model, aid): # aid: ID to select a proper dataset
        def length_reward(decoded_sents, target_len):       
            decosent_length = np.array([cal_len(d) for d in decoded_sents])
            
            lendiff = decosent_length - target_len
            lendiff = np.abs(lendiff) 
            
            length_reward = (-torch.Tensor(lendiff/self.scl)).exp().cuda()
            
            return length_reward

        stopwords = ['in', 'at', 'to', 'on', 'the', "'s", 'of', 'a', 'for', 'with', 'is', 'into', 'by',
                    'his', 'her', 'when', 'and', 'but']

        dayofweek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                     'Thursday',  'Friday', 'Saturday']        
                
        
        target_data = self.vld_datasets            

        binputs, arts, refers, target_lens = target_data[aid]     
        
        batch_size = len(binputs[0])
        
        with torch.no_grad():            
            model.eval()
            
            # Generate summaries
            preds = []
            for idx, bi in enumerate(binputs):

                batch_target_lens = target_lens[batch_size * idx: batch_size * (idx + 1)]
                MAXTOKEN = int(max(batch_target_lens) * 3)                
                MINTOKEN = int(min(batch_target_lens) * 1.3)                   

                with torch.no_grad():

                    attmask = (bi != self.tokenizer.pad_token_id)
                    bo = model.generate(input_ids=bi, do_sample=False, min_length=MINTOKEN,
                                        max_length=MAXTOKEN,
                                        repetition_penalty=self.repeat_penalty,
                                        attention_mask=attmask, # few beams to reduce training time
                                        no_repeat_ngram_size=self.no_repeat_ngram, num_beams=3,
                                        num_return_sequences=3, early_stopping=False)
            
                    str_bo = self.tokenizer.batch_decode(bo, skip_special_tokens=True, 
                                                    clean_up_tokenization_spaces=False)      
                
                    str_bo = np.array(str_bo).reshape(-1, 3) # num_beams=3
                    
                    # Summaries that are closer to a target length is better
                    # Also, filter out summaries that contain inappropriate patterns
                    output = []
                    for sidx, sb in enumerate(str_bo):                        
                        best_s = None
                        best_d = 1000 # a large number indicating distance
                        for s in sb:
                            # Truncate incomplete sentences
                            for _ in range(5):       
                                if len(s.split()) > 1:
                                    
                                    if s.split()[-1] in stopwords: # remove stop words
                                        s = ' '.join(s.split()[:-1])                                          

                            for dw in dayofweek:
                                dw = dw.lower()
                                if dw in s: s = s.replace(dw, '')

                            cl = cal_len(s)                       
                            
                            dist = abs(cl - batch_target_lens[sidx])
                            
                            if best_d > dist: 
                                best_d = dist
                                best_s = s      
                            
                            if best_d == 0: break
                                                
                        output.append(best_s)
                    
                    preds += output


            if len(set(target_lens)) == 1 and target_lens[0] == 13: # truncate summaries only for DUC2004
                trunc_preds = [p[:75] for p in preds]   
                rouge_type = 'recall'
            else:
                trunc_preds = preds # no truncation
                rouge_type = 'f1'

            # Get ROUGE scores
            scores = self.evaluator.get_score(trunc_preds, arts, refers, rouge_type)               

            olens = np.array([len(i.split()) for i in preds]) 
            diff = np.abs(olens - target_lens)
            le = diff.mean().round(2)

            scores.insert(3, le) # Length error
            scores.append(olens.mean().round(3)) # Output length mean
            scores = [float(i) for i in scores]                                    
            
            fluscore = self.evaluator.lm_model.get_lm_score(trunc_preds).cpu().numpy()
            semscore = self.evaluator.ss_model.get_ss_score(trunc_preds, arts).cpu().numpy()

            length_score = length_reward(trunc_preds, target_lens).cpu().numpy()

            reward_product = fluscore * semscore * length_score
            
            model.train()        
            
        return scores, reward_product


def cal_len(text):
    # Gigaword and DUC datasets contain pre-splitted texts
    return len(text.split()) 

class TextDataset(Dataset):
    def __init__(self, tokenizer, args):
        
        print("⚙️  Creating features from Gigaword data from HuggingFace Dataset")

        trndata = load_dataset('gigaword')['train']['document']
        texts, lengths = [], []                
        for l in trndata:
            t = l.strip()
            texts.append(t)
            lengths.append(cal_len(t))
            if len(texts) == args.numdata:
                break
                    
        tids = tokenizer(texts, add_special_tokens=True).input_ids   
        
        self.examples = list(zip(tids, lengths))                

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item] 

def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(tokenizer, args)    
    residual = len(dataset) % args.batch_size_trn
    if residual != 0:
        dataset.examples = dataset.examples[:-residual]
        
    return dataset
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def reward_function(generations, inputs, reward_getter, tokenizer, target_length, tester, lbs, lbl, lbf, scl, isoverlap): 

    def length_reward(decoded_sents, target_len):       
        decosent_length = np.array([cal_len(d) for d in decoded_sents])
        
        lendiff = decosent_length - target_len
        lendiff = np.abs(lendiff) 
        
        length_reward = (-torch.Tensor(lendiff/scl)).exp().cuda()
        
        return length_reward

    reward_semantic = []
    reward_fluency = []
    
    all_emb = []
    
    st = time.time()
        
    reward_semantic = reward_getter[0].get_ss_score(generations, inputs)
    reward_fluency = reward_getter[1].get_lm_score(generations)  

    all_emb = reward_getter[0]._get_sent_emb(generations)

    # if isoverlap == True: # Ablation study for semantic similarlity
    
    #     word_overlap_ratio = [] # For ablation study
    #     for i, g in enumerate(generations):
    #         gset = set(g.split())
    #         iset = set(inputs[i].split()) 
    #         overlap = gset.intersection(iset)

    #         oratio = len(overlap) / len(gset)

    #         word_overlap_ratio.append(oratio)
            
    #     reward_semantic = torch.FloatTensor(word_overlap_ratio).cuda()
    # else:
    #     reward_semantic = reward_semantic.cuda()   

    reward_semantic = reward_semantic.cuda()   
    reward_fluency = reward_fluency.cuda()
    reward_length = length_reward(generations, target_length).cuda()    
    
    reward = reward_length * lbl  + reward_fluency * lbf + reward_semantic * lbs  
        
    return reward, (reward_length, reward_fluency, reward_semantic), all_emb     

def train(args, train_dataset, model, tokenizer) -> Tuple[int, float]:

    print('💫 Loading models for fluency and semantic similarity')
    ss_model = SemanticSimilarity(args.semsim_type)
    lm_model = LanguageModel()    

    rg = (ss_model, lm_model) # Models to compute rewards       
    
    tester = Tester(args, tokenizer, ss_model, lm_model, args.target_len) # Evaluation function   
    
    """ Train the model """
    args.train_batch_size = args.batch_size_trn * max(1, args.n_gpu)   
    
    # def get_cons_weights(target_len, texts, model, tokenizer):
    
    #     def _get_cons_weight(lenA, lenB, model, tokenizer):
    #         tidA = tokenizer.convert_tokens_to_ids(str(lenA))
    #         tidB = tokenizer.convert_tokens_to_ids(str(lenB))

    #         vA = model.shared.weight[tidA]
    #         vB = model.shared.weight[tidB]

    #         cos = (vA * vB).sum() / (vA.norm() * vB.norm())

    #         return 0.5 * (cos+1) # normalization into [0, 1]
        
    #     lensA = [target_len] * len(texts)
    #     lensB = [cal_len(t) for t in texts]
        
    #     lenpairs = list(zip(lensA, lensB))
                        
    #     weights = [_get_cons_weight(la, lb, model, tokenizer) for la, lb in lenpairs]
        
    #     return torch.FloatTensor(weights).cuda()
        
    
    def collate4rl(examples: List[torch.Tensor], args):
                
        long_text_wlen, len_texts = [], []          

        if args.isrange == False: # Fixed length case
            for tl in args.target_len: 
                
                for tid, tlen in examples: # Num data x target length      
                    
                    if tl >= 1: # integer target lengths
                        cl = tl
                    else: # ratio of target lengths
                        cl = max(int(tlen * tl), 1) # tlen: true length, tl: target length                    

                    prefix = '{}: '.format(cl) # cl: current length

                    ids_prefix = tokenizer(prefix, add_special_tokens=False).input_ids

                    input_text = ids_prefix + tid 

                    long_text_wlen.append(torch.LongTensor(input_text))
                    len_texts.append(cl)
        else: # random length case
            for _ in range(args.num_msl): 
                for tid, tlen in examples: # Num data x target length      
                    minlen, maxlen = min(args.target_len), max(args.target_len)   
                    each_range = [minlen, min(maxlen+1, tlen)] # maxlen+1: below randint sample from [min, max)
                    cl = np.random.randint(each_range[0], each_range[1]) # Sample a length regardless of input length!

                    prefix = '{}: '.format(cl) # cl: current length

                    ids_prefix = tokenizer(prefix, add_special_tokens=False).input_ids

                    input_text = ids_prefix + tid 

                    long_text_wlen.append(torch.LongTensor(input_text))
                    len_texts.append(cl)

        pad_long_texts = pad_sequence(long_text_wlen, batch_first=True, 
                                          padding_value=tokenizer.pad_token_id)      
        len_texts = np.array(len_texts)       

        return [pad_long_texts, len_texts]
            
        
    mycollate = partial(collate4rl, args=args)

    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
        collate_fn=mycollate, drop_last=True, num_workers=0) 

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]    

    allparams = list(model.named_parameters())    

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in allparams if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in allparams if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0
        },
    ]    
        
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )    
    
    hp = '_'.join([arg.replace('--', '').replace('=','').replace('_','') for arg in sys.argv[1:]]) 
    
    if args.ptype == 'pegasus':
        output_dir = args.output_dir + '/{}_lr{}_lbl{}_ms{}_mlb{}_ptrx'.format(args.ptype, args.learning_rate, args.lbl, args.lbl_maxstep, args.lbl_min) 
    else:
        output_dir = args.output_dir + '/' + hp + '/'
    
    print("")
    print("🔆 Training ")
    print("  Num examples = %d" % len(train_dataset))
    if args.isrange == False:
        print('  Target Length = {}'.format(args.target_len))         
    else:
        print('  Target Length Range = {} ~ {}'.format(min(args.target_len), max(args.target_len)))         
    print('  Output Dir = {}'.format(output_dir))             
    print("")
    
    if args.save_steps > 0:
        print("💾 Model will be saved per {} steps\n".format(args.save_steps))

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    
    early_stop_count = 0    
    best_eval_perf = 0
    best_eval_perf2, best_eval_perfL = 0, 0
    best_eval_lenerr = 100
    best_avg_len = [100]*4 # Four evaluation cases
    
    batch_reward = []    

    NUMSUBBATCH=1
    
    # maxlens = [args.max_len_s, args.max_len_m, args.max_len_l]
    
    allsteps = len(train_dataloader) * int(args.num_train_epochs) # total number of batches
    epoch_iterator = tqdm(train_dataloader, desc="Progress", total=allsteps)  
    set_seed(args)  # For reproducibility
    for trn_iter in range(args.num_train_epochs):        
        batch_loss = 0
        batch_reward = []
        inc_batch_loss = 0

        model.zero_grad(set_to_none=True)
            
        for step, batch in enumerate(train_dataloader):

            args.scl = max(args.scl * args.lendecay, 1)
            
            model.train()                        
                            
            alltext, alllen = batch

            all_samples = []
            all_greedys = []
            all_each_reward = []
            all_sample_rewards = []
            all_baseline_rewards = [] 
            all_logprobs = []
            all_confidences = []            

            for i in range(args.num_agent): 
                ag = model

                long_text = alltext.to(args.device) # Input long text to summarize
                target_length = alllen # e.g., [8, ..., 8, 10, ..., 10, 13, ..., 13]

                st = time.time()                
                
                # Maximum and minimum number of tokens
                curmaxlen = int(target_length.max() * 3) # preparing 3 tokens per word (from data analysis)
                curminlen = int(target_length.min() * 1.3)

                # Generate baseline summaries (greedy_sents)
                with torch.no_grad():
                    ag.eval() 
                    if args.lead == False:
                        greedy_sents = ag.generate(input_ids=long_text, min_length=curminlen,
                                                max_length=curmaxlen, do_sample=False,
                                                no_repeat_ngram_size=args.no_repeat_ngram,
                                                repetition_penalty=args.repeat_penalty)
                    ag.train()          


                st = time.time()                    

                # Generate target summaries (sample_sents)
                outputs = ag.generate(input_ids=long_text, min_length=curminlen,
                                      max_length=curmaxlen,
                                      output_scores=True, do_sample=True, 
                                      return_dict_in_generate=True, 
                                      no_repeat_ngram_size=args.no_repeat_ngram,
                                      repetition_penalty=args.repeat_penalty)  

                sample_sents, token_logit = outputs[0], outputs[1]   

                if args.ptype in ['pegasus']:
                    token_logit = outputs[2]                                  

                # Computing log loss
                sub_sample_sents = sample_sents[:,1:] # Remove first <s> (BOS) token
                token_prob = torch.stack(token_logit, dim=1).softmax(dim=-1) 

                # [:,1:,None]: Exclude the first start token '<s> or <pad>'
                sample_prob = token_prob.gather(dim=2, index=sub_sample_sents[:,:,None]).squeeze()

                # Padding mask                    
                notpad_mask = (sub_sample_sents != tokenizer.pad_token_id)                    
                
                log_probs = ((sample_prob+1e-24).log()*notpad_mask).sum(dim=1) / notpad_mask.sum(-1)

                
                # Transform token ID numbers into words
                tk_sample_sents = tokenizer.batch_decode(sample_sents, skip_special_tokens=True, 
                                                         clean_up_tokenization_spaces=False)
                
                long_text = long_text[:,2:] # Removing length info. from the input text
                tk_inputs = tokenizer.batch_decode(long_text, skip_special_tokens=True, 
                                                   clean_up_tokenization_spaces=False)

                if args.lead ==  True: # Use lead bias as baseline in reinforcement learning
                    tk_greedy_sents = [' '.join(tk_inputs[i].split()[:target_length[i]]) for i in range(len(target_length))]                    
                else:
                    # Select greedy sampling based on the lead bias approach to save computation
                    tk_greedy_sents = tokenizer.batch_decode(greedy_sents, skip_special_tokens=True, 
                                                         clean_up_tokenization_spaces=False)


                # Error handling (empty sentence generation)
                for gtid in range(len(tk_sample_sents)):
                    sent = tk_sample_sents[gtid]
                    if cal_len(sent) < 3: sent = 'xxx dummy xxx'
                    tk_sample_sents[gtid] = sent

                    sent = tk_greedy_sents[gtid]
                    if cal_len(sent) < 3: sent = 'xxx dummy xxx'
                    tk_greedy_sents[gtid] = sent                       

                st = time.time()                

                # Compute rewards
                with torch.no_grad():                    
                    lbl_weight = min((global_step+1)/args.lbl_maxstep+args.lbl_min, 1)
                    curlbl = args.lbl * lbl_weight

                    sample_reward, sample_all_rw, emb = reward_function(tk_sample_sents, 
                                                                        tk_inputs, rg,
                                                                        tokenizer, target_length,
                                                                        tester, args.lbs, curlbl, args.lbf, args.scl,
                                                                        args.overlap) 

                    baseline_reward, _, _ =  reward_function(tk_greedy_sents, tk_inputs, rg, 
                                                             tokenizer, 
                                                             target_length, tester, args.lbs, curlbl, args.lbf, args.scl,
                                                             args.overlap)                   

                confidence = sample_all_rw[1] * sample_all_rw[2] # the quality of summaries
                    
                all_samples = np.array(tk_sample_sents).reshape(-1, args.batch_size_trn) # len(target lengths) * batch_size
                all_greedys = np.array(tk_greedy_sents).reshape(-1, args.batch_size_trn)
                all_confidences = confidence.reshape(-1, args.batch_size_trn)           
                
                all_each_reward.append(torch.stack(sample_all_rw).mean(-1).cpu().numpy())
                                
                all_sample_rewards.append(sample_reward)
                all_baseline_rewards.append(baseline_reward)                
                
                all_logprobs.append(log_probs)                

                all_length = np.array(target_length).reshape(-1, args.batch_size_trn)
                
            st = time.time()        
            
            if args.num_agent != 1:                
                batch_reward.append(np.array(all_each_reward).mean(1).tolist()) 
            else:
                batch_reward.append(all_each_reward[0].tolist())
                

            # Multi-Summary Learning            
            final_loss = None
            sample_consistencies_4eachlen = []
            baseline_consistencies_4eachlen = []
            for aid in range(args.num_msl):
                sample_consistencies = []
                baseline_consistencies = []
                
                for oaid in range(args.num_msl):
                    if oaid == aid: continue # exclude self-reference case
                                        
                    sample_consis_reward = rg[0].get_ss_score(all_samples[aid], all_samples[oaid])
                    baseline_consis_reward = rg[0].get_ss_score(all_greedys[aid], all_samples[oaid])
                                        
                    target_conf = all_confidences[aid] 
                    other_conf = all_confidences[oaid]

                    tempdiff = other_conf - target_conf
                    tempdiff[tempdiff<0] = 0
                    conf_weight = tempdiff

                    tl = all_length[aid]
                    olens = np.array([cal_len(ii) for ii in all_samples[oaid]])
                    lendiff = np.abs(olens - tl)
                    len_weight = (-torch.Tensor(lendiff/args.scl)).exp().cuda() # length reward
                
                    weight = len_weight * conf_weight.pow(args.alpha)

                    sample_consistencies.append(weight * sample_consis_reward)
                    baseline_consistencies.append(weight * baseline_consis_reward)                
                                         
                consistency_reward_sample = torch.stack(sample_consistencies).mean(0)
                consistency_reward_baseline = torch.stack(baseline_consistencies).mean(0)
                
                sample_consistencies_4eachlen.append(consistency_reward_sample)
                baseline_consistencies_4eachlen.append(consistency_reward_baseline)
                
            sample_consistencies_4eachlen = torch.cat(sample_consistencies_4eachlen)
            baselin_consistencies_4eachlen = torch.cat(baseline_consistencies_4eachlen)            
            
            # Original rewards + the reward from multi-summary learning mechanism
            sample_reward = all_sample_rewards[0] + args.lb * sample_consistencies_4eachlen
            baseline_reward = all_baseline_rewards[0] + args.lb * baselin_consistencies_4eachlen


            # Compute a loss
            rl_loss = -(sample_reward - baseline_reward) * all_logprobs[0]   

            losses = rl_loss.reshape(-1, args.batch_size_trn)

            if global_step % args.logging_steps == 0 and step != 0:                     
                brewards = np.array(batch_reward).mean(0)                

               # You can print below values to console if you want
                wandb.log({
                    'train/epoch': global_step/len(train_dataloader),
                    "train/reward/length": brewards[0],
                    "train/reward/fluency": brewards[1],
                    "train/reward/semantic": brewards[2],
                    "train/reward/msl": sample_consistencies_4eachlen.mean().item(),
                    "train/loss": inc_batch_loss,
                    "train/scl": args.scl
                })                        

            st = time.time()
            agentloss = 0
            st = time.time()

            mean_loss = losses.flatten().mean()
            mean_loss.backward()
            agentloss += mean_loss.cpu().item()                
                

            tr_loss += agentloss
            batch_loss += agentloss                
                
            st = time.time()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
                optimizer.step()
                scheduler.step() 

                model.zero_grad(set_to_none=True)
                
                global_step += 1

                # Evaluation on validation data
                if args.logging_steps > 0 and global_step % args.save_steps == 0:                    
                    if args.evaluate_during_training:  
                                                
                        with torch.no_grad():                       
                            all_tar_scores = []
                            all_reward_product = []
                            NUM_EVAL_CASE= 3 if args.islength else 1 # 3: [8, 10, 13] lengths, 1: [50%] length
                            for eid in range(NUM_EVAL_CASE):  
                                scores, reward_product = tester.get_score(model, eid)             
                                all_tar_scores.append(scores)
                                all_reward_product.append(reward_product)
    
                        all_reward_product = np.stack(all_reward_product).mean()                    

                        tar_rouge1, tar_rouge2, tar_rougeL, lenerr, rflu, rsem, _ = np.array(all_tar_scores).mean(0) # aggregate each reward

                        avglen = np.array(all_tar_scores)[:,-1]
                                                    
                        head = ["Rouge-1","Rouge-2","Rouge-L",
                                "Len_err","Fluency","Semantic",
                                "Avg_len","Examples"]

                        eval_log = {}

                        eval_criterion = all_reward_product
                        
                        if eval_criterion > best_eval_perf:                
                            best_eval_perf = eval_criterion

                            best_eval_perf1 = tar_rouge1
                            best_eval_perf2 = tar_rouge2
                            best_eval_perfL = tar_rougeL
                            best_eval_lenerr = lenerr
                            early_stop_count = 0
                            best_avg_len = avglen

                            best_ARP = all_reward_product

                            os.makedirs(output_dir, exist_ok=True)

                            model_to_save = (model.module if hasattr(model, "module") else model)
                            model_to_save.save_pretrained(output_dir)                      

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))                                       
                        else:
                            if global_step > args.warmup_steps:
                                early_stop_count += 1    

                        # Weight and bias logging
                        eval_log['Eval/Rouge-1'] = best_eval_perf1                                                    
                        eval_log['Eval/Rouge-2'] = best_eval_perf2
                        eval_log['Eval/Rouge-L'] = best_eval_perfL
                        eval_log['Eval/LenErr'] = best_eval_lenerr
                        eval_log['Eval/Criterion_nottrunc'] = best_eval_perf  
                        eval_log['Eval/all_reward_product'] = best_ARP

                        if args.islength == True:
                            for tlidx, tl4log in enumerate([8,10,13]):
                                eval_log['Eval/Avg len {}'.format(tl4log)] = best_avg_len[tlidx]
                        else:
                            eval_log['Eval/Avg len {}'.format('50%')] = best_avg_len[0]

                        wandb.log(eval_log)
                            
                        if early_stop_count > args.patient:
                            print('\n❗️ Performance exceeds the patience count ({}).'
                                  .format(args.patient))
                            exit()

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
            curepoch = (len(train_dataloader) * trn_iter + step)/allsteps * args.num_train_epochs
            epochstr = '{:.2}/{}'.format(curepoch, int(args.num_train_epochs))

            inc_batch_loss = inc_batch_loss + (agentloss/args.num_agent - inc_batch_loss)/(step+1)
            epoch_iterator.update()
    
            epoch_iterator.set_postfix({'Epoch': epochstr, 
                                        'VLD_ROUGE-F1': round(best_eval_perf, 2)})   

            # Free loss and outputs
            del all_sample_rewards
            del all_baseline_rewards

            
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step


# ----- Start of main function ------ #

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


def num(s):
    try: return int(s)        
    except: return float(s)        


def main():    
    parser = argparse.ArgumentParser()        
    
    parser.add_argument("--model", default='msrp', type=str, required=False)
    parser.add_argument("--train_file", default='train.article.txt', type=str, required=False)    
    
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)    
    parser.add_argument("--lb", default=0.01, type=float, help='Weight for the quality reward R_C')   
    parser.add_argument("--lbs", default=1, type=float, help='Weight for semantic-similarity reward R_S')
    parser.add_argument("--lbf", default=1, type=float, help='Weight for fluency reward R_F')
    parser.add_argument("--lbl", default=1, type=float, help='Weight for length reward R_L')
    parser.add_argument("--lbl_maxstep", default=1, type=float, help='')
    parser.add_argument("--lbl_min", default=0.0, type=float, help='')
    parser.add_argument("--overlap", default=False, type=str2bool, help='Weight for length reward R_L')
    parser.add_argument("--lead", default=False, type=str2bool, help='Whether to use lead bias as a baseline')

    parser.add_argument("--scl", default=10, type=float, help='scale for length reward')
    parser.add_argument("--lendecay", default=1, type=float, help='scale for length reward')
    parser.add_argument("--scf", default=1000, type=float, help='scale for fluency reward')
    
    parser.add_argument('--num_agent', type=int, default=1, help="# of agents") # Deprecated
    parser.add_argument('--ptype', default='t5-pretrain/tl20_row1.0_rs0.1_rd0.1_nc1', type=str, help="type of pretrained model")
    parser.add_argument('--no_repeat_ngram', type=int, default=3, help="No repeatition for the n-gram")
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help="Weight as a penalty")    
            
    parser.add_argument("--target_len", default='8,10,13', help="Target lengths")
    parser.add_argument("--num_msl", default=3, type=int, help="# branches of multi-summary learning")
    
    # # Different max length for each target length, reducing training time (no impact on the quality)
    # parser.add_argument("--max_len_s", default=20, type=int, help="Max length for short generation")
    # parser.add_argument("--max_len_m", default=25, type=int, help="Max length for medium generation")
    # parser.add_argument("--max_len_l", default=30, type=int, help="Max length for long generation")
    
    parser.add_argument("--numdata", default=500000, type=int, help="# of training data to use")
    parser.add_argument("--numvlddata", default=500, type=int, help="# of validation data")
    
    parser.add_argument("--alpha", default=0.3, type=float, help="weights")                    

    parser.add_argument("--semsim_type", default='sent2vec', help="sent2vec or sbert") 
    
    parser.add_argument( "--output_dir", default='rebuttal_emnlp_trained', type=str, required=False)    
    
    parser.add_argument("--init_checkpoint",default=None,type=int,help="Model checkpoint for weights initialization.",)    
    parser.add_argument("--batch_size_trn", default=24, type=int, help="Batch size for trn")
    parser.add_argument("--batch_size_eval", default=24, type=int, help="Batch size for eval")

    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", default=True, action="store_true", help="Eval during Trn at each logging step")
    
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500)    
    
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--patient", default=10, type=int, help="# patient steps before early stop")
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=2, type=int)    
    parser.add_argument("--max_steps",default=-1,type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup steps.")    
    
    parser.add_argument("--overwrite_output_dir", default=True, action="store_true", help="Overwrite output dir")
    parser.add_argument("--overwrite_cache", default=True, action="store_true", help="Overwrite cached training and evaluation sets") 
    parser.add_argument("--seed", type=int, default=2022, help="random seed for initialization")    

    parser.add_argument('--wandb', default='disabled', type=str) # disabled or online
    
    args = parser.parse_args()

    wandb.init(project='MSRP', 
               config=args,
               mode=args.wandb)  

    torch.cuda.set_device(args.gpu) 

    args.train_file = './data/train/' + args.train_file

    init_path = args.ptype
    
    # Handle multiple target lengths
    if ',' in args.target_len:  # Fixed lengths
        args.target_len = [num(i) for i in args.target_len.split(',')]    
        args.num_msl = len(args.target_len)  
        args.isrange = False
    elif '~' in args.target_len: # Random lengths
        args.target_len = [num(i) for i in args.target_len.split('~')]        
        args.output_dir += '_range'
        args.isrange = True
    else: # Error case
        exit()
    

    args.islength = any([bool(i> 1) for i in args.target_len])

    assert args.batch_size_trn % args.num_agent == 0
    
    args.batch_size_trn = int(args.batch_size_trn / args.num_msl)
    
    nag = args.num_agent
    ckpt = args.init_checkpoint
    
    args.output_dir = args.output_dir + '_msrp/' 

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)                         
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 0
    
    set_seed(args)

    # Loading a pretrained language model to fine-tune
    if args.ptype == 'pegasus': # PEGASUS not trained on supervised data
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")    
        print("💫 Loading a pretrained model from {}\n".format(init_path))          

        agent = AutoModelForSeq2SeqLM.from_pretrained('google/pegasus-large').to(args.device)   
        agent = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large').to(args.device)   
    else: # T5 model
        tokenizer = AutoTokenizer.from_pretrained("t5-small")    
        
        if 't5-pretrain' in init_path: # using our pretrained model
            init_path = 'anonsubms/t5pretrain'
        else:
            init_path = 't5-small'
            
        print("💫 Loading a pretrained model from {}\n".format(init_path))             

        agent = AutoModelForSeq2SeqLM.from_pretrained(init_path)       
        agent = agent.to(args.device)

    print("📋 Training/evaluation parameters: %s\n" % args)

    # ****** Start training ****** 
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer)                
        
        with torch.autograd.set_detect_anomaly(True):
            global_step, tr_loss = train(args, train_dataset, agent, tokenizer)
        print(" global_step = %s, average loss = %s" % (global_step, tr_loss))


if __name__ == "__main__":
    main()

