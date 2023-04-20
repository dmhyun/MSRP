#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch import nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from LanguageModel import LanguageModel
from SemanticSimilarity import SemanticSimilarity
from pythonrouge.pythonrouge import Pythonrouge
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration


def get_rouge_score(preds, refs, rouge_type='f1'):

    rouge_dict = _get_rouge_perl(preds, refs, rouge_type)

    if rouge_type == 'f1':
        score_rouge = [rouge_dict['r1_f1_mid'], rouge_dict['r2_f1_mid'], rouge_dict['rL_f1_mid']]     
    elif rouge_type == 'recall':
        score_rouge = [rouge_dict['r1_recall_mid'], rouge_dict['r2_recall_mid'], rouge_dict['rL_recall_mid']]     
    else:
        print("Wrong rouge type")
        return
    
    results = ['{:.4}'.format(r) for r in score_rouge]

    return results

def _get_rouge_perl(summaries, all_references, rouge_type):
    summary = [[s] for s in summaries]
    reference = [[] for _ in range(len(summary))]
    for references in all_references:
        for i, r in enumerate(references):
            reference[i].append([r])
    assert len(summary) == len(all_references[0]) # Checking the number of data
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        stemming=True, stopwords=False,
                        word_level=False, length_limit=False, length=100,
                        use_cf=True, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    scores = rouge.calc_score()    
    r = dict()
    rt = 'R' if rouge_type == 'recall' else 'F'
    for score_type in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
        r_type = score_type[0].lower() + score_type[-1]
        r['{}_{}_mid'.format(r_type, rouge_type)] = scores['{}-{}'.format(score_type, rt)]
    return {k: v * 100 for k, v in r.items()}    

def get_ss_score(model, textsA, textsB):
	Aembs = model.encode(textsA)
	Bembs = model.encode(textsB)

	cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	sim = (cos(torch.Tensor(Aembs), torch.Tensor(Bembs)) + 1)/2

	return sim


def build_dataset(tokenizer, tl, dpath, testdata, batch_size=64):    
    arts = [i.strip() for i in open(dpath + '/{}/input.txt'.format(testdata))] 

    if tl >= 1: # Target lengths
        prefix = '{}: '.format(tl)

        controlled_inputs = [prefix+i for i in arts]

        target_lens = [tl] * len(arts)
    else: # Compression ratios
        controlled_inputs = []
        target_lens = [] 
        for at in arts:
            
            cl = max(int(len(at.split()) * tl), 1) # At least one token has to be generated
            prefix = '{}: '.format(cl)
            controlled_inputs.append(prefix + at)
            target_lens.append(cl)

        sortidx = np.array(target_lens).argsort()
        sorted_controlled_inputs = np.array(controlled_inputs)[sortidx]
        controlled_inputs = sorted_controlled_inputs.tolist()
        target_lens = np.array(target_lens)[sortidx].tolist()

    inputs = tokenizer(controlled_inputs, return_tensors='pt', padding=True, 
                   add_special_tokens=True).input_ids.cuda()
    batched_inputs = torch.split(inputs, batch_size)        

    # Load references            
    refers = []    
    for fn in os.listdir(dpath+'/{}'.format(testdata)):
        if 'ref' in fn: # reference we need                                
            each_ref = [r.strip() for r in open(dpath+'/{}/{}'.format(testdata, fn))]

            refers.append(each_ref)      

    if tl < 1:        
        sorted_refers = np.array(refers[0])[sortidx]
        refers = [sorted_refers.tolist()]

        sorted_arts = np.array(arts)[sortidx]
        arts = sorted_arts.tolist()

    return [batched_inputs, arts, refers, target_lens]


def generate_summary(model, target_data, tokenizer, lm_model, ss_model, art, num_beams, batch_size, max_length, min_length):         
    
        stopwords = ['in', 'at', 'to', 'on', 'the', "'s", 'of', 'a', 'for', 'with', 'is', 'into', 'by',
                    'his', 'her', 'when', 'and', 'but']
        
        dayofweek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                     'Thursday',  'Friday', 'Saturday']
    
        binputs, arts, refers, target_lens = target_data                

        batch_size = len(binputs[0])               
        
        with torch.no_grad():                        
            preds = []
            for bidx, bi in tqdm(enumerate(binputs), total=len(binputs)):
                with torch.no_grad():
                    
                    batch_target_lens = target_lens[batch_size * bidx: batch_size * (bidx + 1)]
                    
                    MAXLEN = int(max(batch_target_lens) * max_length)                    
                    MINLEN = int(min(batch_target_lens) * min_length)                           

                    attmask = (bi != tokenizer.pad_token_id)
                    bo = model.generate(input_ids=bi, do_sample=False, min_length=MINLEN,
                                        max_length=MAXLEN, attention_mask=attmask,
                                        no_repeat_ngram_size=3, num_beams=num_beams,
                                        num_return_sequences=num_beams,
                                        early_stopping=False)          
                    
            
                    str_bo = tokenizer.batch_decode(bo, skip_special_tokens=True, 
                                                    clean_up_tokenization_spaces=False)
                
                    # Select the most proper summary
                    str_bo = np.array(str_bo).reshape(-1, num_beams)

                    if num_beams == 1: # It does not need to compute rewards             
                        output = str_bo.reshape(str_bo.shape[0]).tolist()
                        preds += output
                        continue
                                                            
                    # score within a target length & two rewards & no stopwords & no day of week
                    output = []
                    for sidx, sb in enumerate(str_bo): # For each example                                                                  

                        texts = sb.tolist()
                        
                        a = art[bidx * batch_size + sidx]                                                
                        cl = target_lens[bidx * batch_size + sidx]                       

                        # Removing the patterns                        
                        for idx, s in enumerate(sb):   
                            for _ in range(5):
                                if len(s.split()) > 1:
                                    if s.split()[-1] in stopwords: s = ' '.join(s.split()[:-1])

                            for dw in dayofweek:
                                dw = dw.lower()
                                if dw in s: s = s.replace(dw, '') 

                            sb[idx] = s                                                                                          

                        # Content preservation & Fluency
                        s_score = ss_model.get_ss_score(texts, [a]*num_beams).cpu().numpy()
                        l_score = lm_model.get_lm_score(texts).cpu().numpy()                      

                        # Length 
                        tlens = np.array([len(t.split()) for t in texts])                        
                        lenerr= abs(tlens-cl)                                         
                        length_penalty = -(lenerr)  

                        final_score = s_score + l_score + length_penalty
                        
                        best_s = sb[final_score.argmax()]
                        
                        output.append(best_s)
                    preds += output            
        return preds

fn = sys.argv[1]
gpu = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"]=gpu

tokenizer = AutoTokenizer.from_pretrained('t5-small') 
model = T5ForConditionalGeneration.from_pretrained(fn).cuda()

if fn.endswith('_sb'):
    ssmt = 'sbert'
else:
    ssmt = 'sent2vec'

ss_model = SemanticSimilarity(model_type=ssmt)
lm_model = LanguageModel()    

ss_eval_model = SentenceTransformer('all-MiniLM-L6-v2')

if 'ratio' not in fn: 
    tls = [8,10,13]
else:
    tls = [0.5]

dpath = 'data/eval/'
max_length = 3
min_length = 1.5 if 'ratio' not in fn else 1.9
batch_size = 64
num_beams = 20

for tl in tls:    
    dtype = 'Giga' if tl != 13 else 'DUC2004'

    art =  [i.strip() for i in open(dpath+'{}/input.txt'.format(dtype))]

    dataset = build_dataset(tokenizer, tl, dpath, dtype, batch_size)       

    inputart = dataset[1]
    refs = dataset[2]
    target_lengths = np.array(dataset[-1])

    preds = generate_summary(model, dataset, tokenizer, lm_model, ss_model, inputart, num_beams, batch_size, max_length, min_length)            

    if dtype == 'Giga':
        summaries = preds 
    else: # Truncation for DUC2004 dataset
        summaries = preds 
        trunc_summaries = [p[:75] for p in preds]           

    if dtype == 'Giga': # Gigaword
        scores = get_rouge_score(summaries, refs) # Default ROUGE type is F1
    else: # DUC2004
        scores = get_rouge_score(trunc_summaries, refs, rouge_type='recall')

    cp_score = get_ss_score(ss_eval_model, inputart, summaries).cpu().numpy().mean()
    fl_score = lm_model.get_lm_score(summaries).cpu().numpy().mean()

    lenavg = np.array([len(s.split()) for s in preds]).mean()    

    print('{}\tTL={}'.format(dtype, tl))    
    print('R1\tR2\tRL\tFD\tFL\tAL')
    print('{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\n'.format(*(scores+[cp_score, fl_score, lenavg])))  