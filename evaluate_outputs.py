import os
import sys
sys.path.insert(0,'..')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch import nn

from pythonrouge.pythonrouge import Pythonrouge
from LanguageModel import LanguageModel
from SemanticSimilarity import SemanticSimilarity
from sentence_transformers import SentenceTransformer

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

def get_ss_score(textsA, textsB):
	Aembs = model.encode(textsA)
	Bembs = model.encode(textsB)

	cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	sim = (cos(torch.Tensor(Aembs), torch.Tensor(Bembs)) + 1)/2 #* 2  -1

	return sim

def load_refs(dpath):         
	refers = []
	for fn in os.listdir(dpath):
		if 'ref' in fn: # a reference we need                    
			each_ref = [r.strip() for r in open(dpath+'/{}'.format(fn))]
			refers.append(each_ref)     
	return refers

mn = sys.argv[1] # Model name in 'outputs' directory
gpu = sys.argv[2] # GPU index to use

os.environ["CUDA_VISIBLE_DEVICES"]=gpu

# Load models
flumodel = LanguageModel() 
model = SentenceTransformer('all-MiniLM-L6-v2')


# Load data 
ginput = [i.strip() for i in open('data/eval/Giga/input.txt').readlines()]
gref = load_refs('data/eval/Giga/')

dinput = [i.strip() for i in open('data/eval/DUC2004/input.txt').readlines()]
drefs = load_refs('data/eval/DUC2004/')


lens = [8, 10, 0.5, 13]

# Header
print('[{:15}]\t{}\t{}\t{}\t{}\t{}\t{}'.format(mn, 'RF-1', 'RF-2', 'RF-L', 'Fidel', 'Fluen', 'Len'))

for ln in lens:

	# Find a output file for a given length
	fn = None
	for f in os.listdir(mn):
		if str(ln) in f:
			fn = f
			break
	
	if fn == None: 
		print('No file for length: '+str(ln))
		continue

	output = [row.strip() for row in open(mn+'/'+fn).readlines()]

	# Compute ROUGE and score
	if 'DUC' not in fn: # Gigaword dataset
		rouge_result = get_rouge_score(output, gref)
		fidelity = get_ss_score(output, ginput).cpu().numpy().mean()
	else:
		rouge_result = get_rouge_score([o[:75] for o in output], drefs, rouge_type='recall')
		fidelity = get_ss_score(output, dinput).cpu().numpy().mean()

	fluency = flumodel.get_lm_score(output).cpu().numpy().mean()

	# Compute AVG length
	avglen = sum([len(o.split()) for o in output])/len(output)

	print('[Target Len {:4}]\t{}\t{:.3}\t{:.3}\t{:.3}'.format(ln, '\t'.join(rouge_result), fidelity, fluency, avglen))
