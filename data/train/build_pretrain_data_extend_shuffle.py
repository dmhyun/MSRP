#!/usr/bin/env python
import nltk
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

trunlen = 15 # to give a length bias

trndata = [i.strip() for i in open('train.article.txt')]

fdata = [i for i in tqdm(trndata) if len(i.split()) <= trunlen]

print('# of filtered data: {}'.format(len(fdata)))

row = 1.0 # to learn how to remove words (ratio_added_words)
rs = 0.1 # to learn how to reorder text (ratio_shuffle)
rd = 0.1 # to learn how to generate words (ratio_drop)

fn = 't5_tl{}_row{}_rs{}_rd{}.es.pretrn'.format(trunlen, row, rs, rd)

data = []
for fd in tqdm(fdata):

    for _ in range(3): # Perturbation 3 times for each document

        orgwords = np.array(fd.split())

        # # Remove last period (it will be used as output text)
        # fd = fd[:-2]

        clen = len(orgwords)

        # Drop words    
        didxs = random.sample(range(clen), int(clen*rd))

        dropped_words = np.delete(orgwords, didxs)

        dlen = len(dropped_words)    

        # Shuffle only some words
        sidx = random.sample(range(dlen), int(rs * dlen))
        eidx = random.sample(range(dlen), int(rs * dlen))

        for si, ei in list(zip(sidx, eidx)):
            dropped_words[si], dropped_words[ei] = dropped_words[ei], dropped_words[si]    
        shuffled_words = dropped_words.tolist()
        
        target_words = shuffled_words
        tlen = len(target_words)

        # Select noisy words from another document
        oidx = random.randint(0, len(fdata)-1)

        od = fdata[oidx]
        owords = od.split()

        random.shuffle(owords)

        num_cwords = int(len(owords)*row)

        if num_cwords > int(tlen * row): num_cwords = int(tlen * row)

        sample_owords = owords[:num_cwords]

        # Randomly insert noisy words into a target text
        for j in range(len(sample_owords)):
            target_words.insert(random.randint(0, len(target_words)), owords[j])

        # target_words = target_words + owords # make a model to learn the lead bias instead of random insertion

        data.append([' '.join(target_words), fd])        

random.shuffle(data)

pd.DataFrame(data).to_csv('{}'.format(fn), header=['Input', 'Output'], index=False)


