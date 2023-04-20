import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence

from sentence_transformers import SentenceTransformer

# This code and file are based on https://github.com/raphael-sch/HC_Sentence_Summarization

class SemanticSimilarity:
    def __init__(self, model_type):
        self.model_type = model_type
        if self.model_type == 'sent2vec':
            print('Sent2Vec loaded as a semantic-similarity evaluator')
            # Load word embedding and Tf-idf vector
            self.word2idx, _ = self._get_vocabs('sent2vec/title.vocab')
                
            wordvector = np.load('sent2vec/s2v_title.npy')
            
            self.embed_word = Embedding(wordvector.shape[0], wordvector.shape[1])
            self.embed_word.weight.data.copy_(torch.from_numpy(wordvector))
            self.embed_word.weight.requires_grad=False

            idx2tfidf = self._get_idf_vector('sent2vec/title.idf', self.word2idx)  
            
            self.embed_tfidf = Embedding(idx2tfidf.shape[0], 1)
            self.embed_tfidf.weight.data.copy_(torch.from_numpy(idx2tfidf[:,None]))
            self.embed_tfidf.weight.requires_grad=False   
            
            self.embed_word = self.embed_word.cuda()
            self.embed_tfidf = self.embed_tfidf.cuda()
            
            self.pad_val = self.word2idx['PAD']        
        elif self.model_type == 'sbert':
            print('Sentence BERT loaded as a semantic-similarity evaluator')
            self.sbert = SentenceTransformer('all-MiniLM-L6-v2').cuda()
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    def _get_sent_emb(self, texts):       
        if self.model_type == 'sbert':
            sent_emb = self.sbert.encode(texts)
            sent_emb = torch.FloatTensor(sent_emb).cuda() # output of sentBERT is numpy.array
        else: # Sent2Vec 
            # 1. word to index
            input_ids = []
            for text in texts:
                idxs = [self.word2idx[w] for w in text.split() if w in self.word2idx]
                # Handle no word-matching case
                if len(idxs) == 0: idxs.append(self.word2idx['dummy'])
                input_ids.append(torch.LongTensor(idxs))
            
            tensor_input = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_val).cuda()
            
            num_word = (tensor_input != self.pad_val).sum(1)  
            num_word[num_word==0] = 1 # Handle Divide by zero
            
            # 2. get sent embedding with tf-idf
            sent_emb = (self.embed_word(tensor_input) * self.embed_tfidf(tensor_input)).sum(dim=1) / num_word[:,None]    
            
            if sent_emb.isnan().sum() > 0:
                import pdb; pdb.set_trace()
            
            # L2 norm
            sent_emb = F.normalize(sent_emb, dim=-1, p=2)  
        
        return sent_emb

    def get_ss_score(self, textAs, textBs, batch_size=32): 

        if self.model_type == 'sbert': # SentenceBERT
            embAs = self.sbert.encode(textAs)
            embBs = self.sbert.encode(textBs)
            cos_sim = self.cos(torch.Tensor(embAs), torch.Tensor(embBs))
            ss_score = (cos_sim+1)/2
            ss_score = torch.FloatTensor(ss_score).cuda()
            return ss_score
            
        def chunks(lstA, lstB, n):            
            for i in range(0, len(lstA), n):
                yield lstA[i:i + n], lstB[i:i + n]
                
        assert len(textAs) == len(textBs)
        
        with torch.no_grad():        
            all_cos_scores = []
            for batch_textAs, batch_textBs in chunks(textAs, textBs, batch_size):
    
                embAs = self._get_sent_emb(batch_textAs)
                embBs = self._get_sent_emb(batch_textBs)

                cos_scores = (embAs * embBs).sum(-1) / (embAs.norm(dim=-1) * embBs.norm(dim=-1))

                all_cos_scores.append(cos_scores)       

            ss_score = (torch.cat(all_cos_scores) + 1)/2 # from [-1, 1] to [0, 1]            
            
            return ss_score
        
    def _get_vocabs(self, vocab_file):
        pad_token = 'PAD'
        unk_token = 'UNK'
        bos_token = 'BOS'
        eos_token = 'EOS'
        pad_idx = 0
        unk_idx = 1
        bos_idx = 2
        eos_idx = 3

        word2idx = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
        for line in open(vocab_file):
            word = line.strip()
            word2idx[word] = len(word2idx)
        idx2word = {v: k for k, v in word2idx.items()}
        assert word2idx[pad_token] == pad_idx
        assert word2idx[unk_token] == unk_idx
        assert word2idx[bos_token] == bos_idx
        assert word2idx[eos_token] == eos_idx
        return word2idx, idx2word

    def _get_idf_vector(self, idf_file, word2idx):
        words_not_found = list()
        vector = np.zeros(shape=(len(word2idx)), dtype=np.float32) # idf score for each word
        for line in open(idf_file):
            word, idf = line.split()
            idf = float(idf)
            if word in word2idx:
                idx = word2idx[word]
                vector[idx] = idf
            else:
                words_not_found.append(word)
                
        if len(words_not_found) > 0:
            print('{} words not in idf'.format(len(words_not_found)))
            
        return vector
