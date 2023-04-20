import torch
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM        
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

class LanguageModel:
    def __init__(self, div_val=1000, model_name='anonsubms/lm_giga'):        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)   
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

        self.div_val = div_val        
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_lm_score(self, texts, batch_size=32):                
        lm_score = self._get_lm_score(self.model, texts, batch_size)                
        return lm_score

    def get_ppl(self, texts, batch_size=32):
        ppl = self._get_ppl(self.model, texts, batch_size)
        return ppl

    def _get_ppl(self, model, texts, batch_size=32):
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        with torch.no_grad():
            batches = chunks(texts, batch_size)   

            ppls = []
            BN=int(np.ceil(len(texts)/batch_size))
            for batch in batches:
                encodings_dict = self.tokenizer.batch_encode_plus(batch, padding='longest')

                input_ids = torch.tensor(encodings_dict['input_ids']).cuda()
                attn_mask = torch.tensor(encodings_dict['attention_mask']).cuda()

                # Split the data into batches    
                word_dist = model(input_ids).logits[:,:-1,:]
                word_dist = word_dist.softmax(dim=-1)

                token_probs = torch.gather(word_dist, -1, input_ids[:,1:][...,None]).view(input_ids[:,1:].shape)

                attn_mask = attn_mask[:,1:]

                masked_prob = token_probs.log() * attn_mask

                num_valid = attn_mask.sum(1)
                num_valid[num_valid==0] = 1 # Handling zero division

                batch_ppl = (-masked_prob.sum(1) / num_valid).exp()

                ppls.append(batch_ppl)                           

            ppls = torch.cat(ppls)
                
            return ppls

        
    def _get_lm_score(self, model, texts, batch_size=32):

        ppls = self._get_ppl(model, texts, batch_size)

        lm_score = (-ppls/self.div_val).exp()
                
        return lm_score

