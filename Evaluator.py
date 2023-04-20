from SemanticSimilarity import SemanticSimilarity
from LanguageModel import LanguageModel

from pythonrouge.pythonrouge import Pythonrouge

class Evaluator:
    def __init__(self, ss_model=None, lm_model=None, div_val=1000):        
        self.ss_model = SemanticSimilarity() if ss_model==None else ss_model
        self.lm_model = LanguageModel(div_val) if lm_model==None else lm_model

    def get_score(self, preds, inputs, refs, rouge_type):
        
        score_rouge = self.get_rouge_score(preds, refs, rouge_type)
        
        score_fluency = self.lm_model.get_lm_score(preds).mean().item()
        
        score_semsim = self.ss_model.get_ss_score(preds, inputs).mean().item()
        
        results = score_rouge + [score_fluency, score_semsim]
        results = ['{:.4}'.format(r) for r in results]

        return results

    def get_rouge_score(self, preds, refs, rouge_type):

        rouge_dict = self._get_rouge_perl(preds, refs, rouge_type)

        if rouge_type == 'f1':
            score_rouge = [rouge_dict['r1_f1_mid'], rouge_dict['r2_f1_mid'], rouge_dict['rL_f1_mid']]     
        elif rouge_type == 'recall':
            score_rouge = [rouge_dict['r1_recall_mid'], rouge_dict['r2_recall_mid'], rouge_dict['rL_recall_mid']]     
        else:
            print("Wrong rouge type")
            return
        
        results = ['{:.4}'.format(r) for r in score_rouge]

        return results

    def _get_rouge_perl(self, summaries, all_references, rouge_type):
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
    
