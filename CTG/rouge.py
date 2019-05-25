import rouge

class Scorer(object):

    def __init__(self, pad_word, eos_word, unk_word):
        self.pad = pad_word
        self.eos = eos_word
        self.unk = unk_word
        self.unk_replacer = unk_word[::-1]
        self.hypothesis = []
        self.references = []
        self.evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                     max_n=2,
                                     limit_length=True,
                                     length_limit=100,
                                     length_limit_type='words',
                                     apply_avg=True,
                                     alpha=0.5,  # Default F1_score
                                     stemming=True)

    def add(self, ref, pred):
        self.references.append(self._get_sent(ref, replace_unk=True))
        self.hypothesis.append(self._get_sent(pred))

    def _get_sent(self, sent, replace_unk=False):
        sent = sent.replace(' ' + self.pad, '')
        sent = sent.replace(' ' + self.eos, '')
        if replace_unk:
            sent = sent.replace(self.unk, self.unk_replacer)
        return sent

    def score(self):
        scores = self.evaluator.get_scores(self.hypothesis, self.references)
        return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']
