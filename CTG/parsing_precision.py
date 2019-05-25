import re

class Scorer():

    def __init__(self):
        self.refs = []
        self.preds = []

    def add(self, ref, pred, sent):
        ref = self._merge(ref, sent)
        pred = self._merge(pred, sent)
        self.refs.append(ref)
        self.preds.append(pred)