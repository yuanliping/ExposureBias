class Tokenizer():

    def __init__(self,):
        super().__init__()

    @staticmethod
    def tokenize(sent, dictionary):
        return list(map(lambda x: dictionary.index(x), sent))

