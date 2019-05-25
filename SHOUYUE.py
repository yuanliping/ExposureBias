# path = ''
#
#
# def len_stat(path, threshold=70):
#     maxlen = 0
#     with open(path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     total_sens = len(lines)
#     count = 0
#     for line in lines:
#         line = line.strip('\n')
#         len_i = len(line.split())
#         if len_i > threshold:
#             count += 1
#         if len_i > maxlen:
#             maxlen = len_i
#     print('max len: %d' % maxlen)
#     print('total sentences: %d, longer than %d: %d, rate: %f' %
#           (total_sens, threshold, count, count / total_sens))
#
#
# len_stat('datasets/iwslt14.word.de-en/train.en')
# len_stat('datasets/iwslt14.word.de-en/test.en')


'''
挑选句子
'''
from CTG import bleu

scorer = bleu.Scorer('<pad>', '</s>', '<unk>')


def test_bleu(gold, test):
    scorer.add(gold, test)
    s = scorer.score()
    scorer.reset()
    return s


def read_sen(path):
    gold_sen, test_sen = list(), list()
    with open(path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            gold_sen.append(lines[i].strip('\n'))
            test_sen.append(lines[i + 1].strip('\n'))

    return gold_sen, test_sen


our_gold_sen, our_test_sen = read_sen('generation/cuda5_test_34.txt')
base_gold_sen, base_test_sen = read_sen('generation/cuda2_test_35.txt')
for i in range(len(our_gold_sen)):
    if test_bleu(our_gold_sen[i], our_test_sen[i]) < 10 and test_bleu(base_gold_sen[i], base_test_sen[i]) > 70:

        print(our_gold_sen[i])
        print(our_test_sen[i])
        print(base_test_sen[i])
        print('==============')
