def distinct_1(path):
    total_unigram = 0
    distinct_unigram = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for word in line.strip('\n').split():
                total_unigram += 1
                distinct_unigram.add(word)
    return len(distinct_unigram), len(distinct_unigram) / total_unigram


def distinct_2(path):
    total_bigram = 0
    distinct_bigram = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.strip('\n').split()
            for i in range(len(words) - 1):
                total_bigram += 1
                distinct_bigram.add(''.join(words[i:i + 2]))
    return len(distinct_bigram), len(distinct_bigram) / total_bigram


def count_2(path, size=100):
    bigram_freq = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line.strip('\n').split()
            for i in range(len(words) - 1):
                bigram_i = ''.join(words[i:i + 2])
                if bigram_i in bigram_freq:
                    bigram_freq[bigram_i] += 1
                else:
                    bigram_freq[bigram_i] = 1
    bigram_freq = dict(sorted(bigram_freq.items(), key=lambda item: item[1], reverse=True))
    i = 0
    for (k, v) in bigram_freq.items():
        print('%s: %d' % (k, v))
        i += 1
        if i == size:
            break


import sys

test_path = 'generation/cuda7_gold_50.txt'  # sys.argv[1]
d1_num, d1_rate = distinct_1(test_path)
d2_num, d2_rate = distinct_2(test_path)
print('distinct-1: (%d, %f), distinct-2: (%d, %f)' % (d1_num, d1_rate, d2_num, d2_rate))
count_2(test_path)
