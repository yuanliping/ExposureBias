import subprocess


def cat_eng_char(path):
    is_gold = 'gold' in path
    f = open(path, 'r', encoding='utf-8')
    contents = f.read()
    f.close()
    contents = contents.replace('| D I G I T |', '|DIGIT|')
    contents = contents.replace('| U N K |', '|UNK|')
    if is_gold:
        contents = contents.replace('< u n k >', '<unk>')
    else:
        contents = contents.replace('< u n k >', '<knu>')
        contents = contents.replace('< k n u >', '<knu>')
    f = open(path, 'w', encoding='utf-8')
    f.write(contents)
    f.close()


def write_unigram_data(data, path):
    with open(path, 'w', encoding='utf-8')as f:
        for datum in data:
            for word in datum:
                for character in word:
                    f.write(character)
                    f.write(' ')
            f.write('\n')
    cat_eng_char(path)


def read_data(path):
    data = list()
    with open(path, 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip('\n')
            datum = list()
            for word in line.split():
                datum.append(word)
            data.append(datum)

    return data


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


def test_model(device_id, epoch, mode='test'):
    gold_path = 'generation/cuda{}_gold_{}.txt'.format(device_id, epoch)
    data = read_data(gold_path)
    write_unigram_data(data, gold_path)
    test_path = 'generation/cuda{}_{}_{}.txt'.format(device_id, mode, epoch)
    data = read_data(test_path)
    write_unigram_data(data, test_path)

    p = subprocess.Popen('perl multi-bleu.perl ' + gold_path + ' < ' + test_path, shell=True)
    p.wait()

    d1_num, d1_rate = distinct_1(test_path)
    d2_num, d2_rate = distinct_2(test_path)
    print('distinct-1: (%d, %f), distinct-2: (%d, %f)' % (d1_num, d1_rate, d2_num, d2_rate))
