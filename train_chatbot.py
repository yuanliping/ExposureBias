from CTG import bleu

from CTG.datasets.translation import TranslationTask
from CTG.model import WEaMTranslator, GumbelTranslator, TransformerWEaMTranslator
from CTG.criterion import ReconstructionLoss, KLLoss, FeatureRegularizer
from CTG.lr_scheduler import WarmupStableLinearDecay

from test_chatbot import test_model

import math
import random
import time

import torch
from torch.optim import Adam


def main(args):
    device_id = args['device_id']
    torch.cuda.set_device(device_id)

    log_path = args['log']['log_path']
    print(args.__str__())

    seed = args['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    task = TranslationTask(args['task'])
    src_dict, tgt_dict = task.get_source_dictionary(), task.get_target_dictionary()
    if args['model']['name'] == 'WEaM':
        model = WEaMTranslator.build_model(src_dict, tgt_dict, args['model'])
    elif args['model']['name'] == 'Gumbel':
        model = GumbelTranslator.build_model(src_dict, tgt_dict, args['model'])
    elif args['model']['name'] == 'TWEaM':
        model = TransformerWEaMTranslator.build_model(src_dict, tgt_dict, args['model'])
    else:
        return 0
    criterion = ReconstructionLoss(args['criterion'])
    optimizer = Adam(model.parameters(),
                     lr=args['optimization']['lr'])
    lr_scheduler = WarmupStableLinearDecay(args['optimization'], optimizer)

    kl = KLLoss() if args['model']['vae'] else None
    fr = FeatureRegularizer(tgt_dict, args['criterion']) if args['optimization']['regularize_feature'] else None

    epoch, steps = 0, 0
    gen_path = args['log']['gen_path']
    validate_interval = args['log']['validate_interval']
    max_epoch = args['optimization']['max_epoch']
    max_updates = args['optimization']['end_decay']
    best_valid, best_test = 0, 0
    while lr_scheduler.get_lr() > 0 and epoch < max_epoch:
        epoch += 1
        print(lr_scheduler.get_lr())
        print("epoch: %d, %s" % (epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        ppl, steps = train(task, model, criterion, kl, fr,
                           optimizer, lr_scheduler,
                           epoch, steps, max_updates)
        temperature = max(0, 1 - steps / max_updates)
        print('Training Temperature: {}\t Perplexity: {}'.format(temperature, ppl))
        if epoch % validate_interval == 0:
            fgold = open('{}/cuda{}_gold_{}.txt'.format(gen_path, device_id, epoch), 'w')
            ftest = open('{}/cuda{}_valid_{}.txt'.format(gen_path, device_id, epoch), 'w')
            validate(task, 'valid', model, criterion,
                     steps, max_updates, fgold, ftest)
            fgold.close()
            ftest.close()
            test_model(device_id, epoch, mode='valid')

            fgold = open('{}/cuda{}_gold_{}.txt'.format(gen_path, device_id, epoch), 'w')
            ftest = open('{}/cuda{}_test_{}.txt'.format(gen_path, device_id, epoch), 'w')
            validate(task, 'test', model, criterion,
                     steps, max_updates, fgold, ftest)
            fgold.close()
            ftest.close()
            test_model(device_id, epoch, mode='test')


def train(task, model, criterion, kl, fr, optimizer, lr_scheduler,
          epoch, steps, max_updates):
    model.train(True)
    criterion.train(True)
    batch_iter = task.get_iterator('train', epoch, shuffle=True)
    tot_loss, cnt = 0, 0
    for samples, net_input in batch_iter:
        '''TODO: Word Dropout'''
        net_input['temperature'] = 1 - steps / max_updates
        net_input['seqlen'] = int(torch.max(net_input['tgt_lengths']))
        net_output = model(net_input)
        generation, gold = net_output['generation'], net_output['tgt']
        non_pad_mask = net_output['tgt_non_pad_mask']
        hidden_embed = net_output['hidden_embed']
        gold_embed = net_output['gold_embed']
        nll_loss = criterion(generation, gold, non_pad_mask)
        loss = nll_loss
        if kl is not None:
            mean, logvar = net_output['mean'], net_output['logvar']
            kl_loss = kl(mean, logvar)
            loss = nll_loss + kl_loss
        if fr is not None:
            fr_loss = fr(net_output['gold_embed'], net_output['generation_embed'], non_pad_mask)
            loss = nll_loss + fr_loss
        loss.backward()
        tot_loss += nll_loss.data.item()
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
        lr_scheduler.step_update(steps)
        cnt += 1
        if steps == max_updates:
            break
    tot_loss /= cnt
    ppl = math.exp(tot_loss)
    return ppl, steps


def validate(task, split, model, criterion,
             steps, max_updates, fgold, ftest):
    model.train(False)
    criterion.train(False)
    tgt_dict = task.get_target_dictionary()
    batch_iter = task.get_iterator(split, shuffle=False, single_sample=False)
    tot_loss, cnt = 0, 0
    for samples, net_input in batch_iter:
        net_input['temperature'] = 1 - steps / max_updates
        # net_input['seqlen'] = int((torch.max(net_input['src_lengths'])) * 1.2 + 10)
        net_output = model(net_input)
        generation, gold = net_output['generation'], net_output['tgt']
        print_results(gold, generation, tgt_dict, fgold, ftest)
        cnt += 1
    tot_loss /= cnt


def print_results(gold, generation, dictionary, fgold, ftest):
    _, generation = torch.max(generation, dim=-1)
    for i in range(gold.size(0)):
        _gen, _gold = generation[i].data.tolist(), gold[i].data.tolist()
        gen_sent = generate_sentence_from_indices(_gen, dictionary)
        gold_sent = generate_sentence_from_indices(_gold, dictionary)
        fgold.write('{}\n'.format(gold_sent))
        ftest.write('{}\n'.format(gen_sent))


def generate_sentence_from_indices(indices, dictionary):
    tokens = list(map(lambda x: dictionary[x], indices))
    sent = ' '.join(tokens)
    idx = sent.find('</s>')
    if idx > 0:
        sent = sent[:idx]
    return sent.replace('@@ ', '')


args = {'task':
            {'data': 'datasets/STC',
             'source': 'post',
             'target': 'response',
             'splits': ['train', 'valid', 'test'],
             'batch_size': 200,  # 128 for de-en, 32 for en-vi
             'threshold': 3,  # 3 for de-en, 5 for en-vi
             'max_tokens': 100000000,
             'nwords': 40000,
             'maxlen': 10,  # 50 for de-en, 150 for en-vi
             'share_vocab': False},
        'optimization':
            {'max_epoch': 200,
             'lr': 1e-3,
             'warmup_updates': 200,
             'start_decay': 90000,  # 25k for de-en, 100k for en-vi
             'end_decay': 270000,  # 50k for de-en, 200k for en-vi
             'regularize_feature': False},
        'model':
            {'name': 'WEaM',
             'gold_input': False,
             'vae': True,
             'embed_dim': 256,
             'encoder_out_dim': 256,
             'head_num': 8,
             'dropout': 0.3,
             'maxlen': 10,  # 50 for de-en, 150 for en-vi
             'layers': 3,
             'encoder_attention': True,
             'tau': 0.5,
             'residual_connection': False,
             'mask_low_probs': True,
             'margin': '(0,6)',
             'warmup_with_gold': True,
             'test_argmax': True,
             'rbf': False,
             'logit_anneal': False,
             'warmup_margin_with_gold': True,
             'min_gold_margin_rate': 0.5,
             'share_src_tgt_embed': False},
        'criterion':
            {'eps': 0.1,
             'beta': 1e-4},
        'log':
            {'gen_path': 'generation',
             'validate_interval': 1,
             'log_path': 'log'},
        'device_id': 7,
        'seed': 1}
if __name__ == '__main__':
    print('mweam, 60epoch')
    main(args)
