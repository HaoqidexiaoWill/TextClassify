import argparse
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from metric import compute_DOUBAN
from Model.BIMPM import BIMPM

def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    ids,scores,labels = [],[],[]
    losses = []
    for batch in iterator:


        s1, s2 = getattr(batch, 'q1'), getattr(batch, 'q2')
        kwargs = {'p': s1, 'h': s2}

        if args.use_char_emb:
            char_p = Variable(torch.LongTensor(data.characterize(s1)))
            char_h = Variable(torch.LongTensor(data.characterize(s2)))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            kwargs['char_p'] = char_p
            kwargs['char_h'] = char_h

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        losses.append(batch_loss.item())
        '''
        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)
        '''
        logit = pred.detach().cpu().numpy()
        label = batch.label.to('cpu').numpy()
        ids.extend(getattr(batch, 'id'))
        scores.append(logit)
        labels.append(label)

    labels = np.concatenate(labels, 0)
    scores = np.concatenate(scores, 0)
    # print(len(ids),labels.shape,scores.shape)
    eval_DOUBAN_MRR,eval_DOUBAN_mrr,eval_DOUBAN_MAP,eval_Precision1 = compute_DOUBAN(ids,scores,labels)



    # acc /= size
    # acc = acc.cpu().data[0]
    return np.mean(losses), eval_DOUBAN_MRR


def load_model(args, data):
    model = BIMPM(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.to(args.device)

    return model

