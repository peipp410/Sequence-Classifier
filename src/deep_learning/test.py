#!/usr/bin/python

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

import util

def getArg():
    parser = argparse.ArgumentParser(description='Script argparse setting')
    parser.add_argument('-d', dest='device', default='cuda', help='the device: cpu or cuda')
    parser.add_argument('-c', dest='input', default='./test.fa',
                        help='Path to input file, default is ./test.fa')
    parser.add_argument('-o', dest='output', default='./output.txt',
                        help='Path to output file, default is ./output.txt')
    args = parser.parse_args()
    device, input, output = args.device, args.input, args.output

    return device, input, output

def test(model, data):
    params = {'batch_size': 32,
              'shuffle': False,
              'num_workers': 6,
              'drop_last': True}
    test_dataloader = DataLoader(data, **params)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for seqs in test_dataloader:
            seqs = seqs.to(device)
            logits = model(seqs)
            pred_int = torch.argmax(logits, dim=1)
            y_pred = y_pred + pred_int.flatten().tolist()
    return y_pred


def main(device, input, output):
    if device == 'cpu':
        model = torch.load('./model210.37575.pth', map_location='cpu')
    elif device == 'cuda':
        model = torch.load('./model210.37575.pth')

    seqs = util.getSeqs(input)
    seqs_array = util.one_hot_encode(seqs)
    print('Strart predict!')
    pred = test(model, torch.tensor(seqs_array, dtype=torch.float32))

    with open(output, 'w') as f:
        for i in range(len(pred)):
            f.write('> ' + str(i) + '\n')
            f.write(util.transferLabel(pred[i]) + '\n')

    print('Finished!')

if __name__ == "__main__":
    start = time.time()
    device, input, output = getArg()
    main(device, input, output)
    end = time.time()
    print('Running time: {}'.format(end-start))
