#!/usr/bin/env python3
"""Train given model architecture on tick data.

Examples:
    $ ./train.py PureRNN ticks.csv
    $ ./train.py RNNLinear ticks.csv
    $ ./train.py CNN ticks.csv
"""
import argparse
import pdb
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import tqdm

import cnn
from log import get_logger
import rnn


def get_args():
    """Parse and return command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument(
        'architecture',
        help='The model architecture: either PureRNN, RNNLinear, or CNN.')
    parser.add_argument(
        'data',
        help='The CSV file containing the data.')

    # Performance configurations
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='The number of DataLoader workers.')
    parser.add_argument(
        '--cuda', default=False, action='store_true',
        help='Use CUDA.')

    # Hyperparameters
    parser.add_argument(
        '--num_epochs', type=int, default=30,
        help='The number of epochs to train.')
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='The number of samples in a batch.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.1,
        help='The initial learning rate.')
    parser.add_argument(
        '--num_layers', type=int, default=2,
        help='The number of layers of the model.')
    parser.add_argument(
        '--hidden_size', type=int, default=256,
        help='The size off the hidden state.')
    parser.add_argument(
        '--sequence_length', type=int, default=10,
        help='The number of ticks in a single sample.')
    parser.add_argument(
        '--split', default='8,2',
        help='The comma-separated ratio of training and validation datasets.')

    return parser.parse_args(sys.argv[1:])


class TickDataset:
    """A PyTorch-compatible dataset class for tick data.

    Arguments:
        data (pandas.DataFrame): The DataFrame object of a timeseries data,
            with each column representing a feature.
        target (iterable): An iterable containing the target values.
        label (iterable): An iterable containing human labels, e.g. time.
        sequence_length (int): The number of ticks in a single sample.
    """
    def __init__(self, data, target, label, sequence_length):
        self.data = data
        self.target = target
        self.label = label
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, key):
        start = key
        end = start + self.sequence_length

        data = torch.from_numpy(self.data.iloc[start:end].values)
        data = data.type(torch.FloatTensor)
        target = torch.Tensor([self.target[end]])
        label = self.label[end]

        return data, target, label


def main():
    torch.manual_seed(0)

    logger = get_logger()
    args = get_args()

    ticks = pd.read_csv(args.data)

    # Normalize price.

    scaler = StandardScaler()
    ticks.price = scaler.fit_transform(
        ticks.price.values.reshape(-1, 1)).reshape(-1)

    # Create target list.

    #target = []
    #for i in range(len(ticks) - 6):
    #    target.append(ticks.price[i] < max(ticks.price[i+1:i+6]))
    #target = pd.Series(target)
    #target.index = range(6, len(ticks))

    price_prev = ticks.price[:-1].reset_index(drop=True)
    price_next = ticks.price[1:].reset_index(drop=True)
    target = price_prev < price_next
    target.index = range(1, len(ticks))

    label = ticks.time
    data = ticks[['price', 'amount']]

    dataset = TickDataset(data, target, label, args.sequence_length)

    split = list(map(int, args.split.split(',')))
    n_train = int(len(dataset) / sum(split) * split[0])

    data_train = DataLoader(Subset(dataset, range(0, n_train - 1)),
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=True)
    data_val = DataLoader(Subset(dataset, range(n_train, len(dataset))),
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          pin_memory=True,
                          shuffle=False)

    if args.architecture == 'RNNLinear':
        model = rnn.RNNLinear(input_size=2,
                              output_size=1,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers)
    elif args.architecture == 'PureRNN':
        model = rnn.PureRNN(input_size=2,
                            output_size=1,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers)
    else:
        raise RuntimeError('Unrecognized architecture.')

    if args.cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    description = '[{mode}] batch accuracy: {accuracy:.3f}'

    for epoch in range(args.num_epochs):
        progress_bar = tqdm.tqdm(
            total=len(data_train) + len(data_val),
            unit='batch',
            desc='[train] batch accuracy: 0.000',
            leave=False)

        # Train.

        model.train()
        n_total = 0
        n_correct = 0

        for data, target, label in data_train:
            with torch.set_grad_enabled(True):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

            pred = torch.sigmoid(output.squeeze().cpu()) > 0.5
            true = target.squeeze().cpu().type(torch.ByteTensor)
            correct = int(sum(pred == true))
            total = len(data)
            accuracy = correct / total
            n_total += total
            n_correct += correct
            
            pdb.set_trace()

            progress_bar.update(1)
            progress_bar.set_description(
                description.format(mode='train', accuracy=accuracy))

        train_acc = n_correct / n_total

        scheduler.step()

        # Validate.

        model.eval()
        n_total = 0
        n_correct = 0

        for data, target, label in data_val:
            with torch.no_grad():
                if args.cuda:
                    data = data.cuda()

                output = torch.sigmoid(model(data))

            pred = output.squeeze().cpu() > 0.5
            true = target.squeeze().cpu().type(torch.ByteTensor)
            correct = int(sum(pred == true))
            total = len(data)
            accuracy = correct / total
            n_total += total
            n_correct += correct

            progress_bar.update(1)
            progress_bar.set_description(
                description.format(mode='val', accuracy=accuracy))

        val_acc = n_correct / n_total

        progress_bar.close()

        print(f'Epoch {epoch}/{args.num_epochs}: '
              f'train accuracy {train_acc:.3f}, '
              f'val accuracy {val_acc:.3f}')


if __name__ == '__main__':
    main()
