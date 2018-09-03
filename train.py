#!/usr/bin/env python3
"""Train given model architecture on tick data.

Examples:
    $ ./train.py --help
    $ ./train.py PureRNN ticks.csv
    $ ./train.py RNNLinear ticks.csv
    $ ./train.py CNN ticks.csv
"""
import argparse
import hashlib
import os
import pdb
import sys

import pandas as pd
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import tqdm

import cnn
from log import get_logger
import rnn


class RawTextDefaultsArgumentFormatter(
        argparse.RawTextHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter):
    """A ArgumentParser formatter for newlines and default values.
    """
    pass


def get_args():
    """Parse and return command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawTextDefaultsArgumentFormatter)

    # Required arguments
    parser.add_argument(
        'architecture',
        help='The model architecture: either PureRNN, RNNLinear, or CNN.')
    parser.add_argument(
        'csv_file',
        help='The CSV file containing the tick data.')

    # Data settings
    data = parser.add_argument_group('data settings')
    data.add_argument(
        '--split', default='8,2',
        help='The comma-separated ratio of training and validation datasets.')
    data.add_argument(
        '--sequence_length', type=int, default=32,
        help='The number of ticks in a single sample.')
    data.add_argument(
        '--batch_size', type=int, default=64,
        help='The number of samples in a batch.')

    # Model settings
    model = parser.add_argument_group('model settings')
    model.add_argument(
        '--num_layers', type=int, default=4,
        help='The number of layers of the model.')
    model.add_argument(
        '--hidden_size', type=int, default=64,
        help='The size off the hidden state.')
    model.add_argument(
        '--state_file',
        help='The pretrained model state file to load.')
    model.add_argument(
        '--save_every', type=int, default=10,
        help='The number of epochs to save model state.')

    # Optimizer settings
    optimizer = parser.add_argument_group('optimizer settings')
    optimizer.add_argument(
        '--num_epochs', type=int, default=300,
        help='The number of epochs to train. If 0, run validation once.')
    optimizer.add_argument(
        '--optimizer', default='SGD',
        help='The optimizer to use, either SGD or Adam.')
    optimizer.add_argument(
        '--learning_rate', type=float, default=4,
        help='The initial learning rate.')
    optimizer.add_argument(
        '--weight_decay', type=float, default=0.1,
        help='The optimizer L2 penalty.')
    optimizer.add_argument(
        '--step_size', type=int, default=10,
        help='The step size for the learning rate scheduler.')
    optimizer.add_argument(
        '--gamma', type=float, default=0.5,
        help='The gamma value to scale learning rate per step size.')

    # Performance settings
    performance = parser.add_argument_group('performance settings')
    performance.add_argument(
        '--num_workers', type=int, default=4,
        help='The number of DataLoader workers.')
    performance.add_argument(
        '--use_cuda', default=False, action='store_true',
        help='Use CUDA.')

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

        input = torch.from_numpy(self.data.iloc[start:end].values)
        input = input.type(torch.FloatTensor)
        target = torch.Tensor([self.target[end]])
        label = self.label[end]

        return input, target, label


def get_data(csv_file, split, sequence_length, batch_size, num_workers,
             use_cuda):
    """Load, preprocess, and return tick data.
    
    Returns:
        (tuple): A tuple of training and validation data loaders and
        positive class loss weight.
    """
    # Load CSV file as a Pandas Dataset
    ticks = pd.read_csv(csv_file)

    # Normalize values.
    ticks.price = scale(ticks.price.values)
    ticks.amount = scale(ticks.amount.values)

    # Create target list.
    price_prev = ticks.price[:-1].reset_index(drop=True)
    price_next = ticks.price[1:].reset_index(drop=True)
    target = price_prev < price_next
    target.index = range(0, len(ticks) - 1)

    # Create Dataset.
    label = ticks.time[:-1]
    data = ticks[['price', 'amount']][:-1]
    dataset = TickDataset(data, target, label, sequence_length)

    # Split and wrap with DataLoaders.
    split = list(map(int, split.split(',')))
    num_train = int(len(dataset) / sum(split) * split[0])
    data_train = DataLoader(Subset(dataset, range(0, num_train - 1)),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=use_cuda,
                            drop_last=True)
    data_val = DataLoader(Subset(dataset, range(num_train, len(dataset))),
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=use_cuda,
                          shuffle=False)

    # Calculate positive weight.
    pos_weight = sum(~target[:num_train]) / sum(target[:num_train])
    pos_weight = torch.Tensor([pos_weight])
    if use_cuda:
        pos_weight = pos_weight.cuda()

    return data_train, data_val, pos_weight


def get_model(architecture, num_layers, hidden_size, sequence_length,
              use_cuda):
    """Return the model for the given options.

    Returns:
        (torch.nn.Module): The model.
    """
    if architecture == 'RNNLinear':
        model = rnn.RNNLinear(input_size=2,
                              output_size=1,
                              num_layers=num_layers,
                              hidden_size=hidden_size)

    elif architecture == 'PureRNN':
        model = rnn.PureRNN(input_size=2,
                            output_size=1,
                            num_layers=num_layers,
                            hidden_size=hidden_size)

    elif architecture == 'CNN':
        model = cnn.CNN(input_size=2,
                        output_size=1,
                        num_layers=num_layers,
                        hidden_size=hidden_size,
                        sequence_length=sequence_length)

    else:
        raise RuntimeError('Unrecognized architecture.')

    if use_cuda:
        return model.cuda()
    else:
        return model


def train(model, loss_function, optimizer, data):
    """Train the model on the given data.

    Args:
        model (torch.nn.Module): A PyTorch model.
        loss_function (torch.nn.Module): The loss function to compare model
            outputs with target values.
        optimizer (torch.optim.Optimizer): The optimizer algorithm to train the
            model.
        data (torch.utils.data.DataLoader): The data to train on.

    Returns:
        (float): The mean batch loss.
    """
    loss_sum = 0

    # Set the model in train mode.
    model.train()

    # Create progress bar.
    progress_bar = tqdm.tqdm(total=len(data),
                             unit='batch',
                             desc='[train] batch loss: 0.00000',
                             leave=False)

    # Loop through training batches.
    for inputs, targets, labels in data:

        # Reset gradients.
        optimizer.zero_grad()

        # Send data to GPU if CUDA is enabled.
        if next(model.parameters()).is_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Feed forward.
        with torch.set_grad_enabled(True):
            outputs = model(inputs)

        # Compute loss.
        loss = loss_function(outputs, targets)

        # Compute gradients.
        loss.backward()

        # Update parameters.
        optimizer.step()

        # Update progress bar.
        progress_bar.update(1)
        progress_bar.set_description(
            '[train] batch loss: {loss:.5f}'.format(loss=loss.item()))

        # Accumulate loss sum.
        loss_sum += loss.item()

    # Close progress bar.
    progress_bar.close()

    return loss_sum / len(data)


def evaluate(model, data):
    """Evaluate the model on the given data.

    Args:
        model (torch.nn.Module): A PyTorch model.
        data (torch.utils.data.DataLoader): The data to train on.

    Returns:
        (float): The overall precision.
    """
    true_positives_total = 0
    positives_total = 0

    # Set the model on evaluatio mode.
    model.eval()

    # Create progress bar.
    progress_bar = tqdm.tqdm(total=len(data),
                             unit='batch',
                             desc='[evaluate] batch precision: 0.00000',
                             leave=False)

    # Loop through validation batches.
    for inputs, targets, labels in data:

        # Send data to GPU if CUDA is enabled.
        if next(model.parameters()).is_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Feed forward.
        with torch.no_grad():
            outputs = model(inputs)

        # Make predictions.
        predictions = torch.sigmoid(outputs).squeeze().cpu() >= 0.5
        targets = targets.type(torch.ByteTensor).squeeze()

        true_positives = sum(predictions & targets).item()
        positives = sum(predictions).item()
        batch_precision = 0 if positives == 0 else true_positives / positives

        progress_bar.update(1)
        progress_bar.set_description(
            f'[evaluate] batch precision: {batch_precision:.5f}')

        # Accumulate metrics.
        true_positives_total += true_positives
        positives_total += positives

    # Close progress bar.
    progress_bar.close()

    return 0 if positives_total == 0 else true_positives_total / positives_total


def main():
    # Fix random seed.
    torch.manual_seed(0)

    # Prepare log directory.
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    # Create states directory.
    try:
        os.mkdir('states')
    except FileExistsError:
        pass

    # Intialize objects.

    args = get_args()
    args_hash = hashlib.md5(repr(vars(args)).encode()).hexdigest()
    logger = get_logger(os.path.join('logs', f'logs.{args_hash}.txt'))

    data_train, data_val, pos_weight = get_data(args.csv_file,
                                                args.split,
                                                args.sequence_length,
                                                args.batch_size,
                                                args.num_workers,
                                                args.use_cuda)
    model = get_model(args.architecture,
                      args.num_layers,
                      args.hidden_size,
                      args.sequence_length,
                      args.use_cuda)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise RuntimeError('Unrecognized optimizer.')

    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=args.step_size,
                                             gamma=args.gamma)

    # Load state.
    if args.state_file:
        model.load_state_dict(torch.load(args.state_file))

    # Log command arguments.
    logger.info(' '.join(sys.argv))
    logger.info(vars(args))
    logger.info(f'Arguments hash: {args_hash}')

    # If the number of epochs is 0, validate once.
    if args.num_epochs == 0:
        accuracy = evaluate(model, data_val)
        logger.info(f'Validation accuracy: {accuracy:.5f}')

    # Loop epochs.
    for epoch in range(args.num_epochs):

        logger.info(f'Epoch {epoch}')

        # Train.
        mean_loss = train(model, loss_function, optimizer, data_train)
        logger.info(f'  - [training] mean loss: {mean_loss:.5f}')

        # Validate.
        precision = evaluate(model, data_val)
        logger.info(f'  - [validation] precision: {precision:.5f}')

        # Save model state.
        if (epoch + 1) % args.save_every == 0:
            state_file = os.path.join('states', f'{epoch}.{args_hash}.pth')
            torch.save(model.state_dict(), state_file)
            logger.info(f'Saved model state: {state_file}')

        # Update learning rate.
        lr_scheduler.step()


if __name__ == '__main__':
    main()
