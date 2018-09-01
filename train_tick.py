#!/usr/bin/env python3.6
"""
Todos:
    - try normalizing values
    - try using LSTM and GRU
    - try regression
"""
import argparse
import pdb

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm


NUM_EPOCHS = 200
BATCH_SIZE = 100
LENGTH = 10
HIDDEN_SIZE = 128
NUM_LAYERS = 4
DATA = 'btc.tick.csv'
SPLIT = '9,1,0'
COLUMNS = ('price', 'amount')


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()

        self.rnn = nn.LSTM(input_size=2,
                           hidden_size=HIDDEN_SIZE,
                           num_layers=NUM_LAYERS,
                           batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, input):
        output, _ = self.rnn(input)
        output = self.fc(output[:, -1, :])

        return output


class MarketDataset:
    def __init__(self, df, length):
        self.df = df.reset_index(drop=True)
        self.length = length

    def __len__(self):
        return len(self.df) - self.length

    def __getitem__(self, key):
        data = torch.from_numpy(
            self.df.ix[key:(key + self.length - 1), COLUMNS].values)
        data = data.type(torch.FloatTensor)
        rise = torch.Tensor([self.df.rise.values[key + self.length - 1]])
        time = self.df.time.values[key + self.length - 1]

        return (data, rise, time)


def main():
    # Load CSV file into a new DataFrame object and reindex by ascending date.

    dataframe = pd.read_csv(DATA)

    # Create target label column 'rise'.

    price_now = dataframe.price[:-LENGTH].reset_index(drop=True)
    price_next = dataframe.price[LENGTH:].reset_index(drop=True)
    price = price_next - price_now

    scaler = StandardScaler()
    price = scaler.fit_transform(price.values.reshape(-1, 1)).reshape(-1)

    dataframe = dataframe[:-LENGTH]
    dataframe = dataframe.assign(price=price, rise=(price_now < price_next)).dropna()

    # Split data into training, validation, and test.

    split = list(map(int, SPLIT.split(',')))
    n_train = int(len(dataframe) / sum(split) * split[0])
    n_val = int(len(dataframe) / sum(split) * split[1])

    dataframe_train = dataframe[:n_train]
    dataframe_val = dataframe[n_train:(n_train + n_val)]
    dataframe_test = dataframe[(n_train + n_val):]

    # Prepare data loaders.

    dataloader_train = DataLoader(
        MarketDataset(dataframe_train, length=LENGTH),
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True)
    dataloader_val = DataLoader(
        MarketDataset(dataframe_val, length=LENGTH),
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=False)
    dataloader_test = DataLoader(
        MarketDataset(dataframe_test, length=LENGTH),
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=False)

    # Train.

    model = RNNModel()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    description = '[{mode}] batch accuracy: {accuracy:.3f}'

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm.tqdm(
            total=len(dataloader_train) +
            len(dataloader_val),
            unit='batch',
            desc='[train] batch accuracy: 0.000',
            leave=False)
        model.train()
        outputs = []
        rises = []

        for i, (data, rise, date) in enumerate(dataloader_train):
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, rise)
                loss.backward()

                optimizer.step()

                outputs.append(output.squeeze())
                rises.append(rise.squeeze())

                accuracy = accuracy_score(rise.squeeze(),
                                          output.squeeze() > 0.5)
                progress_bar.update(1)
                progress_bar.set_description(
                    description.format(mode='train', accuracy=accuracy))

        train_acc = accuracy_score(torch.cat(rises), torch.cat(outputs) > 0.5)

        scheduler.step()

        model.eval()
        outputs = []
        rises = []
        for i, (data, rise, date) in enumerate(dataloader_val):
            output = nn.functional.sigmoid(model(data))
            outputs.append(output.squeeze())
            rises.append(rise.squeeze())

            accuracy = accuracy_score(rise.squeeze(),
                                      output.squeeze() > 0.5)
            progress_bar.update(1)
            progress_bar.set_description(
                description.format(mode='val', accuracy=accuracy))


        val_acc = accuracy_score(torch.cat(rises), torch.cat(outputs) > 0.5)
        pdb.set_trace()

        progress_bar.close()
        print(f'Epoch {epoch}/{NUM_EPOCHS}: '
              f'train accuracy {train_acc:.3f}, '
              f'val accuracy {val_acc:.3f}')


if __name__ == '__main__':
    main()
