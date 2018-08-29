#!/usr/bin/env python3
"""
"""
import argparse
import datetime
import io
import logging
import os
import requests
import sys

import pandas as pd


BASE_URL = 'http://api.bitcoincharts.com/v1/trades.csv'
SYMBOL = 'bitstampUSD'


def get_args():
    """Parse and return command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # Configurations
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='The batch size to load the data. (default: 64)')
    parser.add_argument(
        '--num_epochs', type=int, default=30,
        help='The number of training epochs to run. (default: 30)')
    parser.add_argument(
        '--sequence_length', type=int, default=100,
        help='The number of ticks in the input sequence.')
    parser.add_argument(
        '--checkpoint',
        help='The path of the checkpoint file to load')

    # Flags
    parser.add_argument(
        '--cuda', default=False, action='store_true',
        help='Use GPU if available.')

    # Required arguments
    parser.add_argument(
        'data_file',
        help='The CSV tick data file. If file does not exist, new data will be'
             'downloaded.')

    return parser.parse_args(sys.argv[1:])


def get_logger():
    """Prepare formatted logger to stream and file.
    Returns:
        logging.Logger: The logger object.
    """
    # Prepare log directory.
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    # Create logger and formatter.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # Create and attach stream handler.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create and attach file handler.
    file_handler = logging.handlers.TimedRotatingFileHandler(
        'logs/log.txt', when='d', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def download_data(data_file):
    """Download BTC to USD tick data from bitcoincharts.com.

    Args:
        data (str): The file to save the new data into.
    """
    # Prepare log directory.
    try:
        os.mkdir('data')
    except FileExistsError:
        pass

    download_start_time = datetime.datetime.now()
    data_start_time = 0

    results = []

    while True:
        response = requests.get(
            f'{BASE_URL}?symbol={SYMBOL}&start={data_start_time}')

        if not response.ok:
            print('Request failed.')
            break

        result = pd.read_csv(io.StringIO(response.text),
                             names=('time', 'price', 'amount'))

        data_start_time = int(result.iloc[-1].time)
        results.append(result[result.time < data_start_time])

        print(datetime.datetime.fromtimestamp(data_start_time), len(result))

        if data_start_time > download_start_time:
            print('Finished.')
            break

    if len(results) > 0:
        pd.concat(results, ignore_index=True).to_csv(data_file)


def get_data(data_file, sequence_length):
    data = pd.read_data(data_file)

    price_prev = data.price[:-LENGTH].reset_index(drop=True)
    price_next = data.price[LENGTH:].reset_index(drop=True)
    rise = price_prev < price_next

    data = data[:-LENGTH]
    data = data.assign(rise=rise).dropna()



def main():
    args = get_args()
    logger = get_logger()

    if not os.path.isfile(args.data):
        logger.info('File not found. Downloading new data..')
        download_data(args.data_file)

    get_data(args.data_file, args.sequence_length)


if __name__ == '__main__':
    main()
