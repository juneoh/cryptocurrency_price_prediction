#!/usr/bin/env python3
"""Download and save BTC-USD tick data from bitcoincharts.com.

Example:
    $ ./data.py --help
    $ ./data.py ticks.csv
"""
import argparse
import datetime
import hashlib
import io
import os
import pdb
import requests
import sys

import pandas as pd

from log import get_logger


BASE_URL = 'http://api.bitcoincharts.com/v1/trades.csv'
SYMBOL = 'bitstampUSD'


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

    parser.add_argument(
        'csv_file',
        help='The file to save the preprocessed tick data into.')

    return parser.parse_args(sys.argv[1:])


def main():
    # Prepare log directory.
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    args = get_args()
    args_hash = hashlib.md5(repr(vars(args)).encode()).hexdigest()
    logger = get_logger(os.path.join('logs', f'logs.{args_hash}.txt'))

    download_start_time = datetime.datetime.now()
    data_start_time = datetime.datetime.fromtimestamp(0)

    results = []

    while True:
        response = requests.get(f'{BASE_URL}?symbol={SYMBOL}&start='
                                f'{int(data_start_time.timestamp())}')

        if not response.ok:
            logger.error('Request failed.')
            break

        result = pd.read_csv(io.StringIO(response.text),
                             names=('time', 'price', 'amount'))

        if len(result) == 0:
            break

        result.time = result.time.apply(datetime.datetime.fromtimestamp)

        logger.info(f'Received {len(result)} ticks from {result.time[0]}')

        data_start_time = result.iloc[-1].time
        results.append(result[result.time < data_start_time])

        if data_start_time > download_start_time:
            break

    # Concatenate and save ticks.
    pd.concat(results, ignore_index=True).to_csv(args.csv_file, index=False)
    logger.info(f'Saved file into {args.csv_file}.')


if __name__ == '__main__':
    main()
