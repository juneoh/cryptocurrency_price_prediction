#!/usr/bin/env python3
"""Download and save BTC-USD tick data from coinmarketcap.com.

Example:
    ./data.py ticks.csv
"""
import argparse
import datetime
import io
import requests
import sys

import pandas as pd

from log import get_logger


BASE_URL = 'http://api.bitcoincharts.com/v1/trades.csv'
SYMBOL = 'bitstampUSD'


def get_args():
    """Parse and return command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        'data',
        help='The file to save the preprocessed tick data into.')

    return parser.parse_args(sys.argv[1:])


def main():
    args = get_args()
    logger = get_logger()

    download_start_time = datetime.datetime.now().timestamp()
    data_start_time = 0

    results = []

    while True:
        response = requests.get(
            f'{BASE_URL}?symbol={SYMBOL}&start={data_start_time}')

        if not response.ok:
            logger.error('Request failed.')
            break

        result = pd.read_csv(io.StringIO(response.text),
                             names=('time', 'price', 'amount'))

        data_start_time = int(result.iloc[-1].time)
        results.append(result[result.time < data_start_time])

        logger.info(f'Received {len(result)} ticks from '
                    f'{datetime.datetime.fromtimestamp(data_start_time)}')

        if data_start_time > download_start_time or len(result) == 1:
            break

    pd.concat(results, ignore_index=True).to_csv(args.data, index=False)
    logger.info(f'Saved file into {args.data}.')


if __name__ == '__main__':
    main()
