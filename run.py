import argparse
import gc
import json
import logging
import sys
import warnings

import pandas as pd
import tabulate
import tqdm

import sdgym

synthesizers = ['PAR']

datasets = [
    'Libras',
    'AtrialFibrillation',
    'BasicMotions',
    'ERing',
    'RacketSports',
    'Epilepsy',
    'PenDigits',
    'JapaneseVowels',
    'StandWalkJump',
    'FingerMovements',
    'EchoNASDAQ',
    'Handwriting',
    'UWaveGestureLibrary',
    'NATOPS',
    'ArticularyWordRecognition',
    'Cricket',
    'SelfRegulationSCP2',
    'LSST',
    'SelfRegulationSCP1',
    'CharacterTrajectories',
    'HandMovementDirection',
    'EthanolConcentration',
    'SpokenArabicDigits',
    'Heartbeat',
    'PhonemeSpectra',
    'MotorImagery',
    'DuckDuckGeese',
    'PEMS-SF',
    'EigenWorms',
    'InsectWingbeat',
    'FaceDetection'
]

def _env_setup(logfile, verbosity):
    gc.enable()
    warnings.simplefilter('ignore')

    FORMAT = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    level = (3 - verbosity) * 10
    logging.basicConfig(filename=logfile, level=level, format=FORMAT)
    logging.getLogger('sdgym').setLevel(level)
    logging.getLogger('sdmetrics').setLevel(level)
    logging.getLogger().setLevel(logging.WARN)

def _print_table(data, sort=None, reverse=False, format=None):
    if sort:
        sort_fields = sort.split(',')
        for field in sort_fields:
            data = data.sort_values(field, ascending=not reverse)

    if format:
        for field, formatter in format.items():
            data[field] = data[field].apply(formatter)

    if 'error' in data:
        error = data['error']
        if pd.isnull(error).all():
            del data['error']
        else:
            long_error = error.str.len() > 30
            data.loc[long_error, 'error'] = error[long_error].str[:30] + '...'

    print(tabulate.tabulate(
        data,
        tablefmt='github',
        headers=data.columns,
        showindex=False
    ))

def _run(distributed=False, workers=1):
    if distributed:
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as ie:
            ie.msg += (
                '\n\nIt seems like `dask` is not installed.\n'
                'Please install `dask` and `distributed` using:\n'
                '\n    pip install dask distributed'
            )
            raise

        processes = workers > 1
        client = Client(
            LocalCluster(
                processes=processes,
                n_workers=workers,
            ),
        )

        workers = 'dask'

    scores = sdgym.run(
        synthesizers=synthesizers,
        datasets=datasets,
        workers=workers,
        show_progress=True,
    )

    if scores is not None:
        _print_table(scores)

    scores.to_csv('result.csv', index=False)

if __name__ == "__main__":
    _run()

