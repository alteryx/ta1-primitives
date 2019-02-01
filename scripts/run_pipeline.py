import argparse
import io
import os
import sys
import tarfile
import urllib

from d3m import index
from d3m.container.dataset import Dataset
from d3m.metadata.base import Context
from d3m.metadata.problem import parse_problem_description
from d3m.runtime import Runtime

import featuretools_ta1

DATASETS_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)
DATA_URL = 'https://d3m-data-dai.s3.amazonaws.com/datasets/{}.tar.gz'


def _download(dataset_name, dataset_path):
    url = DATA_URL.format(dataset_name)

    print('Downloading dataset {} from {}'.format(dataset_name, url))
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    print('Extracting dataset into {}'.format(dataset_path))
    with tarfile.open(fileobj=bytes_io, mode='r:gz') as tf:
        tf.extractall(os.path.dirname(dataset_path))


def ensure_downloaded(dataset_name, datasets_path):
    dataset_path = os.path.join(datasets_path, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        _download(dataset_name, dataset_path)


def load_dataset(root_path, phase):
    path = os.path.join(root_path, phase, 'dataset_' + phase, 'datasetDoc.json')
    return Dataset.load(dataset_uri='file://' + path)


def load_problem(root_path, phase):
    path = os.path.join(root_path, phase, 'problem_' + phase, 'problemDoc.json')
    return parse_problem_description(path)


def run_pipeline(pipeline, dataset_name, datasets_path):
    ensure_downloaded(dataset_name, datasets_path)

    root_path = os.path.join(os.path.abspath(datasets_path), dataset_name)
    train_dataset = load_dataset(root_path, 'TRAIN')
    train_problem = load_problem(root_path, 'TRAIN')

    # Creating an instance on runtime with pipeline description and problem description.
    runtime = Runtime(
        pipeline=pipeline,
        problem_description=train_problem,
        context=Context.TESTING
    )

    # Fitting pipeline on input dataset.
    fit_results = runtime.fit(inputs=[train_dataset])
    fit_results.check_success()

    # Producing results using the fitted pipeline.
    test_dataset = load_dataset(root_path, 'TEST')
    produce_results = runtime.produce(inputs=[test_dataset])
    produce_results.check_success()

    print(produce_results.values)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Get the demo pipeline from a primitive and run it on a dataset.')
    parser.add_argument('-p', '--datasets-path', default=DATASETS_PATH,
                        help="Folder where datasets can be found.")
    parser.add_argument('-d', '--dataset', help="Dataset to run the pipeline with.")
    parser.add_argument('-r', '--run-pipeline', action='store_true',
                        help="Run the pipeline on the dataset.")
    parser.add_argument('primitives', nargs='*', help="Name of the primitives to evaluate.")

    args = parser.parse_args()

    if args.primitives:
        primitives = [index.get_primitive(primitive) for primitive in args.primitives]
    else:
        primitives = featuretools_ta1.PRIMITIVES

    for primitive in primitives:
        print("Running primitive {}".format(primitive))
        pipeline = primitive.get_demo_pipeline()

        dataset = args.dataset or pipeline.dataset
        datasets_path = os.path.abspath(args.datasets_path)

        run_pipeline(pipeline, dataset, datasets_path)
