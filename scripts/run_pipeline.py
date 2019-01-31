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


def load_dataset(dataset_name, datasets_path):
    dataset_root = os.path.join(datasets_path, dataset_name)
    dataset_path = os.path.join(dataset_root, dataset_name + '_dataset')
    dataset_doc_path = os.path.join(dataset_path, 'datasetDoc.json')
    return Dataset.load(dataset_uri='file://' + dataset_doc_path)


def load_problem(dataset_name, datasets_path):
    dataset_root = os.path.join(datasets_path, dataset_name)
    problem_path = os.path.join(dataset_root, dataset_name + '_problem')
    problem_doc_path = os.path.join(problem_path, 'problemDoc.json')
    return parse_problem_description(problem_doc_path)


def run_pipeline(pipeline, dataset_name, datasets_path):
    ensure_downloaded(dataset_name, args.datasets_path)

    dataset = load_dataset(dataset_name, datasets_path)
    problem = load_problem(dataset_name, datasets_path)

    # Creating an instance on runtime with pipeline description and problem description.
    runtime = Runtime(
        pipeline=pipeline,
        problem_description=problem,
        context=Context.TESTING
    )

    # Fitting pipeline on input dataset.
    fit_results = runtime.fit(inputs=[dataset])
    fit_results.check_success()

    # Producing results using the fitted pipeline.
    produce_results = runtime.produce(inputs=[dataset])
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
    parser.add_argument('primitive', nargs='?', help="Name of the primitive to evaluate.")

    args = parser.parse_args()

    if not args.primitive:
        parser.print_help()
        sys.exit(1)

    primitive = index.get_primitive(args.primitive)
    pipeline = primitive.get_demo_pipeline()

    dataset = args.dataset or pipeline.dataset
    datasets_path = os.path.abspath(args.datasets_path)

    run_pipeline(pipeline, dataset, datasets_path)
