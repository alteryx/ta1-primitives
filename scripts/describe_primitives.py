import json
import os
import shutil

import featuretools_ta1


def write(path, filename, contents):
    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)
    print('Writing file {}'.format(full_path))
    with open(full_path, 'w') as json_file:
        json.dump(contents, json_file, indent=4)


def describe(metadata):
    python_path = metadata['python_path']
    print('Describing primitive {}'.format(python_path))

    version = metadata['version']
    path = os.path.join('MIT_FeatureLabs', python_path, version)
    if os.path.exists(path):
        shutil.rmtree(path)

    write(path, 'primitive.json', metadata)

    pipeline = primitive.get_demo_pipeline()
    pipeline_path = os.path.join(path, 'pipelines')
    pipeline_name = pipeline.id + '.json'

    pipeline_dict = pipeline.to_json_structure()
    for step in pipeline_dict['steps']:
        del step['primitive']['digest']

    write(pipeline_path, pipeline_name, pipeline_dict)

    dataset = pipeline.dataset
    meta = {
        'problem': dataset + '_problem',
        'full_inputs': [dataset + '_dataset'],
        'train_inputs': [dataset + '_dataset_TRAIN'],
        'test_inputs': [dataset + '_dataset_TEST'],
        'score_inputs': [dataset + '_dataset_SCORE']
    }
    meta_name = pipeline.id + '.meta'
    write(pipeline_path, meta_name, meta)


if __name__ == '__main__':

    for primitive in featuretools_ta1.PRIMITIVES:
        metadata = primitive.metadata.to_json_structure()
        describe(metadata)
