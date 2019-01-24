import os
import json

from d3m.primitives import featuretools_ta1


def describe(name):
    print('Describing primitive {}'.format(name))
    primitive = getattr(featuretools_ta1, name)
    metadata = primitive.metadata.to_json_structure()
    version = metadata['version']
    filepath = 'MIT_FeatureLabs/d3m.primitives.featuretools_ta1.{}/{}'.format(name, version)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = os.path.join(filepath, 'primitive.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':

    for name in [p for p in dir(featuretools_ta1) if not p.startswith('__')]:
        describe(name)
