import os
import json

import d3m.primitives.featuretools_ta1 as ft_ta1


def describe(name):
    print('Describing primitive {}'.format(name))
    primitive = getattr(ft_ta1, name)
    metadata = primitive.metadata.to_json_structure()
    version = metadata['version']
    filepath = 'MIT_FeatureLabs/d3m.primitives.featuretools_ta1.{}/{}'.format(name, version)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = os.path.join(filepath, 'primitive.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':

    for name in [p for p in dir(ft_ta1) if not p.startswith('__')]:
        describe(name)
