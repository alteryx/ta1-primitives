import os
import json
import d3m.primitives.featuretools_ta1 as fta1

primitive_names = [p for p in dir(fta1) if not p.startswith('__')]
for pname in primitive_names:
    primitive = getattr(fta1, pname)
    metadata = primitive.metadata.to_json()
    version = metadata['version']
    filepath = '../primitives_repo/v2018.4.18/MIT_FeatureLabs/d3m.primitives.featuretools_ta1.{}/{}'.format(pname, version)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = os.path.join(filepath, 'primitive.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)
