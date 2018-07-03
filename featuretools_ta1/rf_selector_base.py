from d3m.metadata import base as metadata_module
from d3m import utils
import os
from . import __version__

METADATA = {
     "source": {
       "name": "MIT_FeatureLabs",
       "contact": "mailto://ben.schreck@featurelabs.com",
       "uris": ["https://doc.featuretools.com"],
       "license": "BSD-3-Clause"

     },
     "keywords": ["feature selection"],
     'version': __version__,
     'installation': [
        {
            "type": metadata_module.PrimitiveInstallationType.PIP,
            "package_uri": "git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@01b32d6eafe972207c877dfbd9a7b106c1920072#egg=sklearn_wrap"
        },
        {'type': metadata_module.PrimitiveInstallationType.PIP,
                       'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1-{version}'.format(
                           git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                           version=__version__
                        ),
        }]
}

__author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'
