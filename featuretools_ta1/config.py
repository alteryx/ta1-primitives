from d3m import utils as d3m_utils
from d3m.metadata import base as metadata_base
import os
from pathlib import Path


AUTHOR = "MIT_FeatureLabs"
CONTACT = "mailto:max.kanter@featurelabs.com"


_git_commit = d3m_utils.current_git_commit(Path(__file__).parents[1])
INSTALLATION = [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1'.format(
                   git_commit=_git_commit,
               ),
            }]


