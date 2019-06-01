from d3m import utils as d3m_utils
from d3m.metadata import base as metadata_base
import os

AUTHOR = "MIT_FeatureLabs"
CONTACT = "mailto:max.kanter@featurelabs.com"

try:
    _git_commit = d3m_utils.current_git_commit("/ta1-primitives")

except:

    _git_commit = d3m_utils.current_git_commit(d3m_utils.current_git_commit(os.path.dirname(__file__)))

INSTALLATION = [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1'.format(
                   git_commit=_git_commit,
               ),
            }]


