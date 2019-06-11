python3 multitable_test.py
python3 singletable_test.py

python3 -m d3m.index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization > /featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/0.5.0/primitive.json

python3 -m d3m.index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization > /featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/0.5.0/primitive.json


rm -rf /primitives/v2019.6.7/MIT_FeatureLabs/ && cp -r /featuretools_ta1/MIT_FeatureLabs/ /primitives/v2019.6.7/MIT_FeatureLabs/
