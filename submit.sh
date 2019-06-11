
VERSION=0.6.0
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION


rm -rf $ST_OUTDIR
mkdir -p $ST_OUTDIR
mkdir $ST_OUTDIR/pipelines

rm -rf $MT_OUTDIR
mkdir -p $MT_OUTDIR
mkdir $MT_OUTDIR/pipelines


python3 multitable_test.py
python3 singletable_test.py

python3 -m d3m.index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization > $ST_OUTDIR/primitive.json
python3 -m d3m.index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization > $MT_OUTDIR/primitive.json


# todo, we should copy directly to primitive repo rather than to our repo first
rm -rf /primitives/v2019.6.7/MIT_FeatureLabs/
cp -r /featuretools_ta1/MIT_FeatureLabs/ /primitives/v2019.6.7/MIT_FeatureLabs/


