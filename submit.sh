
VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION

rm -rf $ST_OUTDIR
mkdir -p $ST_OUTDIR
mkdir $ST_OUTDIR/pipelines

rm -rf $MT_OUTDIR
mkdir -p $MT_OUTDIR
mkdir $MT_OUTDIR/pipelines

# note: the version is hardcoded in these files, so it needs to be updated
# python3 multitable_test.py
# python3 singletable_test.py

for file in /pipeline_tests/*.py
do
  python3 "$file"
done

python3 -m d3m.index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization > $ST_OUTDIR/primitive.json
python3 -m d3m.index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization > $MT_OUTDIR/primitive.json


# todo, we should copy directly to primitive repo rather than to our repo first
rm -rf /primitives/v2019.6.7/MIT_FeatureLabs/
cp -r /featuretools_ta1/MIT_FeatureLabs/ /primitives/v2019.6.7/MIT_FeatureLabs/


