VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION

rm -rf $ST_OUTDIR
mkdir -p $ST_OUTDIR
mkdir $ST_OUTDIR/pipelines

rm -rf $MT_OUTDIR
mkdir -p $MT_OUTDIR
mkdir $MT_OUTDIR/pipelines

# Generate pipelines
for file in /pipeline_tests/*.py
do
  python3 "$file"
done
