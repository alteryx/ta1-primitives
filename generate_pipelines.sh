VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION

rm -rf $ST_OUTDIR
mkdir -p $ST_OUTDIR
mkdir $ST_OUTDIR/pipelines
mkdir $ST_OUTDIR/pipeline_runs

rm -rf $MT_OUTDIR
mkdir -p $MT_OUTDIR
mkdir $MT_OUTDIR/pipelines
mkdir $MT_OUTDIR/pipeline_runs

# Generate pipelines
for file in /pipeline_tests/test_*.py
do
  filename="${file##*/}"
  filebase="${filename%.*}"
  python3 -c "from pipeline_tests.$filebase import generate_only; generate_only()"
done

python3 -m d3m index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization > $ST_OUTDIR/primitive.json
python3 -m d3m index describe -i 4 d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization > $MT_OUTDIR/primitive.json
