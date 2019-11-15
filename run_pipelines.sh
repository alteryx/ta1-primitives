VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION

# Run pipeline and score - Single Table
echo "Running single table pipelines"
for cmdfile in $ST_OUTDIR/pipelines/*.sh
do
  echo "Running $cmdfile"
  sh $cmdfile
done

# Run pipeline and score - Multi Table
echo "Running multi table pipelines"
for cmdfile in $MT_OUTDIR/pipelines/*.sh
do
  echo "Running $cmdfile"
  sh $cmdfile
done
