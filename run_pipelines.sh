VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION

# Run pipeline and score - Single Table
echo "Running single table pipelines"
echo "Unzipping files..."
for file in $ST_OUTDIR/pipeline_runs/*.gz
do
  gunzip $file
done

for file in $ST_OUTDIR/pipeline_runs/*.yml
do
  echo "Running $file"
  python3 -m d3m --pipelines-path $ST_OUTDIR/pipelines/ runtime -d /featuretools_ta1/datasets/ fit-score -u $file
  gzip $file
done

# Run pipeline and score - Multi Table
echo "Running multi table pipelines"
echo "Unzipping files..."
for file in $MT_OUTDIR/pipeline_runs/*.gz
do
  gunzip $file
done
for file in $MT_OUTDIR/pipeline_runs/*.yml
do
  echo "Running $file"
  python3 -m d3m --pipelines-path $MT_OUTDIR/pipelines/ runtime -d /featuretools_ta1/datasets/ fit-score -u $file
  gzip $file
done
