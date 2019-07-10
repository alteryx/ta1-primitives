VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")
ST_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/$VERSION
MT_OUTDIR=/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization/$VERSION

# Run pipeline and score - Single Table
echo "Running single table pipelines"
for ymlfile in $ST_OUTDIR/pipelines/*.yml
do
  echo "Running $ymlfile"
  metafile="${ymlfile%.*}.meta"
  echo "FIT-SCORE"
  python3 -m d3m runtime -d /featuretools_ta1/datasets/ fit-score -m $metafile -p $ymlfile
  echo "EVALUATE"
  python3 -m d3m runtime -d /featuretools_ta1/datasets/ evaluate -m $metafile -p $ymlfile -d /pipeline_tests/kfold_pipeline.yml
done

# Run pipeline and score - Multi Table
echo "Running multi table pipelines"
for ymlfile in $MT_OUTDIR/pipelines/*.yml
do
  echo "Running $ymlfile"
  metafile="${ymlfile%.*}.meta"
  echo "FIT-SCORE"
  python3 -m d3m runtime -d /featuretools_ta1/datasets/ fit-score -m $metafile -p $ymlfile
  echo "EVALUATE"
  python3 -m d3m runtime -d /featuretools_ta1/datasets/ evaluate -m $metafile -p $ymlfile -d /pipeline_tests/kfold_pipeline.yml
done
