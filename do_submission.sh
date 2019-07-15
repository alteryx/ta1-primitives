VERSION=$(python3 -c "import featuretools_ta1; print(featuretools_ta1.__version__)")

# create branch for this version of featuretools ta1 primitives
cd /primitives
BRANCH=featuretools-primtitives-$VERSION
git checkout -B $BRANCH

# make sure it is up-to-date with upstream master
git remote add upstream https://gitlab.com/datadrivendiscovery/primitives.git
git fetch upstream
git merge upstream/master

# todo, should we copy directly to primitive repo rather than to our repo first?
rm -rf /primitives/v2019.6.7/MIT_FeatureLabs/
cp -r /featuretools_ta1/MIT_FeatureLabs/ /primitives/v2019.6.7/MIT_FeatureLabs/


# add new files, commit, and push
git add /primitives/v2019.6.7/MIT_FeatureLabs/
git commit -m 'update pipelines and primitive annotations'
git push origin $BRANCH
