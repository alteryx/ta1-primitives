# Setup instructions for Featuretools TA-1

The following instructions provide an overview of the steps required to setup and run
the code for Featuretools TA-1 locally

1. If you do not already have Docker installed on your system, you will need to install it. Install Docker by following the instructions found at [https://docs.docker.com/install/](https://docs.docker.com/install/). Start Docker after the installation process completes.
2. Clone the [Featuretools TA-1 code repo](https://github.com/Featuretools/ta1-primitives).
3. Clone the [Primitives](https://gitlab.com/kmax12/primitives) repo into the same directory into which you cloned the Featuretools TA-1 repo in Step 2. 
4. Enter your D3M credentials to access the D3M repos.
5. Clone the datasets repo into the root of the Featuretools TA-1 repo that you cloned in step 2. Because the dataset repo is quite large, you should clone the repo without downloading the data files. This process utilizes Git LFS, so if you do not have Git LFS installed, you can install it as directed here: [https://git-lfs.github.com/](https://git-lfs.github.com/).
6. Once Git LFS is installed, change to the root of the Featuretools TA-1 repo from Step 2 and start the data cloning process by executing `git lfs clone https://gitlab.datadrivendiscovery.org/d3m/datasets.git -X "*"`. This process may take some time to complete.
7. After cloing the datasets, change your working directory to the datasets directory and clone the datasets of interest. For running the `singletable_test.py` you will need the `196_autoMpg` dataset. For `multitable_text.py` you will need the `32_wikiqa` dataset. You can clone these datasets with this command: `git lfs pull -I seed_datasets_current/{datasetname}/`, replacing `{datasetname}` with the name of teh dataset you wish to clone. If this process doesn't work you may need to run `git lfs install` and try again.
8. After you have cloned all of the necessary datasets, execute `docker login registry.datadrivendiscovery.org` and enter the login credentials supplied by Max in Step 4.
9. Build the docker container by running `docker build -t d3mft .`.
10. Mount the necessary volumes and launch the docker image in interactive mode with the following command, replacing `{path}` with the full path from your system root to the directory into which you cloned the Featuretools TA-1 repo:
```
docker run \
    -v {path}/ta1-primitives/:/featuretools_ta1/ \
    -v {path}/ta1-primitives/singletable_test.py:/singletable_test.py \
    -v {path}/ta1-primitives/multitable_test.py:/multitable_test.py \
    -v {path}/ta1-primitives/multitable_uu3_wdi_test.py:/multitable_uu3_wdi_test.py \
    -v {path}/primitives/:/primitives \
    -it d3mft
```

11. Run the single table test file with the command `python3 singletable_test.py`. You should see results printed if the file runs successfully. You can also run the multi table test with `python3 multitable_test.py` and it should also print a results table upon successful completion.