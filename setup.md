# Setup instructions for Featuretools TA-1


## Developing Locally

The following instructions provide an overview of the steps required to setup and run
the code for Featuretools TA-1 locally

1. If you do not already have Docker installed on your system, you will need to install it. Install Docker by following the instructions found at [https://docs.docker.com/install/](https://docs.docker.com/install/). Start Docker after the installation process completes.
2. Clone the [Featuretools TA-1 code repo](https://github.com/Featuretools/ta1-primitives).
3. Clone the [Primitives](https://gitlab.com/kmax12/primitives) repo into the same directory as you cloned the Featuretools TA-1 repo in Step 2.
4. Enter your D3M credentials to access the D3M repos.
5. Clone the datasets repo into the root of the Featuretools TA-1 repo from Step 2. Because the dataset repo is quite large, you should clone the repo without downloading the data files. This process utilizes Git LFS, so if you do not have Git LFS installed, you can install it as directed here: [https://git-lfs.github.com/](https://git-lfs.github.com/).
6. Once Git LFS is installed, change to the root of the Featuretools TA-1 repo from Step 2 and start the data cloning process by executing `git lfs clone https://gitlab.datadrivendiscovery.org/d3m/datasets.git -X "*"`. This process may take some time to complete.
7. After cloning the datasets, change your working directory to the datasets directory and download the datasets of interest. For running the `singletable_test.py` you will need the `196_autoMpg` dataset. For `multitable_text.py` you will need the `32_wikiqa` dataset. You can download these datasets with this command: `git lfs pull -I seed_datasets_current/{datasetname}/`, replacing `{datasetname}` with the name of the dataset you wish to clone. If this process doesn't work you may need to run `git lfs install` and try again.
8. After you have downloaded all of the necessary datasets, execute `docker login registry.datadrivendiscovery.org` and enter the login credentials obtained in Step 4.
9. Build the docker container by running `docker build -t d3mft .`.
10. Mount the necessary volumes and launch the docker image in interactive mode with the following command, replacing `{path}` with the full path from your system root to the directory into which you cloned the Featuretools TA-1 repo:

```
docker run \
    -v {path}/ta1-primitives/:/featuretools_ta1/ \
    -v {path}/ta1-primitives/pipeline_tests/:/pipeline_tests/ \
    -v {path}/primitives/:/primitives \
    -v ~/.gitconfig:/etc/gitconfig \
    -it d3mft
```

11. Run any of the test files in the `pipeline_tests` directory with the command `python3 pipeline_tests/test_filename.py`. If the test runs successfully, you should see results printed to the console.


## Submitting

If you have made changes to the primitive implementation, you must push those to the GitHub repo before running the below commands

1. Run `sh featuretools_ta1/make_submission.sh` to generate files for submission.

2. Run `sh featuretools_ta1/do_submission.sh` to create branch on gitlab and push files

3. Create MR based off branch from 2 [here](https://gitlab.com/datadrivendiscovery/primitives/merge_requests)

4. After it gets merged in trigger a run of the pipelines [here](https://dash.datadrivendiscovery.org/pipelines)
