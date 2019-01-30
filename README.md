# MIT-Featuretools TA1 Primitives

## Install

In order to install the project to use primitives use the command:

```
make install
```

In order to install the project for development and be able to generate the JSON
annotations and run the demo pipelines use the command:

```
make install-develop
```

## How to submit a new primtive version

1. Make the necessary changes in the code.
2. Commit the changes and push them to the public repository.
3. Execute the demo pipelines on the datasets using the `sripts/run_pipeline.py` script.
4. Execute the command `make describe` to generate the new annotations. **Important**: This has
  to be executed AFTER the code chanes have been committed. Otherwise, the generated annotations
  will NOT be valid.
5. Validate the generated annotations using the d3m commands.
5. Add the contents from the MIT_FeatureLabs folder to the primitives_repo
6. Optionally commit and push the generated annotations.
