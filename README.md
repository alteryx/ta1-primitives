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
2. Execute the demo pipelines using `make test`. For more options, like running on different
   datasets, use the `sripts/run_pipeline.py` script directly.
3. Execute `make release`. This will generate and release a new tag, as well as the corresponding
   primitive annotations.
4. Copy the MIT_FeatureLabs folder over to the [primitives repo](https://gitlab.com/datadrivendiscovery/primitives)
5. Validate the generated annotations inside the primitives repo using the d3m command:

```
$ ./run_validation.py 'v{current_d3_version}/MIT_FeatureLabs/{primitive_name}/{current_version}/primitive.json'
```

6. In the remote case that the previous validation failed, you can rollback the release
   with the command `make rollback`. After you have executed it, go back to point 1
   to make any further changes needed.
