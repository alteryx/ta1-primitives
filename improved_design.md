Input
* D3M Dataset or Dataframe

HyperParameters
* Agg primitives, trans_primiitves
* max depth
* target entity? not sure if it can be


Fit steps
1. Load into Entity Set
    * map d3m variable types to featuretools
    * optionally normalize entities
    * handle labels - may need to drop to avoid leakage

2. Determine DFS Arguments

3. Run DFS

4. Save features for produce step

5. Format return Dataframe into D3M format


Produce steps
1. reprocess input into Entity Set

2. run calculate feature matrix with saved features

3. format and return Dataframe



Extra stuff to be compliant
* can_accept
* set_training_data
* getstate / setstate / getparams / setparams

Open questions

* is there a clean way to access and update metadata that what we have?
* do we only support single table?
* what is the equivalent of time index in D3M?
