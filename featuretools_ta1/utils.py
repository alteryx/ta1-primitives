import featuretools_ta1.semantic_types as st
from d3m import container
from d3m.metadata import base as metadata_base
import featuretools as ft


def drop_percent_null(fm, features, max_percent_null=.50, verbose=False):
    percents = fm.isnull().sum() / fm.shape[0]
    to_drop = percents[percents > max_percent_null].index
    features = [f for f in features if f.get_name() not in to_drop]
    fm = fm[[f.get_name() for f in features]]

    if verbose:
        print("Dropped: %d features" % (len(to_drop)))
        print("Remaining: %d features" % (len(fm.columns)))

    return fm, features


def select_one_of_correlated(fm, features, threshold=.9, verbose=False):
    if verbose:
        print("Dropping correlated features with threshold: %f" % threshold)

    corr = fm.corr().abs()
    cols = corr.columns
    to_drop = set()

    for c in cols:
        if c in to_drop:
            continue
        drop = corr[corr[c] > threshold].index
        for d in drop:
            if d != c:
                to_drop.add(d)
    features = [f for f in features if f.get_name() not in to_drop]
    fm = fm[[f.get_name() for f in features]]

    if verbose:
        print("Dropped: %d features" % (len(to_drop)))
        print("Remaining: %d features" % (len(fm.columns)))

    return fm, features


def add_metadata(fm, features):
    """takes in a pandas dataframe and a list of featuretools feature
    defintions and returns a d3m dataframe with proper metadata"""
    outputs = container.DataFrame(fm, generate_metadata=True)
    metadata = outputs.metadata

    for i, col in enumerate(outputs.columns):
        metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, i), st.ATTRIBUTE)

        d3m_type = st.ft_to_d3m.get(features[i].variable_type, None)
        if d3m_type:
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, i), d3m_type)

    outputs.metadata = metadata

    return outputs


def find_primary_key(resource_df, return_index=False):
    num_columns = resource_df.metadata.query(["ALL_ELEMENTS"])["dimension"]["length"]

    # find primary key and other variable types
    for i in range(num_columns):
        metadata = resource_df.metadata.query(["ALL_ELEMENTS", i])
        semantic_types = metadata["semantic_types"]
        col_name = metadata["name"]
        if st.PRIMARY_KEY in semantic_types:
            if return_index:
                return i
            return col_name

    return None


def find_target_column(resource_df, return_index=False):
    # todo: refactor this since it share some logic with other functions
    num_columns = resource_df.metadata.query(["ALL_ELEMENTS"])["dimension"]["length"]

    col_list = []
    for i in range(num_columns):
        metadata = resource_df.metadata.query(["ALL_ELEMENTS", i])
        semantic_types = metadata["semantic_types"]
        col_name = metadata["name"]
        if st.TARGET in semantic_types or st.TRUE_TARGET in semantic_types:
            if return_index:
                col_list.append(i)
            else:
                col_list.append(col_name)

    if col_list:
        return col_list

    return None


def get_featuretools_variable_types(resource_df):
    num_columns = resource_df.metadata.query(["ALL_ELEMENTS"])["dimension"]["length"]

    variable_types = {}

    # find primary key and other variable types
    for i in range(num_columns):
        metadata = resource_df.metadata.query(["ALL_ELEMENTS", i])
        semantic_types = metadata["semantic_types"]
        col_name = metadata["name"]
        if st.TEXT in semantic_types:
            variable_types[col_name] = ft.variable_types.Text
        elif st.NUMBER in semantic_types or st.FLOAT in semantic_types or st.INTEGER in semantic_types:
            variable_types[col_name] = ft.variable_types.Numeric
        elif st.DATETIME in semantic_types:
            variable_types[col_name] = ft.variable_types.Datetime
        elif st.BOOLEAN in semantic_types:
            variable_types[col_name] = ft.variable_types.Boolean
        elif st.CATEGORICAL in semantic_types:
            variable_types[col_name] = ft.variable_types.Categorical

    return variable_types
