import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import featuretools.variable_types as vtypes
from d3m.metadata.base import ALL_ELEMENTS, DataMetadata
from d3m.container.pandas import DataFrame


class D3MMetadataTypes(object):
    Table = 'https://metadata.datadrivendiscovery.org/types/Table'
    EntryPoint = 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
    PrimaryKey = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
    UniqueKey = 'https://metadata.datadrivendiscovery.org/types/UniqueKey'
    Categorical = "https://metadata.datadrivendiscovery.org/types/CategoricalData"
    Ordinal = "https://metadata.datadrivendiscovery.org/types/OrdinalData"
    Datetime = "https://metadata.datadrivendiscovery.org/types/Time"
    # TODO: why not a separate one?
    TimeIndicator = "https://metadata.datadrivendiscovery.org/types/Time"
    Boolean = "http://schema.org/Boolean"
    Float = "http://schema.org/Float"
    Integer = "http://schema.org/Integer"
    Text = "http://schema.org/Text"
    Privileged = 'https://metadata.datadrivendiscovery.org/types/PrivilegedData'
    Timeseries = 'https://metadata.datadrivendiscovery.org/types/Timeseries'
    Target = 'https://metadata.datadrivendiscovery.org/types/Target'
    SuggestedTarget = 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
    TrueTarget = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
    Attribute = "https://metadata.datadrivendiscovery.org/types/Attribute"

    KeyTypes = (PrimaryKey,
                UniqueKey)

    FTMapping = {
        Boolean: vtypes.Boolean,
        PrimaryKey: vtypes.Index,
        UniqueKey: vtypes.Id,
        Categorical: vtypes.Categorical,
        Datetime: vtypes.Datetime,
        Float: vtypes.Numeric,
        Integer: vtypes.Numeric,
        Text: vtypes.Text,
        Ordinal: vtypes.Ordinal
    }

    D3MMapping = {
        vtypes.Discrete: Categorical,
        vtypes.Categorical: Categorical,
        vtypes.Ordinal: Ordinal,
        vtypes.Datetime: Datetime,
        vtypes.Index: PrimaryKey,
        vtypes.Id: UniqueKey,
        vtypes.Boolean: Boolean,
        vtypes.Numeric: Float,
        vtypes.Text: Text
    }
    StructuralMapping = {
        vtypes.Discrete: object,
        vtypes.Categorical: object,
        vtypes.Ordinal: str,
        vtypes.Datetime: np.datetime64,
        vtypes.Index: object,
        vtypes.Id: object,
        vtypes.Boolean: bool,
        vtypes.Numeric: float,
        vtypes.Text: str
    }
    ColumnTypes = {
        PrimaryKey, UniqueKey, Categorical, Ordinal,
        Datetime, TimeIndicator, Boolean, Float,
        Integer, Text
    }

    @classmethod
    def is_column_type(cls, d3m_type):
        if d3m_type in cls.ColumnTypes:
            return True

    @classmethod
    def to_ft(cls, d3m_type):
        return cls.FTMapping[d3m_type]

    @classmethod
    def to_d3m(cls, ft_type):
        return cls.D3MMapping[ft_type]

    @classmethod
    def to_default_structural_type(cls, ft_type):
        return cls.StructuralMapping[ft_type]


def convert_variable_type(df, col_name, vtype, target_colname):
    if col_name == target_colname and vtype == vtypes.Boolean:
        vtype = vtypes.Categorical
    elif vtype == vtypes.Boolean:
        if df[col_name].dtype != bool:
            vals = df[col_name].replace(
                r'\s+', np.nan, regex=True).dropna().unique()
            map_dict = infer_true_false_vals(*vals)
            df[col_name] = df[col_name].replace(
                r'\s+', np.nan, regex=True).map(map_dict, na_action='ignore')
            # .astype(bool) converts nan values to True. We want to
            # keep them as nan so we can't cast to bool if there are nans
            if df[col_name].dropna().shape[0] == df.shape[0]:
                df[col_name] = df[col_name].astype(bool)
    elif vtype in (vtypes.Datetime, vtypes.DatetimeTimeIndex):
        try:
            df[col_name] = parse_date(col_name, df[col_name])
        except:
            vtype = vtypes.Categorical
        else:
            if is_numeric_dtype(df[col_name].dtype):
                vtype = vtypes.Numeric
    # # TODO: look structural string type or Metadata Text type?
    # elif ctype == SEMANTIC_TYPES['text']:
        # is_text = infer_text_column(df[col_name])
        # if is_text:
            # vtype = Text
        # else:
            # vtype = Categorical
    return vtype


def infer_true_false_vals(*vals):
    positive_indicators = ['y', 'p', 't']
    true_vals = {v: True for v in vals for p in positive_indicators
                 if isinstance(v, str) and v.lower().find(p) > -1}
    negative_indicators = ['n', 'f']
    false_vals = {v: False for v in vals for n in negative_indicators
                  if isinstance(v, str) and v.lower().find(n) > -1}
    map_dict = true_vals
    map_dict.update(false_vals)
    for v in vals:
        if not isinstance(v, bool) and v not in map_dict:
            map_dict[v] = False
    return map_dict


def infer_text_column(series):
    if series.dtype != str:
        return False
    # heuristics to predict this some other than categorical
    sample = series.sample(min(10000, series.nunique()))
    avg_length = sample.str.len().mean()
    std_length = sample.str.len().std()
    num_spaces = series.str.count(' ').mean()

    if num_spaces > 0:
        spaces_freq = avg_length / num_spaces
        # repeated spaces
        if spaces_freq < (avg_length / 2):
            return True
    if avg_length > 50 and std_length > 10:
        return True


def parse_date(col_name, series):
    numeric_strings = ['year', 'hour', 'minute',
                       'second', 'sec', 'day']
    if any(col_name.lower().find(s) > -1
           for s in numeric_strings):
        try:
            return series.astype(int)
        except:
            try:
                return pd.to_numeric(series, errors='coerce')
            except:
                return pd.to_datetime(series, infer_datetime_format=True)
    else:
        return pd.to_datetime(series, infer_datetime_format=True)


def load_timeseries_as_df(ds, res_id):
    df = ds[res_id]
    ncolumns = ds.metadata.query((res_id, ALL_ELEMENTS))['dimension']['length']
    # TODO: handle additional columns?
    for icol in range(ncolumns):
        metadata = ds.metadata.query((res_id, ALL_ELEMENTS, icol))
        if D3MMetadataTypes.Filename in metadata['semantic_types']:
            root = metadata['location_base_uris'][0]
            colname = metadata['name']
            break
    dfs = []
    for d3m_index, row in df.iterrows():
        filename = row[colname]
        df = pd.read_csv(os.path.join(root, filename))
        df['d3mIndex'] = d3m_index
        dfs.append(df)
    full_df = pd.concat(dfs)
    index_col = "timeseries_index"
    full_df.reset_index(inplace=True, drop=True)
    full_df.index.name = index_col
    full_df.reset_index(inplace=True, drop=False)
    time_index = None
    if 'time' in full_df.columns:
        time_index = 'time'

    return full_df, index_col, time_index


def get_target_columns(metadata: DataMetadata):
    target_columns = []
    is_dataframe = metadata.query(())['structural_type'] == DataFrame
    if not is_dataframe:
        n_resources = metadata.query(())['dimension']['length']
        resource_to_use = n_resources - 1
        if n_resources > 1:
            # find learning data resource
            resource_to_use = [res_id for res_id in range(n_resources)
                               if D3MMetadataTypes.EntryPoint in metadata.query(
                                       (str(res_id), ))['semantic_types']][0]
        ncolumns = metadata.query((str(resource_to_use), ALL_ELEMENTS))['dimension']['length']
    else:
        ncolumns = metadata.query((ALL_ELEMENTS,))['dimension']['length']

    for column_index in range(ncolumns):
        if is_dataframe:
            column_metadata = metadata.query((ALL_ELEMENTS, column_index))
        else:
            column_metadata = metadata.query((str(resource_to_use), ALL_ELEMENTS,
                                              column_index))
        semantic_types = column_metadata.get('semantic_types', [])
        if D3MMetadataTypes.TrueTarget in semantic_types:
            column_name = column_metadata['name']
            target_columns.append(column_name)
    return target_columns
