import os
import json
import pandas as pd
import warnings
import featuretools as ft
from featuretools.variable_types import (Boolean, Numeric,
                                         Categorical, Text, Datetime)
from .normalization import normalize_categoricals


DATASET_SCHEMA_VERSION = '3.0'
PROBLEM_SCHEMA_VERSION = '3.0'
D3M_TYPES = {
    'boolean': Boolean,
    'integer': Numeric,
    'real': Numeric,
    'categorical': Categorical,
    'dateTime': Datetime,
    'string': Categorical, # Text
}


def convert_d3m_dataset_to_entityset(d3m_ds, entities_to_normalize=None,
                                     normalize_categoricals_if_single_table=True):
    uri = d3m_ds.metadata.query(())['location_uris'][0]
    uri = uri.replace('file://', '')
    with open(uri) as f:
        ds_doc = json.load(f)
    resource_ids = [r['resID'] for r in ds_doc['dataResources']]
    table_arrays = {r: d3m_ds[r] for r in resource_ids}
    return load_d3m_dataset_as_entityset(
            uri, table_arrays=table_arrays,
            entities_to_normalize=entities_to_normalize,
            normalize_categoricals_if_single_table=normalize_categoricals_if_single_table)


def load_d3m_dataset_as_entityset(ds_doc_path,
                                  table_arrays=None,
                                  entities_to_normalize=None,
                                  normalize_categoricals_if_single_table=True,
                                  nrows=None):
    if ds_doc_path.startswith('file://'):
        ds_doc_path = ds_doc_path.split('file://')[1]
    ds_root = os.path.dirname(ds_doc_path)
    ds_name = os.path.basename(ds_root)
    with open(ds_doc_path, 'r') as f:
        ds_doc = json.load(f)

    # make sure the versions line up
    if ds_doc['about']['datasetSchemaVersion'] != DATASET_SCHEMA_VERSION:
        warnings.warn(
            "the datasetSchemaVersions in the API and datasetDoc do not match !")

    # locate the special learningData file
    learning_data_res_id, tables = get_tables_by_res_id(
        ds_doc, ds_root, table_arrays, nrows=nrows)
    remove_privileged_features(ds_doc, tables)

    entityset = ft.EntitySet(ds_name)
    target_entity = None
    for res_id, table_doc in tables.items():
        table_name = table_doc['table_name']
        if res_id == learning_data_res_id:
            target_entity = table_name
        df = table_doc['data']
        columns = table_doc['columns']
        variable_types = convert_d3m_columns_to_variable_types(columns, df)
        index = table_doc.get('index', None)
        make_index = False
        if not index:
            index = table_name + "_id"
            make_index = True
        entityset.entity_from_dataframe(table_name,
                                        df,
                                        index=index,
                                        make_index=make_index,
                                        time_index=table_doc.get('time_index',
                                                                 None),
                                        variable_types=variable_types)

    entities_normalized = None
    if normalize_categoricals_if_single_table and len(tables) == 1:
        entities_normalized = normalize_categoricals(
                entityset,
                table_name,
                entities_to_normalize=None)
    else:
        rels = extract_ft_relationships_from_columns(entityset, tables)
        if len(rels):
            entityset.add_relationships(rels)
    return entityset, target_entity, entities_normalized


def extract_ft_relationships_from_columns(entityset, tables):
    rels = []
    for res_id in tables.keys():
        columns = tables[res_id]['columns']
        df = tables[res_id]['data']
        for c in columns:
            if 'refersTo' in c and c['colName'] in df.columns:
                table_name = tables[res_id]['table_name']
                ft_var = entityset[table_name][c['colName']]
                foreign_res_id = c['refersTo']['resID']
                foreign_table_name = tables[foreign_res_id]['table_name']
                res_obj = c['refersTo']['resObject']
                if isinstance(res_obj, dict):
                    if 'columnIndex' in res_obj:
                        column_index = res_obj['columnIndex']
                        column_name = tables[res_id]['columns'][column_index]['colName']
                    else:
                        column_name = res_obj['columnName']
                    ft_foreign_var = entityset[foreign_table_name][column_name]
                    rels.append(ft.Relationship(ft_foreign_var, ft_var))
    return rels


def convert_d3m_columns_to_variable_types(columns, df):
    variable_types = {}
    for c in columns:
        ctype = c['colType']
        col_name = c['colName']
        if col_name not in df.columns:
            continue
        # TODO: Need to infer text vs. categorical (they only give "string")
        vtype = D3M_TYPES[ctype]
        if vtype == Datetime:
            try:
                df[col_name] = parse_date(df[col_name])
            except:
                vtype = Categorical

        variable_types[col_name] = vtype
    return variable_types


def parse_date(col_name, series):
    if col_name.find('year') > -1:
        try:
            series.astype(int)
        except:
            return pd.to_datetime(series, infer_datetime_format=True)
        else:
            return pd.to_datetime(series, format='%Y')


def remove_privileged_features(ds_doc, tables):
    privileged_features = find_privileged_features(ds_doc, tables)
    for res_id, pcols in privileged_features.items():
        if len(pcols) > 0 and res_id in tables:
            tables[res_id]['data'].drop(pcols, axis=1, inplace=True)


def find_privileged_features(ds_doc, tables):
    privileged_features = {}
    if 'qualities' not in ds_doc:
        return privileged_features
    for qual in ds_doc['qualities']:
        if (qual['qualName'] == 'privilegedFeature' and
                qual['qualValue'] == 'True' and
                'restrictedTo' in qual):
            restricted_to = qual['restrictedTo']
            res = restricted_to['resID']

            if res not in privileged_features:
                privileged_features[res] = []
            res_component = restricted_to.get('resComponent', None)
            if res_component is not None:
                restricted_value = list(res_component.values())[0]
                if 'columnIndex' in res_component:
                    restricted_value = tables[res]['columns'][restricted_value]['columnName']
                elif 'columnName' not in res_component:
                    continue
                privileged_features[res].append(restricted_value)
    return privileged_features


def get_tables_by_res_id(ds_doc, ds_root, table_arrays=None, nrows=None):
    tables = {}
    learning_data_res_id = None
    for res in ds_doc['dataResources']:
        res_path = res['resPath']
        table_name = res_path.split('.')[0]
        table_name = table_name.replace("tables/", "")
        res_type = res['resType']
        res_id = res['resID']

        dirname = os.path.basename(os.path.normpath(os.path.dirname(res_path)))

        if res_type == 'table' and dirname == 'tables':
            if 'learningData.csv' in res['resPath']:
                learning_data_res_id = res_id
            columns = res['columns']
            index_cols = [c['colName'] for c in columns if 'index' in c['role']]
            index_col = None
            if len(index_cols):
                index_col = index_cols[0]

            col_names = [c['colName'] for c in columns]
            if table_arrays:
                array = table_arrays[res_id]
                df = pd.DataFrame(array, columns=col_names)
            else:
                df = pd.read_csv(os.path.join(ds_root, res_path),
                                 nrows=nrows)
            if index_col is not None:
                df.index = df[index_col].values

            time_index_cols = [c['colName'] for c in columns
                               if 'timeIndicator' in c['role']]
            time_index = None
            if len(time_index_cols):
                time_index = time_index_cols[0]

            tables[res_id] = {'table_name': table_name,
                              'columns': columns,
                              'data': df,
                              'index': index_col,
                              'time_index': time_index}
    if learning_data_res_id is None:
        raise RuntimeError('could not find learningData resource')
    return learning_data_res_id, tables
