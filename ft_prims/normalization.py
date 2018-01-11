from collections import defaultdict


def normalize_categoricals(es, base_entity, entities_to_normalize=None):
    if entities_to_normalize is not None:
        for norm_info in entities_to_normalize:
            es.normalize_entity(
                    base_entity_id=norm_info['base_entity_id'],
                    new_entity_id=norm_info['new_entity_id'],
                    index=norm_info['index'],
                    additional_variables=norm_info['additional_variables'],
                    make_time_index=norm_info['make_time_index'],
                    convert_links_to_integers=True)
        return entities_to_normalize
    category_vars = []
    other_vars = []
    for v in es[base_entity].variables:
        if v.dtype == 'categorical':
            category_vars.append(v.id)
        elif v.dtype != 'index':
            other_vars.append(v.id)
    additional_vars = defaultdict(list)
    used = set()
    base_df = es[base_entity].df

    category_var_matches = find_equivalent_categories(base_df, category_vars)
    for category_var in category_var_matches:
        for other_var in other_vars:
            if other_var in used:
                continue
            if (base_df.groupby(category_var)[other_var].nunique() == 1).all():
                used.add(other_var)
                additional_vars[category_var].append(other_var)
            additional_vars[category_var].extend(
                    category_var_matches[category_var])

    entities_normalized = []
    for category_var in category_var_matches:
        _add_vars = additional_vars.get(category_var, None)
        new_entity_id = category_var + "_entity"
        es.normalize_entity(base_entity_id=base_entity,
                            new_entity_id=new_entity_id,
                            index=category_var,
                            additional_variables=_add_vars,
                            convert_links_to_integers=True)
        entities_normalized.append({'base_entity_id': base_entity,
                                    'new_entity_id': new_entity_id,
                                    'index': category_var,
                                    'additional_variables': _add_vars})
    return entities_normalized


def find_equivalent_categories(df, cat_vars):
    matches = defaultdict(list)
    for c in cat_vars:
        for d in cat_vars:
            if c != d and d not in matches:
                if ((df.groupby(c)[d].nunique() == 1).all() and
                        (df.groupby(d)[c].nunique() == 1).all()):
                    matches[c].append(d)
    return matches
