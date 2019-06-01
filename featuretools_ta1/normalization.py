from collections import defaultdict


def check_min_categorical_nunique(values, min_categorical_nunique):
    nunique = values.nunique()
    if min_categorical_nunique < 1:
        total = values.shape[0]
        nunique = nunique / total

    return nunique > min_categorical_nunique


def normalize_categoricals(es, base_entity, entities_to_normalize=None,
                           ignore_columns=None,
                           find_equivalent_categories=True,
                           min_categorical_nunique=.1):

    if entities_to_normalize is not None:
        for norm_info in entities_to_normalize:
            es.normalize_entity(
                base_entity_id=norm_info['base_entity_id'],
                new_entity_id=norm_info['new_entity_id'],
                index=norm_info['index'],
                additional_variables=norm_info.get('additional_variables', None),
                make_time_index=norm_info.get('make_time_index', None),
            )

        return entities_to_normalize

    category_vars = []
    other_vars = []
    if ignore_columns is None:
        ignore_columns = []

    for v in es[base_entity].variables:
        if v.name in ignore_columns:
            continue

        if v._dtype_repr == 'categorical':
            values = es[base_entity].df[v.name]
            if check_min_categorical_nunique(values, min_categorical_nunique):
                category_vars.append(v.id)

        elif v._dtype_repr != 'index':
            other_vars.append(v.id)

    additional_vars = defaultdict(list)
    used = set()
    base_df = es[base_entity].df

    if find_equivalent_categories:
        category_var_matches = _find_equivalent_categories(base_df, category_vars)
        for category_var in category_var_matches:
            for other_var in other_vars:
                if other_var in used:
                    continue

                if (base_df.groupby(category_var)[other_var].nunique() == 1).all():
                    used.add(other_var)
                    additional_vars[category_var].append(other_var)

            additional_vars[category_var].extend(category_var_matches[category_var])

    else:
        category_var_matches = category_vars

    entities_normalized = []
    for category_var in category_var_matches:
        _add_vars = additional_vars.get(category_var, None)
        new_entity_id = category_var + "_entity"
        es.normalize_entity(
            base_entity_id=base_entity,
            new_entity_id=new_entity_id,
            index=category_var,
            additional_variables=_add_vars,
            # convert_links_to_integers=True
        )
        entities_normalized.append({
            'base_entity_id': base_entity,
            'new_entity_id': new_entity_id,
            'index': category_var,
            'additional_variables': _add_vars
        })

    return entities_normalized


def _find_equivalent_categories(df, cat_vars):
    matches = {c: [] for c in cat_vars}
    nuniques = {}
    used = set()
    for c in cat_vars:
        for d in cat_vars:
            if (c != d) and (d not in matches) and (d not in used):
                uniques_pair = []
                for tple in [(c, d), (d, c)]:
                    all_nunique = nuniques.get(tple, None)
                    if all_nunique is None:
                        nunique = df.groupby(tple[0])[tple[1]].nunique()
                        all_nunique = (nunique == 1).all()
                        nuniques[tple] = all_nunique

                    uniques_pair.append(all_nunique)

                if all(uniques_pair):
                    matches[c].append(d)
                    used.add(d)

    return matches
