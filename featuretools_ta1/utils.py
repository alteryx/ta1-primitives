import cloudpickle
import os
import featuretools as ft


def serialize_features(features):
    ft._pickling = True
    try:
        serialized = save_obj_pickle(features)
    except Exception:
        ft._pickling = False
        raise
    ft._pickling = False
    return serialized


def load_features(filepath, entityset):
    ft._pickling = True
    ft._current_es = entityset

    try:
        features = load_pickle(filepath)
    except Exception:
        ft._current_es = None
        ft._pickling = False
        raise
    ft._current_es = None
    ft._pickling = False
    return features


def save_obj_pickle(obj, filepath=None):
    if filepath is not None:
        if os.path.isfile(filepath):
            with open(filepath, "wb") as out:
                cloudpickle.dump(obj, out)
        else:
            # assume file-like buffer
            cloudpickle.dump(obj, filepath)
    else:
        return cloudpickle.dumps(obj)


def load_pickle(filepath):
    try:
        is_file = os.path.isfile(filepath)
    except ValueError:
        # assume serialized string
        return cloudpickle.loads(filepath)
    else:
        if is_file:
            filestream = open(filepath, "rb")
            return cloudpickle.loads(filepath)
        else:
            return cloudpickle.load(filestream)
