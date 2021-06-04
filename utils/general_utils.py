import hashlib
import json
import types
from types import SimpleNamespace

def make_meta_dict(meta_object):
        meta_dict = {}
        for field in meta_object.fields:
            meta_dict[field] = getattr(meta_object, field)
        return meta_dict


def hash_string_to_int(s:str, length:int = 8) -> int:
    """
    hashes a string using sha256 and returns the interger components 
    I might need to salt the hases to ensure there's no duplication of random seed 
    .. but this is all over kill for now 
    """
    h = int(hashlib.sha256(s.encode('utf-8')).hexdigest(),16) % 10**length

    return h

def dict_to_sns(d:dict) -> types.SimpleNamespace:
    """
    creates a SimpleNamespace object from a dictonary. Nested dictonaries not handled brillaintly

    sns.foo['bar']

    Args:
        d (dict): dictonary which can be nested to any degree 

    Returns:
        types.SimpleNamespace: [description]
    """
    return SimpleNamespace(**d)


def create_namespace_from_dict(d:dict) -> types.SimpleNamespace:
    """
    Creates a SimpleNamespace object from a dictionary, handles nested dictonaires nicely. 
    objects will be accessbile as sns.foo.bar 

    Args:
        dictionary (dict): [description]

    Returns:
        types.SimpleNamespace: [description]
    """

    # convert the dictionary to a string 
    json_object = json.dumps(d)

    # load the string with json parser 
    sns = json.loads(json_object, object_hook=dict_to_sns)

    return sns
