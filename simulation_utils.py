import hashlib

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
