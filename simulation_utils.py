

def make_meta_dict(meta_object):
        meta_dict = {}
        for field in meta_object.fields:
            meta_dict[field] = getattr(meta_object, field)
        return meta_dict