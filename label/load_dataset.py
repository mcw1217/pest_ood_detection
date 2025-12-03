def load_dataset(name):
    if name.lower() == 'ninco':
        from datasets.ninco import get_dataset
    elif name.lower() == 'openimageo':
        from datasets.openimageo import get_dataset
    elif name.lower() == 'texture':
        from datasets.texture import get_dataset
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    return get_dataset()
