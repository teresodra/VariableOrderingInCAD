import yaml


def write_yaml_to_file(py_obj, filename):
    with open(f'{filename}.yaml', 'w',) as f:
        yaml.dump(py_obj, f, sort_keys=False)
    print('Written to file successfully')


def read_yaml_from_file(filename):
    with open(f'{filename}.yaml') as f:
        # py_obj = yaml.safe_load(f)
        py_obj = yaml.load(f, Loader=yaml.Loader)
    print('Read from file successfully')
    return py_obj
