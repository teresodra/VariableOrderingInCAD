import os
# from config.general_values import dataset_qualities
from config.general_values import purposes


def find_hyperparams_filename(model_name, paradigm, training_quality):
    return os.path.join(os.path.dirname(__file__),
                        'config', 'hyperparameters',
                        f'{model_name}-{paradigm}-{training_quality}')


def find_model_filename(model_name, paradigm, training_quality):
    return os.path.join(os.path.dirname(__file__),
                        'config', 'models',
                        f'{model_name}-{paradigm}-{training_quality}.txt')


def find_dataset_filename(purpose, dataset_quality=None, paradigm=''):
    if purpose == "unclean":
        return os.path.join(os.path.dirname(__file__),
                            'datasets', 'before_processing',
                            'dataset_without_repetition_return_ncells_with_subdir.txt')
    # 'dataset_with_repetition_return_ncells.txt')
    # for returning "repeated" instances
    # those with the same number of cells for all projections
    elif purpose == "clean":
        return os.path.join(os.path.dirname(__file__),
                            'datasets',
                            'clean_dataset.txt')
    elif purpose == 'instances':
        return os.path.join(os.path.dirname(__file__),
                            'datasets',
                            'dataset_instances.csv')
    elif purpose in purposes:
        return os.path.join(os.path.dirname(__file__),
                            'datasets', f'{purpose}',
                            f'{dataset_quality}-{purpose}-{paradigm}-dataset.txt')
    else:
        raise Exception(f"Purpose {purpose} not found")


def find_output_filename(training_method):
    return os.path.join(os.path.dirname(__file__), 'results',
                        f'ml_trained_in_{training_method}.csv')


def find_other_filename(search):
    return os.path.join(os.path.dirname(__file__), 'config',
                        f'{search}.txt')


def find_all_info(model_name, paradigm, training_quality):
    return os.path.join(os.path.dirname(__file__), 'results',
                        f'{model_name}-{paradigm}-{training_quality}.txt')
