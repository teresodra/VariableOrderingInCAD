
purposes = ['Train', 'Test']
dataset_qualities = ['Biased', 'Balanced', 'Augmented']


def aveg(given_list):
    return sum(given_list)/len(given_list)


def aveg_not_zero(given_list):
    return sum(given_list)/max(1, len([1 for elem in given_list
                                      if elem != 0]))


operations = [sum, max, aveg]  # , aveg_not_zero
