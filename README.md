# Effects of data balancing and augmention on accuracy

## Author: Tereso del Río

## Date: October 2022 

 This repository contains proof of the positive impact that data balancing and augmentation can have on accuracy.

- It also contains an installable package called dataset_manipulation that can balance and augment polynomial data. It is not necessary to install it.

- Running the module main.py, a file called ml_results.csv (also included in the repository) will be generated showing a comparison between a variety of models trained in data without manipulation, in balanced data and in augmented data.
    - For some models, there is not a big difference, but keep in mind that these are the models that worked as well as random (accuracy close to 0.167) in the first place using the hyperparameters found by Dorian in [1].
    - However, there is an amazing improvement in random forest and k-nearest-neighbours, where accuracies have an increment of up to 50% when data is augmented.

[1] Florescu, D., England, M. (2020). A Machine Learning Based Software Pipeline to Pick the Variable Ordering for Algorithms with Polynomial Inputs. Bigatti, A., Carette, J., Davenport, J., Joswig, M., de Wolff, T. (eds) Mathematical Software, ICMS 2020. ICMS 2020. Lecture Notes in Computer Science, vol 12097. Springer, Cham. https://doi.org/10.1007/978-3-030-52200-1_30