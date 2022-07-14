This is the implementation code for reproducing the experiment results.


'cd ./ntl/'
To run the NTL code, you need to download the corresponding datasets and place them in a file './data'. After downloading the data, you need to modify the path in 'utils_digit.py'. Then you can run the NTL on digits via the command 'python3 ntl_digit.py'. The experiments on other datasets are similar.


'cd ./augmentation/'
To run the augmentation code, you need to download the corresponding datasets similar to the NTL code. Then you also need to modify the data path in 'return_dataset.py'. Then you can run the augmentation on, for example, cifar10 via the command 'python3 gan_aug_cifar_sm.py'. After the augmentation, you can run source-only NTL in './ntl/'.