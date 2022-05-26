# README of DFlare

This is the implementation of DFlare.

## Note
For DNN models and seed inputs, this folder only includes the three models collects from Diffchaser and the corresponding seed inputs.
Otherwise, its size will exceed the limit of GitHub Repo.
We are working on host all models and code in a seperate website. 
If you want to access them before that, please contact us.

## Instructions
The main logic of DFlare is in `test_gen_main.py`. To run the code, please run the wrapper: `gen_wrapper_tflite.py`

Here is an example:
```bash
python3 gen_wrapper_tflite.py --dataset mnist --arch lenet1 --maxit 10000000 --seed 0 --num 500 --cps_type quan --output_dir ./results
```

where `model` can be one of 'lenet1', 'lenet5', 'resnet'.
If `model` is `resnet`, then it should be `--dataset cifar`.


`seed` can be 0,1,2,3,4.

## Dataset and Models.
`./diffchaser_models` list the models from diffchaser.
`seed_inputs.p` stores the seed inputs used by us.

To apply it on other models, please revise the `gen_wrapper_tflite.py'

## Environment.
DFlare majorly requires the following package.
```
pyflann-py3==0.1.0
opencv-python-headless==4.5.3.56
numpy
```

However, the tflite model requires much more. Here we give the list of package in our environment.  A full list package for reference is in `tflite_requirement.txt`.

### System Package
```
pip: 21.2.4
cuda: 10.1
```

### python package
```
Keras==2.4.3
tensorflow==2.2.0
```


