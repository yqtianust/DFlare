# README of DFlare

This is the implementation of DFlare.

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

### Models used in our paper

### The model prepared by us

Pelase downloadn it from [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ytianas_connect_ust_hk/EmJ0HLIS1JZPoRJB9wUzZswBgfZ8mbLn54sCwNiaUyzoPg?e=hCv7Ol) and follow the instructions in the readme. 

#### ImageNet models 
Models are available at [PyTorch](https://pytorch.org/vision/stable/models.html#quantized-models)

#### DeepSpeech models 

Moldes are available at [DeepSpeech](https://github.com/mozilla/DeepSpeech).

Please note that the compressed model and the original models cannot be executed in the same enviroment. 
The compressed models need the python package ``deepspeech-tflite'' while the original models need the package ``deepspeech'' or ``deepspeech-gpu''.

More details are available at [here](https://deepspeech.readthedocs.io/en/r0.9/USING.html#getting-the-pre-trained-model)


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

## Questions

Leave a question using issue report or contact me via yongqiang.tian - at - uwaterloo.ca




