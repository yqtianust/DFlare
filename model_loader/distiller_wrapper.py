import os
from typing import Tuple

import torch
from distiller import apputils, distiller
from distiller.models import create_model
from torch import nn
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter, ClassErrorMeter


def load_data(dataset: str, dataset_path: str, batch_size: int, workers: int) -> \
        Tuple[DataLoader, DataLoader, DataLoader]:
    """
    load dataset
    :param dataset: dataset type, one of 'imagenet', 'cifar10'
    :param dataset_path: the path to the dataset
    :param batch_size: the batch size when loading the dataset
    :param workers: number of workers used when loading dataset
    :return: train_loader, validation_loader, test_loader
    """
    # load data using distiller apputils,
    # TODO there are several params of apputils.load_data() is omitted. We may customize
    _train_loader, _val_loader, _test_loader, _ = \
        apputils.load_data(dataset, os.path.expanduser(dataset_path), batch_size, workers)
    return _train_loader, _val_loader, _test_loader


def load_checkpoint(arch: str, dataset: str, checkpoint_path: str, device: str = None, device_ids=None) -> nn.Module:
    """
    load checkpoint (model)
    :param arch: architecture of the model, options shown in distiller/apputils/image_classifier.py
    :param dataset: dataset type, one of 'imagenet', 'cifar10'
    :param checkpoint_path: the path to the checkpoint (pth or pth.tar) file
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :param device_ids: id of devices used
    :return: loaded model
    """
    # Create a pytorch model based on the model architecture and dataset
    _model: nn.Module = create_model(False, dataset, arch, parallel=True, device_ids=device_ids)
    # load checkpoint
    _model, compression_scheduler, optimizer, start_epoch = \
        apputils.load_checkpoint(_model, checkpoint_path, model_device=device)
    if optimizer is None:
        # TODO Not sure what is this for. Just copy from distiller image_classifier.py with default arguments
        optimizer = torch.optim.SGD(_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    if compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(_model)
    # TODO by now we only return resumed model, leaving compression scheduler and optimizer
    return _model


def evaluate_model(_model: nn.Module, data_loader: DataLoader, device: str = 'cuda'):
    # Switch to evaluation mode
    _model.eval()

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    losses = {'objective_loss': AverageValueMeter()}
    classerr = ClassErrorMeter(accuracy=True, topk=(1, 5))
    with torch.no_grad():
        for validation_step, (inputs, target) in enumerate(data_loader):
            print("----", validation_step, "----")
            inputs, target = inputs.to(device), target.to(device)
            # print(inputs.shape,target.shape)
            # print(target.data)
            # compute output from model
            output = _model(inputs)
            # print(output.shape,output.detach().shape,target.shape)
            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.detach(), target)
            print("==> Top1: {}    Top5: {}    Loss: {}".format(classerr.value()[0], classerr.value()[1],
                                                                losses['objective_loss'].mean))


if __name__ == '__main__':
    _, _, test_loader = load_data("cifar10", "~/evaluate-project/data.cifar10", 1024, workers=16)
    model = load_checkpoint("vgg16_cifar", "cifar10",
                            "~/evaluate-project/compress_evaluate/output/cifar10/prune/vgg16/2020.07.20-220129/best.pth.tar")
    evaluate_model(model, test_loader)
