import argparse
from collections import OrderedDict
from os import path
from typing import Union

import numpy as np
import torch
from torch import nn

from . import distiller_wrapper

# the root dir of all cifar10 models
models_root_dir = "./cifar10_models"
pth_path = {"vgg16_org": "./train/vgg16/2020.07.20-172356/best.pth.tar",
            "vgg16_quan": "./quantization/vgg16/2020.07.21-134951/quantized_checkpoint.pth.tar",
            "vgg16_prune": "./prune/vgg16/2020.07.20-220129/best.pth.tar",
            "vgg16_kd": "./distillation/vgg16-vgg11/2020.07.21-135548/best.pth.tar",
            "resnet20_org": "./train/resnet20/2020.07.20-162220/best.pth.tar",
            "resnet20_quan": "./quantization/resnet20/2020.07.20-200129/quantized_checkpoint.pth.tar",
            "resnet20_prune": "./prune/resnet20/2020.07.20-192456/best.pth.tar",
            "resnet20_kd": "./distillation/resnet20-simplenet/2020.07.20-200823/best.pth.tar",
            "plain20_org": "./train/plain20/2020.07.20-164051/best.pth.tar",
            "plain20_quan": "./quantization/plain20/2020.07.21-134407/quantized_checkpoint.pth.tar",
            "plain20_prune": "./prune/plain20/2020.07.20-212728/best.pth.tar",
            "plain20_kd": "./distillation/plain20-simplenet/2020.07.21-135424/best.pth.tar",
            }


def get_pred_and_prob(output):
    if type(output) is not np.ndarray:
        output = output.numpy()

    argmin = np.argsort(output, axis=1)
    pred = argmin[:, -1]
    prob = output[np.arange(0, len(pred)), pred]
    return pred, prob


def remove_invalid_key(ckpt, keyword="module."):
    new_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace(keyword, "")  # remove `module.`
        new_dict[name] = v
    return new_dict


def load_original_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    """
    Load original model of cifar10 dataset with different architecture.
    :param arch: one of 'resnet20', 'vgg16', 'plain20'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    return distiller_wrapper.load_checkpoint(arch="{}_cifar".format(arch),
                                             dataset="cifar10",
                                             checkpoint_path=path.join(models_root_dir,
                                                                       pth_path["{}_org".format(arch)]),
                                             device=device)


def load_pruned_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    """
    Load pruned model of cifar10 dataset with different architecture.
    :param arch: one of 'resnet20', 'vgg16', 'plain20'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    return distiller_wrapper.load_checkpoint(arch="{}_cifar".format(arch),
                                             dataset="cifar10",
                                             checkpoint_path=path.join(models_root_dir,
                                                                       pth_path["{}_prune".format(arch)]),
                                             device=device)


def load_quantized_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    """
    Load quantized model of cifar10 dataset with different architecture.
    :param arch: one of 'resnet20', 'vgg16', 'plain20'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    return distiller_wrapper.load_checkpoint(arch="{}_cifar".format(arch),
                                             dataset="cifar10",
                                             checkpoint_path=path.join(models_root_dir,
                                                                       pth_path["{}_quan".format(arch)]),
                                             device=device)


def load_distilled_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    """
    Load knowledge-distilled model of cifar10 dataset with different architecture.
    :param arch: one of 'resnet20', 'vgg16', 'plain20'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    distillation_map = {
        "resnet20": "simplenet",
        "vgg16": "vgg11",
        "plain20": "simplenet",
    }
    return distiller_wrapper.load_checkpoint(arch="{}_cifar".format(distillation_map[arch]),
                                             dataset="cifar10",
                                             checkpoint_path=path.join(models_root_dir, pth_path["{}_kd".format(arch)]),
                                             device=device)


def load_compressed_model(arch: str, cps_type: str, device: Union[str, None] = None) -> nn.Module:
    """
    load model based on given architecture and compress type
    :param arch: one of 'resnet20', 'vgg16', 'plain20'
    :param cps_type: compress type, one of 'quan', 'prune', 'kd'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    if cps_type == "prune":
        return load_pruned_model(arch, device)
    elif cps_type == "quan":
        return load_quantized_model(arch, device)
    elif cps_type == "kd":
        return load_distilled_model(arch, device)
    else:
        raise NotImplemented


def main(args):
    device = torch.device("cpu" if args.cpu else "cuda")

    print("using {} original model and {} model".format(args.model, args.cps_type))
    if "vgg16" in args.model:
        org_model = load_original_model("vgg16", str(device))
        cps_model = load_compressed_model("vgg16", args.cps_type, str(device))
    elif "resnet20" in args.model:
        org_model = load_original_model("resnet20", str(device))
        cps_model = load_compressed_model("resnet20", args.cps_type, str(device))
    elif "plain20" in args.model:
        org_model = load_original_model("plain20", str(device))
        cps_model = load_compressed_model("plain20", args.cps_type, str(device))
    else:
        raise NotImplemented

    org_model.eval()
    cps_model.eval()

    print("Load data")
    # cifar10 dataset path
    dataset_path = "./data/"
    _, _, test_loader = distiller_wrapper.load_data("cifar10", dataset_path, batch_size=args.batch_size, workers=16)

    print("evaluate origin model")
    distiller_wrapper.evaluate_model(org_model, test_loader)
    print()
    print("evaluate compressed model")
    distiller_wrapper.evaluate_model(cps_model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cps-type", type=str, choices=["quan", "prune", "kd"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--find-num", type=int, default=2000, help="The number of inputs to be found")
    parser.add_argument("--cpu", action='store_true')
    args = parser.parse_args()
    # random.seed(args.seed)
    #
    # assert not os.path.exists(
    #     os.path.join(args.save_dir, args.model)), "Dir {} exists, please change to another one".format(os.path.join(
    #     args.save_dir, args.model))

    main(args)
