from os import path
from typing import Union
from torch import nn

# the root dir of all cifar10 models
from . import distiller_wrapper

models_root_dir = "./mnist_models"
pth_path = {"lenet4_org": "./train/lenet4/2020.07.30-140826/best.pth.tar",
            "lenet4_prune": "./prune/lenet4/2020.07.30-150319/best.pth.tar",
            "lenet4_quan": "./quantization/lenet4/best.pth.tar",
            "lenet5_org": "./train/lenet5/2020.07.30-144508/best.pth.tar",
            "lenet5_prune": "./prune/lenet5/2020.07.30-183405/best.pth.tar",
            "lenet5_quan": "./quantization/lenet5/best.pth.tar",
            "cnn_org": "./train/cnn/2020.07.30-194946/best.pth.tar",
            "cnn_prune": "./prune/cnn/2020.07.30-231349/best.pth.tar",
            "cnn_quan": "./quantization/cnn/best.pth.tar",
            }

mut_pth_path = {"cnn_conv1_0_05": "./mutated/cnn/cnn_conv1_0_05.pt",
                "cnn_fc1_0_10": "./mutated/cnn/cnn_fc1_0_10.pt",
                "cnn_conv1_0_10": "./mutated/cnn/cnn_conv1_0_10.pt",
                "cnn_fc1_0_05": "./mutated/cnn/cnn_fc1_0_05.pt",
                "cnn_conv1_0_01": "./mutated/cnn/cnn_conv1_0_01.pt",
                "cnn_fc1_0_01": "./mutated/cnn/cnn_fc1_0_01.pt",
                "lenet4_conv1_0_10": "./mutated/lenet4/lenet4_conv1_0_10.pt",
                "lenet4_conv1_0_05": "./mutated/lenet4/lenet4_conv1_0_05.pt",
                "lenet4_fc1_0_10": "./mutated/lenet4/lenet4_fc1_0_10.pt",
                "lenet4_fc1_0_05": "./mutated/lenet4/lenet4_fc1_0_05.pt",
                "lenet4_fc1_0_01": "./mutated/lenet4/lenet4_fc1_0_01.pt",
                "lenet4_conv1_0_01": "./mutated/lenet4/lenet4_conv1_0_01.pt",
                "lenet5_conv1_0_01": "./mutated/lenet5/lenet5_conv1_0_01.pt",
                "lenet5_conv1_0_10": "./mutated/lenet5/lenet5_conv1_0_10.pt",
                "lenet5_fc1_0_05": "./mutated/lenet5/lenet5_fc1_0_05.pt",
                "lenet5_fc1_0_10": "./mutated/lenet5/lenet5_fc1_0_10.pt",
                "lenet5_conv1_0_05": "./mutated/lenet5/lenet5_conv1_0_05.pt",
                "lenet5_fc1_0_01": "./mutated/lenet5/lenet5_fc1_0_01.pt"}


def load_original_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    """
    Load original model of mnist dataset with different architecture.
    :param arch: one of 'lenet4', 'lenet5', 'cnn'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    return distiller_wrapper.load_checkpoint(arch="{}_mnist".format(arch),
                                             dataset="mnist",
                                             checkpoint_path=path.join(models_root_dir,
                                                                       pth_path["{}_org".format(arch)]),
                                             device=device)


def load_pruned_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    """
    Load pruned model of mnist dataset with different architecture.
    :param arch: one of 'lenet4', 'lenet5', 'cnn'
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    return distiller_wrapper.load_checkpoint(arch="{}_mnist".format(arch),
                                             dataset="mnist",
                                             checkpoint_path=path.join(models_root_dir,
                                                                       pth_path["{}_prune".format(arch)]),
                                             device=device)


def load_quantized_model(arch: str, device: Union[str, None] = None) -> nn.Module:
    return distiller_wrapper.load_checkpoint(arch="{}_mnist".format(arch),
                                             dataset="mnist",
                                             checkpoint_path=path.join(models_root_dir,
                                                                       pth_path["{}_quan".format(arch)]),
                                             device=device)


def load_mutated_model_str(mut_str, device):
    values = mut_str.split("_")
    arch = values[0]
    mut_layer = values[1]
    mut_seed = int(values[2])
    mut_per = int(values[3]) / 100
    return load_mutated_model(arch, mut_layer, mut_seed, mut_per, device)


def load_mutated_model(arch, mut_layer, mut_seed, mut_per, device):
    key = "{}_{}_{}_{:02}".format(arch, mut_layer, mut_seed, int(mut_per * 100))
    # print(key)
    # print(mut_pth_path.keys())
    assert key in mut_pth_path.keys()

    from distiller.models import create_model
    _model = create_model(False, dataset="mnist", arch="{}_mnist".format(arch), parallel=False, device_ids=device)
    import torch
    _ckpt = torch.load(path.join(models_root_dir, mut_pth_path[key]))
    # log(checkpoint)
    _model.load_state_dict(_ckpt['model_state_dict'])
    return _model


def load_compressed_model(arch: str, cps_type: str, device: Union[str, None] = None) -> nn.Module:
    """
    load model based on given architecture and compress type
    :param arch: one of 'lenet4', 'lenet5', 'cnn'
    :param cps_type: compress type, only 'prune' is supported currently
    :param device: if set, call model.to($model_device). This should be set to either 'cpu' or 'cuda'.
    :return: the loaded model
    """
    if cps_type == "prune":
        return load_pruned_model(arch, device)
    if cps_type == "quan":
        return load_quantized_model(arch, device)
    else:
        raise NotImplemented


if __name__ == '__main__':
    # test
    dataset_path = "./"
    # dataset_path = "./data.mnist"
    _, _, test_loader = distiller_wrapper.load_data("mnist", dataset_path, 16, 8)
    model = load_original_model("lenet4", "cuda")
    distiller_wrapper.evaluate_model(model, test_loader, "cuda")

    model = load_quantized_model("lenet4", "cuda")
    distiller_wrapper.evaluate_model(model, test_loader, "cuda")

    model = load_mutated_model("lenet4", "conv1", 0, 0.05, "cuda")
    distiller_wrapper.evaluate_model(model, test_loader, "cuda")
