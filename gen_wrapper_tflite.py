import os

from myLib.Inputs import SubsetInputs
from myUtils import create_folder, myLogger
from test_gen_main import tf_gen
from proj_utils import common_argparser
from model_loader.load_tflite import load_cps_model, load_org_model
from myLib.Result import PredictResult

import numpy as np


def create_predict_function_tflite(org_model, cps_model):
    def predict(input_img):
        org_vec = org_model(input_img)
        cps_vec = cps_model(input_img)
        # cross_entropy = -torch.sum(org_output * torch.log(cps_output), dim=1)

        org_result = PredictResult(org_vec)
        cps_result = PredictResult(cps_vec)

        return org_result, cps_result

    return predict


def preprocessing_mnist_tflite(img: np.array) -> np.array:
    img = img.astype(np.float32)
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    if len(img.shape) == 3:

        # change to HWC from CHW
        if img.shape[0] == 1 or img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        img = img[np.newaxis, ...]

    img = img / 255.0

    return img


def preprocessing_cifar_tflite(img: np.array) -> np.array:
    img = img.astype(np.float32)

    if len(img.shape) == 3:
        img = img[np.newaxis, ...]

    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]

    return img


def main():
    parser = common_argparser()
    parser.add_argument("--cps_type", choices=["tflite", "quan"])
    args = parser.parse_args()

    assert args.dataset == "mnist" or args.dataset == "cifar"

    if "lenet1" in args.arch or "lenet5" in args.arch:
        assert args.dataset == "mnist"
    elif "resnet" in args.arch:
        assert args.dataset == "cifar"

    # TODO: change logger filename

    output_file_folder_name = "{}_{}_{}_{}".format(args.arch, args.cps_type, args.seed, args.attack_mode)

    save_dir = os.path.join(args.output_dir, "{}-tflite".format(args.dataset),
                            output_file_folder_name)
    create_folder(save_dir)

    logger_filename = os.path.join(args.output_dir,"{}-tflite".format(args.dataset), "{}.log".format(output_file_folder_name))
    logger = myLogger.create_logger(logger_filename)
    print(args)
    logger(args)

    logger("Load model")

    # TODO load model
    if "lenet1" in args.arch or "lenet5" in args.arch or "resnet" in args.arch:
        org_model_raw, org_model = load_org_model("./diffchaser_models/{}.h5".format(args.arch))
    else:
        raise NotImplemented

    if "tflite" in args.cps_type:
        cps_model_raw, cps_model = load_cps_model("./diffchaser_models/{}.lite".format(args.arch))
    elif "quan" in args.cps_type:
        cps_model_raw, cps_model = load_cps_model("./diffchaser_models/{}-quan.lite".format(args.arch))
    else:
        raise NotImplemented

    # create predict function
    predict_f = create_predict_function_tflite(org_model, cps_model)

    # prepare inputs set
    input_sets = SubsetInputs(args.dataset, "tensorflow", args.arch, "quan-lite", args.seed)

    # start the attack
    logger("Stat the attack")

    if args.dataset == "mnist":
        preprocessing = preprocessing_mnist_tflite
    elif args.dataset == "cifar":
        preprocessing = preprocessing_cifar_tflite
    else:
        raise NotImplementedError

    # pm_attack(args, input_sets, logger, save_dir, predict_f, preprocessing)

    # start the attack
    print("Start the attack")
    logger("Start the attack")
    tf_gen(args, input_sets, logger, save_dir, predict_f, preprocessing)


if __name__ == '__main__':
    """
    DFlare for compressed model using tflite
    """

    main()
