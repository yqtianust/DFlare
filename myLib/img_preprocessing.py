import numpy as np


def preprocessing_mnist(img: np.array) -> np.array:
    img = img.astype(np.float32)

    # cv2 blur will return 28x28 matrix.
    # if len(img.shape) == 2:
    #     img = img[..., np.newaxis]

    if len(img.shape) == 3:
        # change to CHW
        if img.shape[-1] == 1:
            img = np.transpose(img, (2, 0, 1))

        img = img[np.newaxis, ...]

    img = img / 255.0
    mean = (0.1307,)
    std = (0.3081,)
    for ch in range(len(mean)):
        img[:, ch, :, :] = (img[:, ch, :, :] - mean[ch]) / std[ch]

    return img


def preprocessing_cifar(img: np.array) -> np.array:
    img = img.astype(np.float32)

    if len(img.shape) == 3:

        # change to CHW
        if img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))

        img = img[np.newaxis, ...]

    img = img / 255.0
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    for ch in range(len(mean)):
        img[:, ch, :, :] = (img[:, ch, :, :] - mean[ch]) / std[ch]

    return img
