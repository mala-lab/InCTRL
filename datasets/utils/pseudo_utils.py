import torch
import math
import numpy as np
import imgaug.augmenters as iaa
import cv2

augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([augmenters[aug_ind[0]],
                            augmenters[aug_ind[1]],
                            augmenters[aug_ind[2]]]
                            )
    return aug

def augment_image(image, anomaly_source_path, resize_shape=(256, 256)):
    aug = randAugmenter() # rand aug
    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(resize_shape[1], resize_shape[0]))

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((resize_shape[0], resize_shape[1]), (perlin_scalex, perlin_scaley))
    perlin_noise = rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8

    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)

    no_anomaly = torch.rand(1).numpy()[0]
    if no_anomaly > 0.5:
        image = image.astype(np.float32)
        return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
    else:
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

def transform_image(self, image_path, anomaly_source_path, resize_shape=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))

    do_aug_orig = torch.rand(1).numpy()[0] > 0.7
    if do_aug_orig:
        image = rot(image=image)

    image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
    augmented_image, anomaly_mask, has_anomaly = augment_image(image, anomaly_source_path)
    augmented_image = np.transpose(augmented_image, (2, 0, 1))
    image = np.transpose(image, (2, 0, 1))
    anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
    return image, augmented_image, anomaly_mask, has_anomaly

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise