# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under the Apache-2.0 License.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask
from .torchvision_datasets import CocoDetection as TvCocoDetection
from .torchvision_datasets import CocoDetection_semi as TvCocoDetection_semi
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import datasets.transforms_with_record as Tr


# COCO dataset class    Only used for the Supervised/"Burn-in" stage of training
class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1,
                 mixup_imageset=None):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode,
                                            local_rank=local_rank,
                                            local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        if mixup_imageset is not None and isinstance(mixup_imageset, CocoDetection_semi):  # New
            self.mixup_dataset = mixup_imageset
        else:
            self.mixup_dataset = None
        # TODO: Apply mixup in this function/class

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # Start of mixup stuff
        if self.mixup_dataset is not None:
            img_ground, target_ground, indicator_ground, labeltype_ground = super(CocoDetection_semi, self.mixup_dataset).__getitem__(idx)
            # the indicators and label types are assumed to be the same
            img, target = mixup_coco((img, img_ground), (target, target_ground), (10, 2))
        # End of mixup stuff

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


# COCO dataset class    Used for semi-supervised and unsupervised training
class CocoDetection_semi(TvCocoDetection_semi):
    def __init__(self, img_folder, ann_file, transforms_strong, transforms_weak, return_masks, cache_mode=False,
                 local_rank=0,
                 local_size=1, mixup_imageset=None):
        # img_folder: The path to the folder containing the images.
        # ann_file: The path to the COCO annotations file.
        # transforms_strong: A list of strong transformations or a callable object that applies strong augmentations to the images.
        # transforms_weak: A list of weak transformations or a callable object that applies weak augmentations to the images.
        # return_masks: A boolean indicating whether to return the mask information.
        # cache_mode: An optional boolean specifying whether to cache the dataset in memory. Defaults to False.
        # local_rank: An optional integer specifying the local rank of the process. Defaults to 0.
        # local_size: An optional integer specifying the total number of processes. Defaults to 1.
        # mixup_dataset: An optional coco dataset to perform mixup with. Has to be a CocoDetection_semi object

        # call the initializer of the CocoDetection_semi function from /torchvision_datasets/coco.py
        super(CocoDetection_semi, self).__init__(img_folder, ann_file,
                                                 cache_mode=cache_mode,
                                                 local_rank=local_rank,
                                                 local_size=local_size)
        self._transforms_strong = transforms_strong  # initializes the self._transforms_strong attribute with a list of transforms from either version of make_coco_strong_transforms_with_record()
        self._transforms_weak = transforms_weak  # initializes the self._transforms_weak attribute with a list of transforms from either version of make_coco_weak_transforms_with_record()
        self.prepare = ConvertCocoPolysToMask(
            return_masks)  # initializes the self.prepare attribute with an instance of the ConvertCocoPolysToMask class

        if mixup_imageset is not None and isinstance(mixup_imageset, CocoDetection_semi):  # New
            self.mixup_dataset = mixup_imageset
        else:
            self.mixup_dataset = None

    def __getitem__(self, idx):
        img, target, indicator, labeltype = super(CocoDetection_semi, self).__getitem__(
            idx)  # Get the image, its associated annotations, indicator, and label type (can be either "fully" or "Unsup" from what I can tell) info
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}  # Before being processed, target is a 2 field dictionary, containing an image's id and its associated annotations

        # TODO: Apply mixup here (probably? so you can concatenate the annotations and process the ground annotations as well)
        # Start of mixup stuff
        if self.mixup_dataset is not None:
            img_ground, target_ground, indicator_ground, labeltype_ground = super(CocoDetection_semi, self.mixup_dataset).__getitem__(idx)
            # the indicators and label types are assumed to be the same
            img, target = mixup_coco((img, img_ground), (target, target_ground), (10, 2))
        # End of mixup stuff

        img, target = self.prepare(img, target)  # Run the image and its associated annotations through the __call__ function of the ConvertCocoPolysToMask class object contained in self.prepare to get a processed target object

        if self._transforms_strong is not None:  # Apply any associated transforms to the fetched image
            record_q = {}
            record_q['OriginalImageSize'] = [img.height, img.width]
            img_q, target_q, record_q = self._transforms_strong(img, target, record_q)
        if self._transforms_weak is not None:
            record_k = {}
            record_k['OriginalImageSize'] = [img.height, img.width]
            img_k, target_k, record_k = self._transforms_weak(img, target, record_k)
        return img_q, target_q, record_q, img_k, target_k, record_k, indicator, labeltype


# Only used if the final overall goal of the model involves segmentation
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


# This is the class (it's a function really) that creates the target objects/annotation dictionaries of images, given the image and its associated annotations
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        # formatted_target = json.dumps(target, indent=2)
        print("TARGET", target["image_id"], "Annotations:", len(target["annotations"]))

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # formatted_dict = json.dumps(anno[0:3], indent=2)
        # print("ANNO:", formatted_dict)

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        points = [obj["point"] for obj in anno]
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)
        print("POINTS SHAPE:", points.shape)

        perspectives = [obj["perspective"] for obj in anno]  # New
        perspectives = torch.as_tensor(perspectives, dtype=torch.float32)  # .reshape(-1, 2)  # New
        print("PERSPECTIVES SHAPE:", perspectives.shape)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        print("BOXES SHAPE:", boxes.shape)
        print("KEEP SHAPE:", keep.shape)
        print("KEEP VALUE:", keep)

        boxes = boxes[keep]
        classes = classes[keep]
        points = points[keep]
        perspectives = perspectives[keep]  # New
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}

        target["boxes"] = boxes
        target["labels"] = classes
        target["points"] = points
        target["perspectives"] = perspectives
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # for a small of number samples on objects365, the instance number > 300, so we simply only pick up the first 300 annotaitons. 300 is the query number
        if target["boxes"].shape[0] > 300:
            fields = ["boxes", "labels", "area", "points", "iscrowd"]
            for field in fields:
                target[field] = target[field][:300]
                # TODO: probably change this to have either an even distribution of aerial and ground annotations, or set the split to be equal to the lamba value used in the mixup

        return image, target


def mixup(input, alpha, share_lam=False):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    if share_lam:
        lam = beta.sample().to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam
    else:
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam


# def NewMixup(dataset1, dataset2, alpha):
#     if not isinstance(alpha, (list, tuple)):
#         alpha = [alpha, alpha]
#     beta = torch.distributions.beta.Beta(*alpha)
#     lam = beta.sample([dataset1.__len__()]).to(device="cuda")
#     lam = torch.max(lam, 1. - lam)
#     lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
#
#     data_loader_1 = DataLoader(dataset1)
#     data_loader_2 = DataLoader(dataset2)
#
#     mixup_outputs = []
#     mixup_lambdas = []
#
#     for input_1, input_2 in zip(data_loader_1, data_loader_2):
#         lam = beta.sample([input_1.shape[0]]).to(device='cuda')
#         lam = torch.max(lam, 1. - lam)
#         lam_expanded = lam.view([-1] + [1] * (input_1.dim() - 1))
#
#         # Perform mixup using corresponding images from both datasets
#         output = lam_expanded * input_1 + (1. - lam_expanded) * input_2
#         mixup_outputs.append(output)
#         mixup_lambdas.append(lam)
#
#     output = lam_expanded * dataset1 + (1. - lam_expanded) * dataset2
#     return output, lam


def mixup_coco(images, targets, alpha=(0.75, 0.75)):
    """
    Perform mixup on two images from a COCO formatted dataset and combine their associated information.

    Args:
        images (tuple): A tuple containing two PIL.Image objects.
        targets (tuple): A tuple containing two target annotation lists.
        alpha (tuple): Mixup parameter. The two coefficients that control the interpolation strength between images and annotations. Defaults to 0.75. (10,2) should give you a distribution spread that's focused between 0.75 and 1

    Returns:
        mixed_image (PIL.Image): The mixed image obtained by combining the input images.
        mixed_target (dictionary): The combined target annotations, along with perspective filtering mask and lambda value used in the mixup.
    """

    # Generate a random mixup ratio
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    lam = beta.sample()
    lam = torch.max(lam, 1. - lam)
    # mixup_ratio = torch.distributions.beta.Beta(alpha, alpha).sample().item()

    # Mix the images using the mixup_ratio
    image1, image2 = images
    image1_tensor = torch.from_numpy(np.array(image1)).float()
    image2_tensor = torch.from_numpy(np.array(image2)).float()
    mixed_image_tensor = lam * image1_tensor + (1 - lam) * image2_tensor
    mixed_image = Image.fromarray(mixed_image_tensor.numpy().astype(np.uint8))

    # Mix the target annotations
    target1, target2 = targets  # target is a 2 item dictionary, containing an image's id and list of associated annotations
    if target1['image_id'] != target2['image_id']:
        raise ValueError("The image_id values are not the same, cannot combine dictionaries.")
    combined_annotations = target1['annotations'] + target2['annotations']
    perspective_mask = torch.cat((torch.ones(len(target1['annotations']), dtype=torch.int), torch.zeros(len(target2['annotations']), dtype=torch.int)))
    mixed_target = {'image_id': target1['image_id'], 'annotations': combined_annotations, 'perspective_mask': perspective_mask, 'lambda': lam}
    # perspective mask: a tensor of ones and zeroes that's to be used to quickly filter the list of annotations to just the aerial or just the ground annotations through matrix operations
    # lambda: the lambda value used in the mixup. Passed along with the annotations for the purpose of the weighted loss implementation

    # Mix the indicators
    # if isinstance(indicators[0], (int, float)) and isinstance(indicators[1], (int, float)):
    #     mixed_indicator = mixup_ratio * indicators[0] + (1 - mixup_ratio) * indicators[1]
    # else:
    #     mixed_indicator = indicators[0]

    # Mix the label_types
    # if isinstance(label_types[0], (int, float)) and isinstance(label_types[1], (int, float)):
    #     mixed_label_type = mixup_ratio * label_types[0] + (1 - mixup_ratio) * label_types[1]
    # else:
    #     mixed_label_type = label_types[0]

    return mixed_image, mixed_target  #, mixed_indicator, mixed_label_type


# Only Used by build (for supervised learning/burn in stage)
def make_coco_transforms600(
        image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(i * 600 / 800) for i in scales]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1000),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1000),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# 800 pixels    Only Used by build (for supervised learning/burn in stage)
def make_coco_transforms(
        image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# 800 pixels
def make_coco_strong_transforms_with_record(
        image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            Tr.RandomColorJiter(),
            Tr.RandomGrayScale(),
            Tr.RandomGaussianBlur(),
            # Tr.RandomContrast(),
            # Tr.RandomAdjustSharpness(),
            # Tr.RandomPosterize(),
            # Tr.RandomSolarize(),
            Tr.RandomSelect(
                Tr.RandomResize(scales, max_size=1333),
                Tr.Compose([
                    Tr.RandomResize([400, 500, 600]),
                    Tr.RandomSizeCrop(384, 600),
                    Tr.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
            Tr.RandomErasing1(),
            Tr.RandomErasing2(),
            Tr.RandomErasing3(),
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# 800 pixels    Only used
def make_coco_weak_transforms_with_record(
        image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_strong_transforms_with_record600(
        image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [int(i * 600 / 800) for i in scales]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            Tr.RandomColorJiter(),
            Tr.RandomGrayScale(),
            Tr.RandomGaussianBlur(),
            # Tr.RandomContrast(),
            # Tr.RandomAdjustSharpness(),
            # Tr.RandomPosterize(),
            # Tr.RandomSolarize(),
            Tr.RandomSelect(
                Tr.RandomResize(scales, max_size=1000),
                Tr.Compose([
                    Tr.RandomResize([400, 500, 600]),
                    Tr.RandomSizeCrop(384, 600),
                    Tr.RandomResize(scales, max_size=1000),
                ])
            ),
            normalize,
            Tr.RandomErasing1(),
            Tr.RandomErasing2(),
            Tr.RandomErasing3(),
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_weak_transforms_with_record600(
        image_set):  # image_set can be either 'train' or 'val' and will return a list of transforms to be performed

    normalize = Tr.Compose([
        Tr.ToTensor(),
        Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return Tr.Compose([
            Tr.RandomHorizontalFlip(),
            normalize,
        ])

    if image_set == 'val':
        return Tr.Compose([
            Tr.RandomResize([600], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if args.dataset_file == "voc_omni":
        PATHS = {
            "train": (root / "VOCdevkit/VOC20072012trainval/JPEGImages",
                      root / "VOCdevkit/VOC20072012trainval" / args.annotation_json_label),
            "val": (
                root / "VOCdevkit/VOC2007test/JPEGImages",
                root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "voc_semi":
        PATHS = {
            "train": (root / "VOCdevkit/VOC2007trainval/JPEGImages",
                      root / "VOCdevkit/VOC2007trainval" / args.annotation_json_label),
            "val": (
                root / "VOCdevkit/VOC2007test/JPEGImages",
                root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "bees_omni":
        PATHS = {
            "train": (root / "ML-Data", root / args.annotation_json_label),
            "val": (root / "ML-Data", root / 'instances_bees_val.json'),
        }
    elif args.dataset_file == "crowdhuman_omni":
        PATHS = {
            "train": (root / "Images", root / args.annotation_json_label),
            "val": (root / "Images", root / 'test_fullbody.json'),
        }
    elif args.dataset_file == "objects_omni":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_label),
            "val": (root / "val_objects365", root / "annotations" / 'objects365_val_w_indicator.json'),
        }
    elif args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point":
        PATHS = {
            "train": (root / "val2014", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / 'instances_w_indicator_val2017.json'),
        }
    elif args.dataset_file == "dvd":
        PATHS = {
            "train": (root / "scaled_dataset/train/droneview",
                      root / "supervised_annotations" / args.annotation_json_aerial_label,
                      root / "scaled_dataset/train/groundview",
                      root / "supervised_annotations" / args.annotation_json_ground_label),
            "val": (root / "scaled_dataset/val/droneview",
                    root / "supervised_annotations" / 'aerial/aligned_ids/aerial_valid_aligned_ids_w_annotation.json'),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / 'instances_w_indicator_val2017.json'),
        }

    if image_set == "train":
        img_folder_aerial, ann_file_aerial, img_folder_ground, ann_file_ground = PATHS[image_set]
    else:
        img_folder, ann_file = PATHS[image_set]

    if args.pixels == 600:
        if image_set == "train":
            dataset_ground = CocoDetection(img_folder_ground, ann_file_ground,
                                           transforms=make_coco_transforms600(image_set),
                                           return_masks=args.masks,
                                           cache_mode=args.cache_mode,
                                           local_rank=get_local_rank(),
                                           local_size=get_local_size())

            dataset = CocoDetection(img_folder_aerial, ann_file_aerial,
                                    transforms=make_coco_transforms600(image_set),
                                    return_masks=args.masks,
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size(),
                                    mixup_imageset=dataset_ground)
        else:
            dataset = CocoDetection(img_folder, ann_file,
                                    transforms=make_coco_transforms600(image_set),
                                    return_masks=args.masks,
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size())
    elif args.pixels == 800:
        if image_set == "train":
            dataset_ground = CocoDetection(img_folder_ground, ann_file_ground,
                                           transforms=make_coco_transforms(image_set),
                                           return_masks=args.masks,
                                           cache_mode=args.cache_mode,
                                           local_rank=get_local_rank(),
                                           local_size=get_local_size())

            dataset = CocoDetection(img_folder_aerial, ann_file_aerial,
                                    transforms=make_coco_transforms(image_set),
                                    return_masks=args.masks,
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size(),
                                    mixup_imageset=dataset_ground)
        else:
            dataset = CocoDetection(img_folder_ground, ann_file_ground,
                                    transforms=make_coco_transforms(image_set),
                                    return_masks=args.masks,
                                    cache_mode=args.cache_mode,
                                    local_rank=get_local_rank(),
                                    local_size=get_local_size())

    return dataset


def build_semi_label(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    if args.dataset_file == "voc_semi":
        PATHS = {
            "train": (root / "VOCdevkit/VOC2007trainval/JPEGImages",
                      root / "VOCdevkit/VOC2007trainval" / args.annotation_json_label),
            "val": (
                root / "OCdevkit/VOC2007test/JPEGImages",
                root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "voc_omni":
        PATHS = {
            "train": (root / "VOCdevkit/VOC20072012trainval/JPEGImages",
                      root / "VOCdevkit/VOC20072012trainval" / args.annotation_json_label),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point":
        PATHS = {
            "train": (root / "val2014", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "bees_omni":
        PATHS = {
            "train": (root / "ML-Data", root / args.annotation_json_label),
            "val": (root / "ML-Data", root / 'instances_bees_val.json'),
        }
    elif args.dataset_file == "crowdhuman_omni":
        PATHS = {
            "train": (root / "Images", root / args.annotation_json_label),
            "val": (root / "Images", root / 'test_fullbody.json'),
        }
    elif args.dataset_file == "objects_omni":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_label),
            "val": (root / "val_objects365", root / "annotations" / 'objects365_val_w_indicator.json'),
        }
    elif args.dataset_file == "dvd":
        PATHS = {
            "train": (root / "scaled_dataset/train/droneview",
                      root / "supervised_annotations" / args.annotation_json_aerial_label,
                      root / "scaled_dataset/train/groundview",
                      root / "supervised_annotations" / args.annotation_json_ground_label),
            "val": (root / "scaled_dataset/val/droneview",
                    root / "supervised_annotations" / 'aerial/aligned_ids/aerial_valid_aligned_ids_w_annotation.json'),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_label),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }

    if image_set == "train":
        img_folder_aerial, ann_file_aerial, img_folder_ground, ann_file_ground = PATHS[image_set]
    else:
        img_folder, ann_file = PATHS[image_set]
    if args.pixels == 600:
        if image_set == "train":
            dataset_ground = CocoDetection_semi(img_folder_ground, ann_file_ground,
                                                transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                                transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                                return_masks=args.masks,
                                                cache_mode=args.cache_mode,
                                                local_rank=get_local_rank(),
                                                local_size=get_local_size())
            dataset = CocoDetection_semi(img_folder_aerial, ann_file_aerial,
                                         transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size(),
                                         mixup_imageset=dataset_ground)
        else:
            dataset = CocoDetection_semi(img_folder, ann_file,
                                         transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size())

    elif args.pixels == 800:
        if image_set == "train":
            dataset_ground = CocoDetection_semi(img_folder_ground, ann_file_ground,
                                                transforms_strong=make_coco_strong_transforms_with_record(image_set),
                                                transforms_weak=make_coco_weak_transforms_with_record(image_set),
                                                return_masks=args.masks,
                                                cache_mode=args.cache_mode,
                                                local_rank=get_local_rank(),
                                                local_size=get_local_size())
            dataset = CocoDetection_semi(img_folder_aerial, ann_file_aerial,
                                         transforms_strong=make_coco_strong_transforms_with_record(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size(),
                                         mixup_imageset=dataset_ground)
        else:
            dataset = CocoDetection_semi(img_folder, ann_file,
                                         transforms_strong=make_coco_strong_transforms_with_record(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size())

    return dataset


def build_semi_unlabel(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if args.dataset_file == "voc_semi":
        PATHS = {
            "train": (root / "VOCdevkit/VOC2012trainval/JPEGImages",
                      root / "VOCdevkit/VOC2012trainval" / args.annotation_json_unlabel),
            "val": (
                root / "OCdevkit/VOC2007test/JPEGImages",
                root / "VOCdevkit/VOC2007test" / 'instances_VOC_test2007.json'),
        }
    elif args.dataset_file == "coco_35to80_tagsU" or args.dataset_file == "coco_35to80_point":
        PATHS = {
            "train": (root / "train2014", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "voc_omni":
        PATHS = {
            "train": (root / "VOCdevkit/VOC20072012trainval/JPEGImages",
                      root / "VOCdevkit/VOC20072012trainval" / args.annotation_json_unlabel),
            "val": (root / "val2014", root / "annotations" / f'{mode}_val2014.json'),
        }
    elif args.dataset_file == "coco_add_semi":
        PATHS = {
            "train": (root / "unlabeled2017", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }
    elif args.dataset_file == "crowdhuman_omni":
        PATHS = {
            "train": (root / "Images", root / args.annotation_json_unlabel),
            "val": (root / "Images", root / 'test_fullbody.json'),
        }
    elif args.dataset_file == "bees_omni":
        PATHS = {
            "train": (root / "ML-Data", root / args.annotation_json_unlabel),
            "val": (root / "ML-Data", root / 'instances_bees_val.json'),
        }
    elif args.dataset_file == "coco_objects_tagsU" or args.dataset_file == "coco_objects_points":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }
    elif args.dataset_file == "objects_omni":
        PATHS = {
            "train": (root / "train_objects365", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val_objects365", root / "annotations" / 'objects365_val_w_indicator.json'),
        }
    elif args.dataset_file == "dvd":
        PATHS = {
            "train": (root / "unsupervised_annotations/aerial",
                      root / "unsupervised_annotations" / args.annotation_json_aerial_unlabel,
                      root / "unsupervised_annotations/ground",
                      root / "unsupervised_annotations" / args.annotation_json_ground_unlabel),
            "val": (root / "scaled_dataset/val/droneview",
                    root / "supervised_annotations" / 'aerial/aligned_ids/aerial_valid_aligned_ids_w_annotation.json'),
        }
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / args.annotation_json_unlabel),
            "val": (root / "val2017", root / "annotations" / f'{mode}_w_point_val2017.json'),
        }

    if image_set == "train":
        img_folder_aerial, ann_file_aerial, img_folder_ground, ann_file_ground = PATHS[image_set]
    else:
        img_folder, ann_file = PATHS[image_set]

    if args.pixels == 600:
        if image_set == "train":
            dataset_ground = CocoDetection_semi(img_folder_ground, ann_file_ground,
                                                transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                                transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                                return_masks=args.masks,
                                                cache_mode=args.cache_mode,
                                                local_rank=get_local_rank(),
                                                local_size=get_local_size())
            dataset = CocoDetection_semi(img_folder_aerial, ann_file_aerial,
                                         transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size(),
                                         mixup_imageset=dataset_ground)
        else:
            dataset = CocoDetection_semi(img_folder, ann_file,
                                         transforms_strong=make_coco_strong_transforms_with_record600(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record600(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size())

    elif args.pixels == 800:
        if image_set == "train":
            dataset_ground = CocoDetection_semi(img_folder_ground, ann_file_ground,
                                                transforms_strong=make_coco_strong_transforms_with_record(image_set),
                                                transforms_weak=make_coco_weak_transforms_with_record(image_set),
                                                return_masks=args.masks,
                                                cache_mode=args.cache_mode,
                                                local_rank=get_local_rank(),
                                                local_size=get_local_size())
            dataset = CocoDetection_semi(img_folder_aerial, ann_file_aerial,
                                         transforms_strong=make_coco_strong_transforms_with_record(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size(),
                                         mixup_imageset=dataset_ground)
        else:
            dataset = CocoDetection_semi(img_folder, ann_file,
                                         transforms_strong=make_coco_strong_transforms_with_record(image_set),
                                         transforms_weak=make_coco_weak_transforms_with_record(image_set),
                                         return_masks=args.masks,
                                         cache_mode=args.cache_mode,
                                         local_rank=get_local_rank(),
                                         local_size=get_local_size())

    return dataset
