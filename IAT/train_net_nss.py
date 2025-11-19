"""
Modified FC-CLIP training script to train on the synthetic NSS processed dataset.

This version assumes the dataset created by `mask_and_selection.py` with structure:

    processed_dataset/
      images/
        real_00000.png
        real_00001.png
        ...
      masks/
        scene_00000_obj_00.png
        scene_00000_obj_01.png
        ...
      scene_00000.json
      scene_00001.json
      ...

Each scene_XXXXX.json is expected to have the format:

    {
      "scene_id": "scene_00000",
      "real_image": "images/real_00000.png",
      "canvas_width": 1024,
      "canvas_height": 1024,
      "objects": [
        {
          "index": 0,
          "name": "car",
          "box": {"x": 120, "y": 430, "w": 260, "h": 140},
          "mask_path": "masks/scene_00000_obj_00.png",
          "clip_score": 0.35,
          "mask_uncertainty": 0.08
        },
        ...
      ]
    }

This script:
  * Registers a Detectron2 instance-segmentation dataset called `nss_processed_train`
    from `./processed_dataset`.
  * Maps each unique `name` in the JSONs to an integer category id.
  * Converts the binary masks into COCO-style RLE segmentations (via pycocotools).
  * Sets `dataset_dict["is_synthetic"] = True` for every sample so that FCCLIP
    can apply the DreamMask synthetic–real alignment loss.
  * Overrides cfg.DATASETS.TRAIN and cfg.INPUT.DATASET_MAPPER_NAME to use this dataset
    with the existing MaskFormer instance mapper.
"""

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except Exception:
    pass

import copy
import itertools
import logging
import os
import json
from collections import OrderedDict
from typing import Any, Dict, List, Set

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from fcclip import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_fcclip_config,
)

# -------------------------------------------------------------------------
# Custom NSS processed dataset registration
# -------------------------------------------------------------------------

PROCESSED_DATASET_ROOT = "processed_dataset"
PROCESSED_DATASET_NAME = "nss_processed_train"


def _load_nss_processed_dataset(root_dir: str) -> List[Dict[str, Any]]:
    """
    Build a detectron2-style instance segmentation dataset from processed_dataset.

    Returns a list of dataset dicts, each with:
      - file_name
      - height
      - width
      - image_id
      - is_synthetic = True
      - annotations: list of {
          bbox, bbox_mode, category_id, segmentation (RLE), iscrowd
        }

    Also sets MetadataCatalog[PROCESSED_DATASET_NAME].thing_classes.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"processed_dataset root '{root_dir}' not found. "
            f"Please run mask_and_selection.py first."
        )

    scene_files = sorted(
        f
        for f in os.listdir(root_dir)
        if f.startswith("scene_") and f.endswith(".json")
    )
    if not scene_files:
        raise RuntimeError(
            f"No scene_XXXXX.json files found under '{root_dir}'. "
            f"Check that mask_and_selection.py finished correctly."
        )

    dataset_dicts: List[Dict[str, Any]] = []
    category_name_to_id: Dict[str, int] = {}
    next_cat_id = 0

    for img_id, scene_file in enumerate(scene_files):
        json_path = os.path.join(root_dir, scene_file)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        real_rel = meta.get("real_image")
        if real_rel is None:
            raise KeyError(f"Missing 'real_image' in {json_path}")

        image_path = os.path.join(root_dir, real_rel)
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Real image specified in {json_path} not found: {image_path}"
            )

        with Image.open(image_path) as im:
            width, height = im.size

        record: Dict[str, Any] = {
            "file_name": image_path,
            "image_id": img_id,
            "height": height,
            "width": width,
            # DreamMask: mark synthetic samples so FCCLIP can use SRA loss.
            "is_synthetic": True,
        }

        objects = meta.get("objects", [])
        annos: List[Dict[str, Any]] = []

        for obj in objects:
            name = obj.get("name", "object")
            if name not in category_name_to_id:
                category_name_to_id[name] = next_cat_id
                next_cat_id += 1
            cat_id = category_name_to_id[name]

            box = obj.get("box", {})
            x = float(box.get("x", 0.0))
            y = float(box.get("y", 0.0))
            w = float(box.get("w", 1.0))
            h = float(box.get("h", 1.0))

            mask_rel = obj.get("mask_path")
            if mask_rel is None:
                # Skip objects without mask (shouldn't happen in processed_dataset)
                continue
            mask_path = os.path.join(root_dir, mask_rel)
            if not os.path.exists(mask_path):
                # Skip if mask missing
                continue

            # Load binary mask and convert to COCO RLE
            from PIL import Image as _Image
            with _Image.open(mask_path) as m_im:
                mask_np = np.array(m_im.convert("L")) > 0  # bool mask

            mask_fortran = np.asfortranarray(mask_np.astype(np.uint8))
            rle = mask_utils.encode(mask_fortran)
            # pycocotools returns bytes for 'counts'; convert to utf-8 string
            rle["counts"] = rle["counts"].decode("ascii")

            anno = {
                "bbox": [x, y, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat_id,
                "segmentation": rle,
                "iscrowd": 0,
            }
            annos.append(anno)

        if not annos:
            # If a scene has no valid objects, skip it
            continue

        record["annotations"] = annos
        dataset_dicts.append(record)

    # Build thing_classes list in category id order
    thing_classes = [None] * len(category_name_to_id)
    for name, idx in category_name_to_id.items():
        thing_classes[idx] = name

    MetadataCatalog.get(PROCESSED_DATASET_NAME).set(
        thing_classes=thing_classes,
        evaluator_type="coco",  # reuse COCO-style instance evaluator if needed
    )

    return dataset_dicts


def register_nss_processed_dataset(
    root_dir: str = PROCESSED_DATASET_ROOT,
    dataset_name: str = PROCESSED_DATASET_NAME,
) -> None:
    """Register the NSS processed dataset into Detectron2's DatasetCatalog."""
    if dataset_name in DatasetCatalog.list():
        # Already registered
        return

    DatasetCatalog.register(
        dataset_name, lambda root=root_dir: _load_nss_processed_dataset(root)
    )
    # Metadata (thing_classes, evaluator_type) is set inside _load_nss_processed_dataset


# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------


class Trainer(DefaultTrainer):
    """Extension of the Trainer class adapted to FCCLIP + NSS dataset."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given dataset."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, output_dir=output_folder)
            )
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(
                    COCOPanopticEvaluator(dataset_name, output_folder)
                )
        # COCO
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                COCOEvaluator(dataset_name, output_dir=output_folder)
            )
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Mapillary Vistas
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if (
            evaluator_type == "ade20k_panoptic_seg"
            and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """It now calls :func:`detectron2.solver.build_lr_scheduler`."""
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults: Dict[str, Any] = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg,
                name,
                output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"),
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Register our synthetic NSS processed dataset and override the train set
    register_nss_processed_dataset(PROCESSED_DATASET_ROOT, PROCESSED_DATASET_NAME)
    cfg.defrost()
    cfg.DATASETS.TRAIN = (PROCESSED_DATASET_NAME,)
    cfg.DATASETS.TEST = ()  # no test set by default for synthetic data
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    cfg.freeze()

    default_setup(cfg, args)
    # Setup logger for "fcclip" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="fcclip"
    )
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        frozen_params_exclude_text = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                continue
            # ignore text tower
            if (
                "clip_model.token_embedding" in n
                or "clip_model.positional_embedding" in n
                or "clip_model.transformer" in n
                or "clip_model.ln_final" in n
                or "clip_model.text_projection" in n
            ):
                continue
            frozen_params_exclude_text += p.numel()
        print(
            f"total_params: {total_params}, trainable_params: {trainable_params}, "
            f"frozen_params: {frozen_params}, "
            f"frozen_params_exclude_text: {frozen_params_exclude_text}"
        )

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
