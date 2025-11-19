"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple, Optional, List, Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.transformer_decoder.fcclip_transformer_decoder import (
    MaskPooling,
    get_classification_logits,
)

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


@META_ARCH_REGISTRY.register()
class FCCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # FC-CLIP
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        # DreamMask synthetic–real alignment
        memory_bank_size: int,
        lambda_sra: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
            pixel_mean, pixel_std: per-channel mean/std for input normalization
            semantic_on / panoptic_on / instance_on: which heads to run at inference
            test_topk_per_image: instance segmentation parameter, keep topk instances per image
            geometric_ensemble_alpha / beta / ensemble_on_valid_mask: FC-CLIP open-vocab ensembling args
            memory_bank_size: length of real-feature memory queue per class (β in paper)
            lambda_sra: weight of synthetic–real alignment loss L_sra (λ in paper)
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        self.train_text_classifier: Optional[torch.Tensor] = None
        self.test_text_classifier: Optional[torch.Tensor] = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent)  # use this for void

        # text classifiers & overlapping mask
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(
            train_metadata, train_metadata
        )
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = (
            self.prepare_class_names_from_metadata(test_metadata, train_metadata)
        )

        # DreamMask synthetic–real alignment
        self.memory_bank_size = memory_bank_size
        self.lambda_sra = lambda_sra

        # number of training categories (for memory bank)
        try:
            num_train_cats = len(self.train_metadata.stuff_classes)
        except Exception:
            num_train_cats = len(self.train_metadata.thing_classes)

        feat_dim = backbone.dim_latent
        # memory bank: [num_train_cats, β, feat_dim]
        self.register_buffer(
            "real_memory_bank",
            torch.zeros(num_train_cats, memory_bank_size, feat_dim),
        )
        # write pointer per class: [num_train_cats]
        self.register_buffer(
            "real_memory_ptr",
            torch.zeros(num_train_cats, dtype=torch.long),
        )

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(", ", ",")
                x_ = x_.split(",")  # there can be multiple synonyms for single class
                res.append(x_)
            return res

        # get text classifier class name lists
        try:
            class_names = split_labels(metadata.stuff_classes)  # includes thing+stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except Exception:
            # for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)

        train_class_names_flat = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names_flat).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=""):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        num_templates: List[int] = []
        templated_class_names: List[str] = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)  # how many templates for current classes
        class_names_flat = templated_class_names
        return category_overlapping_mask, num_templates, class_names_flat

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        (
            self.category_overlapping_mask,
            self.test_num_templates,
            self.test_class_names,
        ) = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier_chunks = []
                # avoid OOM when num classes is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier_chunks.append(
                        self.backbone.get_text_classifier(
                            self.train_class_names[idx : idx + bs], self.device
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier_chunks, dim=0)

                # average across templates and normalize
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT),
                    len(VILD_PROMPT),
                    text_classifier.shape[-1],
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier_chunks = []
                # avoid OOM when num classes is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier_chunks.append(
                        self.backbone.get_text_classifier(
                            self.test_class_names[idx : idx + bs], self.device
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier_chunks, dim=0)

                # average across templates and normalize
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT),
                    len(VILD_PROMPT),
                    text_classifier.shape[-1],
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        # DreamMask config (memory bank & alignment loss)
        memory_bank_size = cfg.MODEL.DREAMMASK.MEMORY_BANK_SIZE
        lambda_sra = cfg.MODEL.DREAMMASK.LAMBDA_SRA

        # for test metadata: fall back to train if TEST is empty (e.g. synthetic-only training)
        if len(cfg.DATASETS.TEST) > 0:
            test_meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        else:
            test_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": test_meta,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            # DreamMask synthetic–real alignment
            "memory_bank_size": memory_bank_size,
            "lambda_sra": lambda_sra,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Optional "is_synthetic": bool flag, True if sample is synthetic.
                   * Other information: "height", "width" used in inference.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat(
            [text_classifier, F.normalize(self.void_embedding.weight, dim=-1)],
            dim=0,
        )
        features["text_classifier"] = text_classifier
        features["num_templates"] = num_templates
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification targets
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                gt_instances = None
                targets = None

            # 1) Original Mask2Former / FC-CLIP losses
            raw_losses = self.criterion(outputs, targets)
            losses: Dict[str, torch.Tensor] = {}
            for k, v in raw_losses.items():
                if k in self.criterion.weight_dict:
                    losses[k] = v * self.criterion.weight_dict[k]

            # 2) DreamMask synthetic–real alignment loss L_sra
            if gt_instances is not None and "clip_vis_dense" in features:
                clip_dense = features["clip_vis_dense"]
                loss_sra = self.compute_sra_loss(
                    clip_dense=clip_dense,
                    gt_instances=gt_instances,
                    batched_inputs=batched_inputs,
                )
                if loss_sra is not None:
                    losses["loss_sra"] = loss_sra * self.lambda_sra

            return losses

        # ----------------------- inference -----------------------
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # We ensemble the pred logits of in-vocab and out-vocab
        clip_feature = features["clip_vis_dense"]
        mask_for_pooling = F.interpolate(
            mask_pred_results,
            size=clip_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if "convnext" in self.backbone.model_name.lower():
            pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
            pooled_clip_feature = self.backbone.visual_prediction_forward(
                pooled_clip_feature
            )
        elif "rn" in self.backbone.model_name.lower():
            pooled_clip_feature = self.backbone.visual_prediction_forward(
                clip_feature, mask_for_pooling
            )
        else:
            raise NotImplementedError

        out_vocab_cls_results = get_classification_logits(
            pooled_clip_feature,
            text_classifier,
            self.backbone.clip_model.logit_scale,
            num_templates,
        )
        in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
        out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

        # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
        out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
        in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
        category_overlapping_mask = self.category_overlapping_mask.to(self.device)

        if self.ensemble_on_valid_mask:
            # Only include out_vocab cls results on masks with valid pixels
            valid_masking = (
                (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
            )
            valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
            alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
            beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
            alpha = alpha * valid_masking
            beta = beta * valid_masking
        else:
            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta

        cls_logits_seen = (
            (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha).log()
            * category_overlapping_mask
        )
        cls_logits_unseen = (
            (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta).log()
            * (1 - category_overlapping_mask)
        )
        cls_results = cls_logits_seen + cls_logits_unseen

        # Used for filtering void predictions.
        is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
        mask_cls_probs = torch.cat(
            [cls_results.softmax(-1) * (1.0 - is_void_prob), is_void_prob],
            dim=-1,
        )
        mask_cls_results = torch.log(mask_cls_probs + 1e-8)

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results: List[Dict[str, Any]] = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["instances"] = instance_r

        return processed_results

    # ------------------------------------------------------------------
    # DreamMask synthetic–real alignment helpers
    # ------------------------------------------------------------------
    def update_real_memory_bank(self, class_id: int, feat: torch.Tensor) -> None:
        """
        Update real feature memory bank for class_id with one new feature feat.
        feat is expected to be L2-normalized and no grad.
        """
        if class_id < 0 or class_id >= self.real_memory_bank.shape[0]:
            return
        with torch.no_grad():
            ptr = int(self.real_memory_ptr[class_id].item())
            self.real_memory_bank[class_id, ptr] = feat
            ptr = (ptr + 1) % self.memory_bank_size
            self.real_memory_ptr[class_id] = ptr

    def get_real_prototype(self, class_id: int) -> Optional[torch.Tensor]:
        """
        Get prototype M_r^p for class_id by averaging non-zero vectors in its memory bank.
        Returns None if no valid real feature has been stored for this class.
        """
        if class_id < 0 or class_id >= self.real_memory_bank.shape[0]:
            return None
        bank = self.real_memory_bank[class_id]  # [β, C]
        norms = bank.norm(dim=1)
        valid = norms > 0
        if not valid.any():
            return None
        proto = bank[valid].mean(dim=0)
        proto = F.normalize(proto, dim=0)
        return proto

    def compute_sra_loss(
        self,
        clip_dense: torch.Tensor,
        gt_instances: List[Instances],
        batched_inputs: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """
        Compute synthetic–real alignment loss L_sra over a mini-batch.

        For each image:
          - If is_synthetic == False: treat as real, update memory bank only.
          - If is_synthetic == True: treat as synthetic, compute
                L_sra = 1 - cos(F_s^p, M_r^p)
            where F_s^p is pooled CLIP dense feature and M_r^p is prototype from
            real memory bank for the same class p.
        """
        B, C, Hc, Wc = clip_dense.shape
        total_loss = clip_dense.new_tensor(0.0)
        count = 0

        for i in range(B):
            instances: Instances = gt_instances[i]
            if not hasattr(instances, "gt_masks"):
                continue
            if instances.gt_masks.tensor.numel() == 0:
                continue

            is_synthetic = bool(batched_inputs[i].get("is_synthetic", False))
            gt_masks = instances.gt_masks.tensor.float()  # [N, H, W]
            gt_classes = instances.gt_classes             # [N]

            # resize GT masks to CLIP feature resolution
            masks = F.interpolate(
                gt_masks.unsqueeze(1),
                size=(Hc, Wc),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [N, Hc, Wc]

            feat_map = clip_dense[i]                # [C, Hc, Wc]
            feat_flat = feat_map.view(C, -1)        # [C, Hc*Wc]

            for k in range(masks.shape[0]):
                m = masks[k]                       # [Hc, Wc]
                area = m.sum()
                if area < 1.0:
                    continue

                weights = (m.view(-1) / (area + 1e-6)).unsqueeze(0)  # [1, Hc*Wc]
                obj_feat = (feat_flat * weights).sum(dim=1)          # [C]
                obj_feat = F.normalize(obj_feat, dim=0)

                cls_id = int(gt_classes[k].item())

                if is_synthetic:
                    # compute L_sra for synthetic objects
                    proto = self.get_real_prototype(cls_id)
                    if proto is None:
                        continue
                    cos_sim = (obj_feat * proto).sum()
                    total_loss = total_loss + (1.0 - cos_sim)
                    count += 1
                else:
                    # update memory bank for real objects (no grad)
                    self.update_real_memory_bank(cls_id, obj_feat.detach())

        if count == 0:
            return None
        return total_loss / float(count)

    # ------------------------------------------------------------------
    # Original helper methods (targets & inference)
    # ------------------------------------------------------------------
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros(
            (h, w), dtype=torch.int32, device=cur_masks.device
        )
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list: Dict[int, int] = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = (
                                current_segment_id + 1
                            )

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = (
            torch.arange(num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = (
                    lab
                    in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                )

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result