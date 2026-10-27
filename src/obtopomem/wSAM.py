# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
import os
import logging

from typing import Any, Optional, List, Dict

from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)

from .paths import SAM_MODEL_DIR


logger = logging.getLogger(__name__)

class wSAM:
    """
    A wrapper class for the Segment Anything Model (SAM) to simplify its usage.
    """

    # Define model types and their corresponding checkpoint paths.
    # Note: The user must ensure these checkpoints are available at the specified paths.

    samModelExports = {
        "vit_h": SAM_MODEL_DIR / "sam_vit_h_4b8939.pth",
        "vit_l": SAM_MODEL_DIR / "sam_vit_l_0b3195.pth",
        "vit_b": SAM_MODEL_DIR / "sam_vit_b_01ec64.pth"
    }

    DEFAULT_MODEL_TYPE = "vit_l"

    def __init__(self,
                 model_type: str = DEFAULT_MODEL_TYPE,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95,
                 box_nms_thresh: float = 0.7,
                 crop_nms_thresh: float = 0.7
                ):
        """
        Initializes the wSAM wrapper.

        Args:
            model_type: The type of SAM model to load (e.g., "vit_h", "vit_l", "vit_b").
            pred_iou_thresh: A filtering threshold for predicted mask quality. Masks with a predicted IoU below this threshold are discarded.
            stability_score_thresh: A filtering threshold for mask stability. Masks with a stability score below this threshold are discarded.
            box_nms_thresh: The box IoU cutoff used by non-maximal suppression to filter duplicate masks within a single crop. Lowering this value will reduce overlapping masks.
            crop_nms_thresh: The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops. Lowering this value will reduce overlapping masks.
        """
        if model_type not in self.samModelExports:
            raise ValueError(f"Unknown model type: {model_type}. Available types are: {list(self.samModelExports.keys())}")

        self.model_type = model_type

        self.checkpoint_path = str(self.samModelExports[model_type])

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found at: {self.checkpoint_path}")

        logger.info(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")

        # Load the SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)

        # Initialize the automatic mask generator with configurable parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_nms_thresh=crop_nms_thresh,
        )

        self.predictor = SamPredictor(self.sam)

        logger.info(f"SAM model '{self.model_type}' loaded successfully on device '{self.device}'.")
        logger.info(f"SamAutomaticMaskGenerator initialized with: pred_iou_thresh={pred_iou_thresh}, stability_score_thresh={stability_score_thresh}, box_nms_thresh={box_nms_thresh}, crop_nms_thresh={crop_nms_thresh}")

    @property
    def device(self) -> torch.device:
        """Returns the device that PyTorch will use."""
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def segment_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Generates segmentation masks for a single input frame using the automatic mask generator.

        The input frame should be in RGB format.

        Args:
            frame: The input image as a NumPy array (H, W, 3) in RGB format.

        Returns:
            A list of masks, where each mask is a dictionary containing:
                - 'segmentation': The binary mask as a NumPy array.
                - 'area': The area of the mask in pixels.
                - 'bbox': The bounding box of the mask in XYWH format.
                - 'predicted_iou': The model's own prediction of the mask's quality.
                - 'point_coords': The point prompts used to generate the mask.
                - 'stability_score': A measure of the mask's quality.
                - 'crop_box': The crop of the image used to generate the mask in XYWH format.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be an HxWx3 RGB image.")

        logger.debug(f"Generating masks for frame with shape: {frame.shape}")

        # The mask generator expects an RGB image (uint8).
        if frame.dtype != np.uint8:
            logger.warning(f"Frame dtype is {frame.dtype}, converting to uint8. SAM expects uint8 RGB images.")
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        masks = self.mask_generator.generate(frame)

        logger.debug(f"Generated {len(masks)} masks for the frame.")

        return masks

    def segment_frame_with_prompts(
        self,
        frame: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generates masks for a single frame using specific prompts (points, box).
        If a frame is provided, it will be set in the predictor. If the frame is None,
        it will use the previously set frame.

        Args:
            frame: The input image as a NumPy array (H, W, 3) in RGB format.
            points: A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
            point_labels: A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.
            box: A length 4 array given a box prompt to the model, in XYXY format.
            multimask_output: If true, the model will return multiple masks.

        Returns:
            A list of masks, where each mask is a dictionary containing:
                - 'segmentation': The binary mask as a NumPy array.
                - 'predicted_iou': The model's own prediction of the mask's quality.
                - 'logits': The low resolution logits for the mask.
        """
        if frame is not None:
            if not isinstance(frame, np.ndarray):
                raise TypeError("Input frame must be a NumPy array.")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Input frame must be an HxWx3 RGB image.")
            logger.debug(f"Setting image for predictor with frame shape: {frame.shape}")
            self.predictor.set_image(frame)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )

        # logger.debug(f"Generated {len(masks)} masks with prompts.")

        results = []
        for i in range(len(masks)):
            results.append({
                "segmentation": masks[i],
                "predicted_iou": scores[i],
                "logits": logits[i],
            })

        return results

    def segment_frame_for_multiple_prompts(
        self,
        frame: np.ndarray,
        prompts_list: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Generates the best mask for each set of prompts in a list, for a single frame.
        This method is efficient for segmenting multiple objects in the same image,
        as it computes the image embedding only once.

        Args:
            frame: The input image as a NumPy array (H, W, 3) in RGB format.
            prompts_list: A list of dictionaries. Each dictionary represents one object to segment
                          and should contain prompt keys like 'points', 'point_labels', or 'box'.

        Returns:
            A list of dictionaries, where each dictionary is the best mask data
            (segmentation, predicted_iou, etc.) for a corresponding prompt set.
        """
        self.predictor.set_image(frame)

        all_best_masks = []
        for i, prompts in enumerate(prompts_list):
            masks_data = self.segment_frame_with_prompts(
                frame=None,  # Important: Pass None to avoid re-setting the image
                points=prompts.get("points"),
                point_labels=prompts.get("point_labels"),
                box=prompts.get("box"),
                multimask_output=True
            )

            if not masks_data:
                # [TODO]
                # Preserve prompt-mask alignment by failing explicitely.
                # if SAM unexpectedly returns no masks.
                raise RuntimeError(
                    f"SAM returned no masks for prompt set {i}."
                )

            # Select the best mask for this set of prompts
            best_mask_data = max(masks_data, key=lambda x: x['predicted_iou'])
            all_best_masks.append(best_mask_data)

        return all_best_masks
