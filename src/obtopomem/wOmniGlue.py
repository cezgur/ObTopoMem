# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import triton # Workaround for TensorFlow 2.20 / Triton 3.7 import conflict.
import torch
import tensorflow as tf
import numpy as np
import logging

from typing import Tuple

from omniglue.omniglue_extract import OmniGlue
from omniglue import utils as omniglue_utils

from .paths import OMNIGLUE_MODEL_DIR

logger = logging.getLogger(__name__)


class wOmniGlue:
    """
    A class wrapping OmniGlue for easier access.
    """
    omniglueModelExports = {
        "og_export": OMNIGLUE_MODEL_DIR / "og_export",
        "sp_export": OMNIGLUE_MODEL_DIR / "sp_v6",
        "dino_export": OMNIGLUE_MODEL_DIR / "dinov2_vitb14_pretrain.pth"
    }

    DINO_FEATURE_DIM = 768
    DINO_PATCH_SIZE = 14
    MATCH_THRESHOLD = 0.5
    def __init__(self):
        # Configure TensorFlow to not pre-allocate all GPU memory.
        # This is crucial for allowing PyTorch (used by DINO) to share the VRAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")

        self.og = OmniGlue(
            og_export = str(self.omniglueModelExports["og_export"]),
            sp_export = str(self.omniglueModelExports["sp_export"]),
            dino_export = str(self.omniglueModelExports["dino_export"])
        )

    @property
    def device(self) -> str:
        """Returns the device that PyTorch will use."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _get_dino_processed_shape(original_height: int, original_width: int) -> Tuple[int, int]:
        """
        Calculate the image dimensions after DINO preprocessing,
        accounting for the maximum image size and patch alignment.
        """
        h_image, w_image = original_height, original_width
        max_size = 630
        patch_size = wOmniGlue.DINO_PATCH_SIZE

        # Match the resizing performed by the DINO extractor.
        if max(h_image, w_image) > max_size:
            if h_image > w_image:
                w_image = int(max_size * w_image / h_image)
                h_image = max_size
            else:
                h_image = int(max_size * h_image / w_image)
                w_image = max_size

        # Round both dimensions down to multiples of the patch size
        h_resize = (h_image // patch_size) * patch_size
        w_resize = (w_image // patch_size) * patch_size

        return h_resize, w_resize

    def getSPFeatures(self, frame: np.ndarray, num_features: int = 1024) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts SuperPoint features from a single frame.

        Args:
            frame: The input image as a NumPy array (H, W, 3).
            num_features: The maximum number of keypoints to extract. If 0,
                          all detected keypoints are kept.

        Returns:
            A tuple containing:
                - keypoints (np.ndarray): (N, 2) array of (x, y) coordinates.
                - descriptors (np.ndarray): (N, 256) array of SuperPoint descriptors.
                - scores (np.ndarray): (N,) array of keypoint confidence scores.
        """
        keypoints, descriptors, scores = self.og.sp_extract(frame, num_features=num_features)
        # Check for NaNs in SuperPoint outputs and replace them with zeros.
        if np.any(np.isnan(keypoints)) or np.any(np.isnan(descriptors)) or np.any(np.isnan(scores)):
            logger.warning(f"NaN values detected in SuperPoint features. Keypoints: {np.isnan(keypoints).sum()}, Descriptors: {np.isnan(descriptors).sum()}, Scores: {np.isnan(scores).sum()}. Replacing with zeros.")
            keypoints = np.nan_to_num(keypoints)
            descriptors = np.nan_to_num(descriptors)
            scores = np.nan_to_num(scores)
        return keypoints, descriptors, scores

    def getDINOFeatures(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts dense DINOv2 features from a given frame.

        Args:
            frame (np.ndarray): The input frame as a NumPy array.

        Returns:
            np.ndarray: A NumPy array containing the dense DINO features.
        """
        # Ensure the input frame is float32 for consistency with TF operations
        # and to prevent potential type-related issues in the underlying model.
        frame_float = frame.astype(np.float32)

        dino_features = self.og.dino_extract(frame_float)
        if np.any(np.isnan(dino_features)):
            nan_count = np.isnan(dino_features).sum()
            total_count = dino_features.size
            logger.warning(f"DINO features contain {nan_count}/{total_count} NaN values.")
        return dino_features

    def getDINODescriptors(self,
                           dinoFeatures: np.ndarray,
                           sp_features: Tuple[np.ndarray, np.ndarray, np.ndarray],
                           height: int,
                           width: int) -> np.ndarray:
        """
        Looks up DINO descriptors for given SuperPoint keypoint locations using a
        performant, vectorized implementation.

        This method overrides the slow, iterative approach from the original
        OmniGlue library by performing bilinear interpolation for all keypoints
        at once using TensorFlow operations.

        Args:
            dinoFeatures: The dense DINO feature map for the frame.
            sp_features: The SuperPoint features tuple, where sp_features[0] are the keypoints.
            height: The original height of the frame.
            width: The original width of the frame.

        Returns:
            An (N, D) NumPy array of DINO descriptors corresponding to the N keypoints.
        """
        keypoints = tf.convert_to_tensor(sp_features[0], dtype=tf.float32)

        # 1. Determine the shape of the feature map DINO actually processed.
        h_resize, w_resize = self._get_dino_processed_shape(height, width)

        if h_resize == 0 or w_resize == 0:
            # If the image is too small, return empty descriptors
            num_kps = sp_features[0].shape[0]
            return np.zeros((num_kps, self.DINO_FEATURE_DIM), dtype=np.float32)

        h_feat, w_feat = h_resize // self.DINO_PATCH_SIZE, w_resize // self.DINO_PATCH_SIZE

        # 2. Reshape the 1D DINO features into a 3D feature map.
        expected_elements = h_feat * w_feat * self.DINO_FEATURE_DIM
        if dinoFeatures.size != expected_elements:
            raise ValueError(
                f"DINO feature shape mismatch. Expected {expected_elements} elements "
                f"({h_feat}*{w_feat}*{self.DINO_FEATURE_DIM}), but got {dinoFeatures.shape[0] * dinoFeatures.shape[1]} "
                f"from input shape {dinoFeatures.shape}. This indicates an upstream issue."
            )
        dino_features_map = tf.reshape(
            dinoFeatures, (h_feat, w_feat, self.DINO_FEATURE_DIM)
        )
        if tf.reduce_any(tf.math.is_nan(dino_features_map)):
            logger.warning("NaN values detected in reshaped DINO feature map.")
            # Optionally, replace NaNs with zeros or a small value here if they are expected
            # dino_features_map = tf.where(tf.math.is_nan(dino_features_map), tf.zeros_like(dino_features_map), dino_features_map)

        # 3. Normalize keypoints from original image coordinates to feature map coordinates.
        # Note: keypoints are (x, y), so we scale by (width, height).
        img_size = tf.cast(tf.stack([width, height]), dtype=tf.float32)
        feat_size = tf.cast(tf.stack([w_feat, h_feat]), dtype=tf.float32)
        keypoints_feat_normalized = keypoints / img_size * feat_size

        if tf.reduce_any(tf.math.is_nan(keypoints_feat_normalized)):
            logger.warning("NaN values detected in normalized keypoints.")
            # This could happen if img_size is zero or contains NaNs, which it shouldn't.

        # 4. Perform vectorized bilinear interpolation. The key is to use unclipped
        # coordinates for weight calculation and clipped coordinates for gathering.
        x = tf.clip_by_value(keypoints_feat_normalized[:, 0], 0.0, tf.cast(w_feat - 1, tf.float32))
        y = tf.clip_by_value(keypoints_feat_normalized[:, 1], 0.0, tf.cast(h_feat - 1, tf.float32))

        # Calculate floor and ceil for interpolation points.
        x0, y0 = tf.floor(x), tf.floor(y)
        x1_unc, y1_unc = x0 + 1, y0 + 1 # Unclipped ceil coordinates for weight calculation

        # Create clipped integer indices for safe gathering from the feature map.
        x0_c = tf.cast(x0, tf.int32)
        y0_c = tf.cast(y0, tf.int32)
        x1_c = tf.cast(tf.clip_by_value(x1_unc, 0, w_feat - 1), tf.int32)
        y1_c = tf.cast(tf.clip_by_value(y1_unc, 0, h_feat - 1), tf.int32)

        # Gather descriptor values from the four corners for all keypoints at once.
        Ia = tf.gather_nd(dino_features_map, tf.stack([y0_c, x0_c], axis=1))
        Ib = tf.gather_nd(dino_features_map, tf.stack([y1_c, x0_c], axis=1))
        Ic = tf.gather_nd(dino_features_map, tf.stack([y0_c, x1_c], axis=1))
        Id = tf.gather_nd(dino_features_map, tf.stack([y1_c, x1_c], axis=1))

        # Check for NaNs in gathered values
        if tf.reduce_any(tf.math.is_nan(Ia)) or tf.reduce_any(tf.math.is_nan(Ib)) or \
           tf.reduce_any(tf.math.is_nan(Ic)) or tf.reduce_any(tf.math.is_nan(Id)):
            logger.warning("NaN values detected in gathered DINO feature map values (Ia, Ib, Ic, Id).")

        # Calculate weights using the original (unclipped) fractional coordinates.
        wa = (x1_unc - x)[:, None] * (y1_unc - y)[:, None]
        wb = (x1_unc - x)[:, None] * (y - y0)[:, None]
        wc = (x - x0)[:, None] * (y1_unc - y)[:, None]
        wd = (x - x0)[:, None] * (y - y0)[:, None]

        interpolated_descriptors = wa * Ia + wb * Ib + wc * Ic + wd * Id

        if tf.reduce_any(tf.math.is_nan(interpolated_descriptors)):
            logger.warning("NaN values detected in interpolated_descriptors before numpy conversion.")

        result_numpy = interpolated_descriptors.numpy()
        if np.any(np.isnan(result_numpy)):
            nan_count = np.isnan(result_numpy).sum()
            logger.warning(f"NaN values detected in final DINO descriptors after interpolation. Count: {nan_count} / {result_numpy.size}. Replacing with zeros.")
            # Replace any remaining NaNs as a final safeguard.
            result_numpy = np.nan_to_num(result_numpy)
        return result_numpy

    def findMatches(
        self,
        sp_features0: Tuple[np.ndarray, np.ndarray, np.ndarray],
        dino_descriptors0: np.ndarray,
        height0: int,
        width0: int,
        sp_features1: Tuple[np.ndarray, np.ndarray, np.ndarray],
        dino_descriptors1: np.ndarray,
        height1: int,
        width1: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds matches between two sets of pre-computed features.

        This method takes raw feature data as input, runs the OmniGlue matching
        model in both directions, combines their soft assignment probabilities
        via element-wise multiplication, and then returns the raw match data.

        Args:
            sp_features0: SuperPoint features for the first image (keypoints, descriptors, scores).
            dino_descriptors0: DINO descriptors for the first set of keypoints.
            height0: Height of the first image.
            width0: Width of the first image.
            sp_features1: SuperPoint features for the second image.
            dino_descriptors1: DINO descriptors for the second set of keypoints.
            height1: Height of the second image.
            width1: Width of the second image.

        Returns:
            A tuple containing:
                - match_indices (np.ndarray): An (N, 2) array where each row [i, j]
                  indicates a match between keypoint i from the first set and
                  keypoint j from the second set.
                - match_confidences (np.ndarray): An (N,) array of confidence scores
                  for each match.
        """

        logger.debug(f"SPFeatures0 shape: {sp_features0[0].shape}, DINODescriptors0 shape: {dino_descriptors0.shape}")
        logger.debug(f"SPFeatures1 shape: {sp_features1[0].shape}, DINODescriptors1 shape: {dino_descriptors1.shape}")

        if not all([sp_features0[0].shape[0] > 0, sp_features1[0].shape[0] > 0, dino_descriptors0.shape[0] > 0, dino_descriptors1.shape[0] > 0]):
            logger.warning("One or more feature sets are empty. Skipping matching.")
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)

        # Explicitly check for NaNs in the inputs to the matcher
        if np.isnan(dino_descriptors0).any() or np.isnan(dino_descriptors1).any():
            logger.error("NaN values detected in input descriptors. Aborting matching for this pair.")
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)

        # Handle cases where one or both frames have no keypoints
        if sp_features0[0].shape[0] == 0 or sp_features1[0].shape[0] == 0:
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)

        # Handle cases where either frame has exactly one keypoints.
        if sp_features0[0].shape[0] == 1 or sp_features1[0].shape[0] == 1:
            logger.error(
                f"OmniGlue matching does not support singleton keypoint sets. "
                f"Received {sp_features0[0].shape[0]} and {sp_features1[0].shape[0]} keypoints."
            )
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)

        # 1. Construct inputs for the OmniGlue matcher (frame0 -> frame1)
        inputs_fwd = self.og._construct_inputs(
            width0, height0, width1, height1, sp_features0, sp_features1,
            tf.convert_to_tensor(dino_descriptors0, dtype=tf.float32),
            tf.convert_to_tensor(dino_descriptors1, dtype=tf.float32)
        )
        # Debug: Check for NaNs in inputs to the forward matcher
        for k, v in inputs_fwd.items():
            # Only check for NaNs if the tensor is a floating-point type
            if tf.is_tensor(v) and v.dtype.is_floating:
                if tf.reduce_any(tf.math.is_nan(v)):
                    logger.error(f"NaN detected in input tensor '{k}' for forward pass before matcher call.")
                # Optionally, print more details about the problematic tensor
                # logger.error(f"Tensor '{k}' shape: {v.shape}, sample: {v.numpy().flatten()[:5]}")

        # 2. Construct inputs for the OmniGlue matcher (frame1 -> frame0)
        inputs_bwd = self.og._construct_inputs(
            width1, height1, width0, height0, sp_features1, sp_features0,
            tf.convert_to_tensor(dino_descriptors1, dtype=tf.float32),
            tf.convert_to_tensor(dino_descriptors0, dtype=tf.float32)
        )
        # Debug: Check for NaNs in inputs to the backward matcher
        for k, v in inputs_bwd.items(): # type: ignore
            # Only check for NaNs if the tensor is a floating-point type
            if tf.is_tensor(v) and v.dtype.is_floating:
                if tf.reduce_any(tf.math.is_nan(v)):
                    logger.error(f"NaN detected in input tensor '{k}' for backward pass before matcher call.")
                # logger.error(f"Tensor '{k}' shape: {v.shape}, sample: {v.numpy().flatten()[:5]}")

        # 3. Perform OmniGlue matching inference (forward: frame0 -> frame1)
        og_outputs_fwd = self.og.matcher.signatures['serving_default'](**inputs_fwd) # type: ignore
        soft_assignment_fwd = og_outputs_fwd['soft_assignment'][:, :-1, :-1] # Shape (1, N0, N1)

        # Replace NaN values in forward soft assignment with zero
        if tf.reduce_any(tf.math.is_nan(soft_assignment_fwd)):
            nan_mask = tf.math.is_nan(soft_assignment_fwd)

            nan_count = tf.reduce_sum(tf.cast(nan_mask, tf.int32)).numpy()
            not_nan_count = tf.reduce_sum(tf.cast(tf.math.logical_not(nan_mask), tf.int32)).numpy()
            logger.debug(f"NaN values detected in forward soft assignment. Replacing with zeros. Count: {nan_count} / {not_nan_count}")
            soft_assignment_fwd = tf.where(tf.math.is_nan(soft_assignment_fwd), tf.zeros_like(soft_assignment_fwd), soft_assignment_fwd)

        # Log forward pass confidences
        fwd_confidences = soft_assignment_fwd.numpy().flatten()
        if fwd_confidences.size > 0:
            logger.debug(f"Forward pass (0->1) - Max Confidence: {np.max(fwd_confidences):.4f}, Avg Confidence: {np.mean(fwd_confidences):.4f}")
        else:
            logger.debug("Forward pass (0->1) - No confidences to report (empty input).")

        # 4. Perform OmniGlue matching inference (backward: frame1 -> frame0)
        og_outputs_bwd = self.og.matcher.signatures['serving_default'](**inputs_bwd) # type: ignore
        soft_assignment_bwd = og_outputs_bwd['soft_assignment'][:, :-1, :-1] # Shape (1, N1, N0)

        # Replace NaN values in backward soft assignment with zero
        if tf.reduce_any(tf.math.is_nan(soft_assignment_bwd)):
            nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(soft_assignment_bwd), tf.int32)).numpy()
            logger.debug(f"NaN values detected in backward soft assignment. Replacing with zeros. Count: {nan_count}")
            soft_assignment_bwd = tf.where(tf.math.is_nan(soft_assignment_bwd), tf.zeros_like(soft_assignment_bwd), soft_assignment_bwd)

        # Log backward pass confidences
        bwd_confidences = soft_assignment_bwd.numpy().flatten()
        if bwd_confidences.size > 0:
            logger.debug(f"Backward pass (1->0) - Max Confidence: {np.max(bwd_confidences):.4f}, Avg Confidence: {np.mean(bwd_confidences):.4f}")
        else:
            logger.debug("Backward pass (1->0) - No confidences to report (empty input).")

        # 5. Apply element-wise multiplication for bidirectional consistency
        # Transpose soft_assignment_bwd to match shape of soft_assignment_fwd (1, N0, N1)
        soft_assignment_bwd_transposed = tf.transpose(soft_assignment_bwd, perm=[0, 2, 1]) # Shape (1, N0, N1)

        # Element-wise multiplication of probabilities
        combined_soft_assignment = soft_assignment_fwd * soft_assignment_bwd_transposed

        # ---------------------------------------------------------
        # DIAGNOSTIC: detect when NO potential matches exist
        # ---------------------------------------------------------
        threshold = self.MATCH_THRESHOLD ** 2
        combined_np = combined_soft_assignment.numpy().squeeze()

        if combined_np.ndim == 0:
            logger.debug(">>> [DEBUG] combined_soft_assignment is scalar. This will cause NO MATCHES.")
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)
        elif combined_np.ndim == 1:
            logger.debug(">>> [DEBUG] combined_soft_assignment is 1-D. Shape:", combined_np.shape)
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)
        elif combined_np.ndim == 2:
            N0, N1 = combined_np.shape
            above = (combined_np >= threshold)
            total_above = int(above.sum())

            if total_above == 0:
                logger.debug(f">>> [DEBUG] No matches possible: combined_soft_assignment has "
                             f"0 entries >= threshold ({threshold}).")
                return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)
            else:
                logger.debug(f">>> [DEBUG] Potential matches before TF: {total_above}")

        else:
            logger.debug(">>> [DEBUG] Unexpected combined_soft_assignment shape:", combined_np.shape)
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)

        # ---------------------------------------------------------

        # 6. Generate match matrix using the combined soft assignment
        match_matrix = (
            omniglue_utils.soft_assignment_to_match_matrix(
                combined_soft_assignment, self.MATCH_THRESHOLD ** 2
            ).numpy().squeeze()
        )

        # 7. Filter out matches with 0.0 confidence keypoints using vectorization
        match_indices = np.argwhere(match_matrix)
        if match_indices.shape[0] == 0:
            return np.empty((0,2), dtype=int), np.empty((0,), dtype=np.float32)

        # As an additional safeguard, filter out any matches that involve a keypoint
        # with a score of 0.0, which might indicate a padding or invalid keypoint.
        # The primary filtering is done via MATCH_THRESHOLD on the match confidence.
        keep_mask = (sp_features0[2][match_indices[:, 0]] > 0.0) & \
                    (sp_features1[2][match_indices[:, 1]] > 0.0)

        valid_match_indices = match_indices[keep_mask]

        # 8. Get the confidence scores for the valid matches from the combined soft assignment
        combined_soft_assignment_np = combined_soft_assignment.numpy().squeeze()
        match_confidences = combined_soft_assignment_np[valid_match_indices[:, 0], valid_match_indices[:, 1]]

        # Log final combined match confidences
        if match_confidences.size > 0:
            logger.debug(f"Combined Matches - Max Confidence: {np.max(match_confidences):.4f}, Avg Confidence: {np.mean(match_confidences):.4f}. Total matches: {match_confidences.size}")
        else:
            logger.debug("Combined Matches - No valid matches found.")

        return valid_match_indices, match_confidences

    @staticmethod
    def get_dino_patch_size_in_image_coords(
        original_height: int,
        original_width: int
    ) -> Tuple[float, float]:
        """
        Calculates the size of a DINO patch in the original image's pixel coordinates.

        This is necessary because DINO processes a resized version of the image. This
        method replicates the resizing logic from DINOExtract to determine the scaling
        factor and projects the patch size back to the original image's coordinate system.

        Args:
            original_height: The height of the original image.
            original_width: The width of the original image.

        Returns:
            A tuple (patch_height_in_img_coords, patch_width_in_img_coords) representing
            the DINO patch dimensions in the original image's pixel space.
        """
        h_resize, w_resize = wOmniGlue._get_dino_processed_shape(original_height, original_width)
        dino_patch_size = wOmniGlue.DINO_PATCH_SIZE

        if h_resize == 0 or w_resize == 0:
            return (0.0, 0.0)

        patch_height_in_img_coords = (dino_patch_size / h_resize) * original_height
        patch_width_in_img_coords = (dino_patch_size / w_resize) * original_width

        return (patch_height_in_img_coords, patch_width_in_img_coords)

    @staticmethod
    def get_dino_feature_map_shape(
        original_height: int,
        original_width: int
    ) -> Tuple[int, int]:
        """
        Calculates the DINO feature map dimensions for a given original image size.

        This provides a direct way to know the output shape of the DINO feature
        extraction without running a dummy forward pass.

        Args:
            original_height: The height of the original image.
            original_width: The width of the original image.

        Returns:
            A tuple (feature_map_height, feature_map_width).
        """
        h_resize, w_resize = wOmniGlue._get_dino_processed_shape(original_height, original_width)
        dino_patch_size = wOmniGlue.DINO_PATCH_SIZE

        if h_resize == 0 or w_resize == 0:
            return (0, 0)

        h_feat = h_resize // dino_patch_size
        w_feat = w_resize // dino_patch_size

        return (h_feat, w_feat)
