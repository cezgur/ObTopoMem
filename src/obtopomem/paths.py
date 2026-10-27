# Copyright 2026 Ata Baran Fırat
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

# Root directory of the ObTopoMem repository
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model directories
MODEL_DIR = PROJECT_ROOT / "models"

OMNIGLUE_MODEL_DIR = MODEL_DIR / "omniglue"
SAM_MODEL_DIR = MODEL_DIR / "sam"
