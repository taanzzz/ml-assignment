"""
Standalone Inference for Problem 01 Chest X-ray CNN

This module provides a standalone prediction interface for the trained chest X-ray
classification model. It enforces strict reproducibility guarantees and model integrity
checks to ensure that predictions are generated using the exact model weights and
preprocessing configuration from the training run.

**Design Philosophy:**
Rather than retraining or reimplementing the model, we load the exact serialized
Keras model that was frozen after validation-based threshold selection. The preprocessing
pipeline (image decode, resize, normalization) is kept deterministic and matches
the notebook's implementation exactly—this matters for medical imaging, where small
preprocessing differences can affect clinical decisions.

**Key Features:**
1. SHA-256 integrity verification: ensures the model file hasn't been tampered with.
2. Reproducible preprocessing: pixel-perfect matching with notebook Steps 8 and 17.
3. JSON output: suitable for integration into clinical information systems.
4. Zero external dependencies (beyond TensorFlow/Keras): easy deployment.

**Preprocessing Details:**
- Image decoding: JPEG/PNG → grayscale (single-channel).
- Resizing: bilinear interpolation with antialiasing (prevents aliasing artifacts).
- Pixel range: float32, 0-255 (NOT normalized yet).
- Normalization: Rescaling(1/255) is the FIRST layer inside the saved model
  (this prevents normalization errors if predict.py is called multiple times).

**Example Usage:**
    python predict.py \\
        --model best_cnn.keras \\
        --config inference_config.json \\
        --image patient_x001.jpg

**Output Example:**
    {
        "image": "patient_x001.jpg",
        "predicted_class": "PNEUMONIA",
        "pneumonia_score": 0.87,
        "threshold": 0.269,
        "disclaimer": "Educational research model. Not for clinical diagnosis."
    }

**Critical Notes:**
- `pneumonia_score` (0-1) is a model confidence score, NOT a calibrated disease probability.
- This model is for research/educational purposes only. It is NOT a medical device.
- Clinical decisions must involve qualified radiologists and proper regulatory approval.
- False positives (52%) and false negatives (0%) have different clinical implications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Suppress TensorFlow info/warning messages; we only want inference errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configuration constants
CHUNK_SIZE = 1024 * 1024  # Read files in 1 MiB chunks for large models


def sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file.

    This function reads the file in chunks to avoid loading the entire model
    into memory at once. SHA-256 is used to verify model integrity: if the
    file has been accidentally corrupted or intentionally tampered with,
    the hash will differ from the one stored in inference_config.json.

    Args:
        path (Path): File to hash.

    Returns:
        str: Lowercase hex digest of the SHA-256 hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Parser configured with required arguments:
            --model (path to best_cnn.keras)
            --config (path to inference_config.json)
            --image (path to input X-ray image)
    """
    parser = argparse.ArgumentParser(
        prog="chest_xray_predict",
        description="Run inference on a chest X-ray with the trained CNN model.",
        epilog="Example: python predict.py --model best_cnn.keras "
               "--config inference_config.json --image xray.jpg",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the best_cnn.keras file from training run.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to inference_config.json (same run as model).",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to a single chest X-ray image (JPEG or PNG format).",
    )
    return parser


def main() -> None:
    """
    Main inference pipeline.

    Steps:
    1. Parse command-line arguments.
    2. Validate that all required files exist.
    3. Load configuration and verify model integrity.
    4. Preprocess the image (decode, resize, normalize).
    5. Run inference and generate output.
    6. Print results as JSON.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If model integrity check fails.
    """
    # Parse arguments
    args = build_parser().parse_args()

    # Lazy import: only load TensorFlow when actually running inference.
    # This keeps `python predict.py --help` fast.
    import tensorflow as tf

    # ──────────────────────────────────────────────────────────────────
    # Step 1: File validation
    # ──────────────────────────────────────────────────────────────────
    if not args.config.is_file():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Load configuration
    # ──────────────────────────────────────────────────────────────────
    config = json.loads(args.config.read_text(encoding="utf-8"))

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Integrity check (model hash verification)
    # ──────────────────────────────────────────────────────────────────
    # The config was frozen (along with the model) right after training.
    # The SHA-256 inside must match the actual model file. If it doesn't,
    # the model has either been corrupted or we're using mismatched files
    # from different training runs—both of which would be errors.
    model_hash = sha256_file(args.model)
    expected_hash = config["model_sha256"]
    if model_hash != expected_hash:
        raise ValueError(
            f"Model integrity check failed!\n"
            f"Expected SHA-256: {expected_hash}\n"
            f"Computed SHA-256: {model_hash}\n\n"
            f"Possible causes:\n"
            f"  - Model file was corrupted or modified.\n"
            f"  - Using model and config from different training runs.\n"
            f"  - File transfer error (e.g., incomplete download).\n\n"
            f"Solution: Ensure model and config come from the SAME training run.\n"
            f"See inference_config.json for run_id and model_sha256."
        )

    # ──────────────────────────────────────────────────────────────────
    # Step 4: Preprocess image (match notebook exactly)
    # ──────────────────────────────────────────────────────────────────
    # Image preprocessing pipeline (identical to notebook Steps 8 & 17):
    # 1. Read raw file bytes
    # 2. Decode to grayscale, handling any format (JPEG/PNG)
    # 3. Resize to stored dimensions using bilinear interpolation + antialiasing
    # 4. Cast to float32 (0-255 range)
    # 5. Model's first layer (Rescaling 1/255) normalizes to [0, 1]
    
    image_raw = tf.io.read_file(str(args.image))
    image = tf.io.decode_image(image_raw, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])  # Explicit shape for static analysis
    image = tf.image.resize(
        image,
        config["image_size"],
        method="bilinear",
        antialias=True
    )
    image = tf.cast(image, tf.float32)  # Keep 0-255; model rescales internally

    # ──────────────────────────────────────────────────────────────────
    # Step 5: Load model and run inference
    # ──────────────────────────────────────────────────────────────────
    # Load without compiling since we only do inference (no training).
    model = tf.keras.models.load_model(args.model, compile=False)
    
    # Single image → add batch dimension, then inference
    score = float(model(image[None, ...], training=False).numpy().reshape(-1)[0])

    # ──────────────────────────────────────────────────────────────────
    # Step 6: Apply threshold and generate output
    # ──────────────────────────────────────────────────────────────────
    threshold = float(config["threshold"])
    
    # Binary classification: if score >= threshold → PNEUMONIA, else → NORMAL
    label_idx = int(score >= threshold)
    predicted_class = config["class_mapping"][str(label_idx)]

    # Prepare JSON result
    result = {
        "image": args.image.name,
        "predicted_class": predicted_class,
        "pneumonia_score": round(score, 6),
        "threshold": threshold,
        "disclaimer": "Educational research model. Not for clinical diagnosis.",
    }

    # ──────────────────────────────────────────────────────────────────
    # Step 7: Print result
    # ──────────────────────────────────────────────────────────────────
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
