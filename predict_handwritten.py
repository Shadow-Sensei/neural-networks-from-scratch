"""Predict a handwritten digit photographed on light paper.

Usage:
    python predict_handwritten.py photo.jpg --model mnist_model.npz
    python predict_handwritten.py photo.jpg --model mnist_model.npz --preview

Requires: numpy, Pillow. Matplotlib is only required for --preview.
This script DOES NOT retrain the model. It uses the 784->16->16->10 weights.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


EXPECTED_SHAPES = {
    "W1": (16, 784), "B1": (16,),
    "W2": (16, 16), "B2": (16,),
    "W3": (10, 16), "B3": (10,),
}


def load_model(model_path: Path) -> dict[str, np.ndarray]:
    """Read the six learned parameter arrays and validate their dimensions."""
    with np.load(model_path, allow_pickle=False) as saved:
        params = {}
        for name, shape in EXPECTED_SHAPES.items():
            if name not in saved:
                raise ValueError(f"Model is missing {name}.")
            params[name] = np.asarray(saved[name])
            if params[name].shape != shape:
                raise ValueError(
                    f"{name} has shape {params[name].shape}, expected {shape}. "
                    "This script assumes architecture 784 -> 16 -> 16 -> 10."
                )
    return params


def preprocess_photo(image_path: Path) -> np.ndarray:
    """Convert a dark digit on light paper into a centered MNIST-like 28x28 array.

    Returns values in [0, 1] with a BRIGHT digit on a DARK background.
    The digit is scaled to fit within a 20x20 box, preserving its aspect ratio.
    """
    with Image.open(image_path) as source:
        # EXIF transpose also handles phone camera orientation correctly.
        grayscale = ImageOps.exif_transpose(source).convert("L")

    pixels = np.asarray(grayscale, dtype=np.float32)
    if pixels.size == 0:
        raise ValueError("The image is empty.")

    # Estimate background brightness from image edges (ideally plain white paper).
    border = np.concatenate((pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]))
    paper_brightness = float(np.median(border))
    ink_strength = np.clip(paper_brightness - pixels, 0, 255)

    # Reject faint paper texture and select sufficiently dark handwriting.
    # The background is estimated from the image edges; a blue or black pen
    # on light paper should have noticeably greater contrast than the paper.
    contrast = float(np.percentile(ink_strength, 99.5))
    if contrast < 12:
        raise ValueError("Couldn't find a dark digit. Use darker ink and even lighting.")

    threshold = max(35.0, paper_brightness * 0.20)
    mask = ink_strength >= threshold
    if not np.any(mask):
        raise ValueError("No digit detected. Use darker ink or improve the lighting.")

    ys, xs = np.nonzero(mask)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1

    # Only keep detected ink; suppress paper texture in the cropped area.
    cropped = np.where(
        mask[top:bottom, left:right],
        ink_strength[top:bottom, left:right],
        0.0,
    )

    # Rescale the pen strokes to a useful range before shrinking to 28x28.
    positive_ink = cropped[cropped > 0]
    scale = float(np.percentile(positive_ink, 95))
    if scale > 0:
        cropped = np.clip(cropped * (255.0 / scale), 0, 255)

    # Crop to the ink bounding box and rescale (with a margin) into a 28x28 canvas.
    h, w = cropped.shape
    new_w = max(1, round(w * 20 / max(w, h)))
    new_h = max(1, round(h * 20 / max(w, h)))
    cropped_image = Image.fromarray(np.uint8(np.clip(cropped, 0, 255)), mode="L")
    resized = cropped_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    # Adjust the center of mass to approximately the MNIST image center.
    arr = np.asarray(canvas, dtype=np.float32)
    total = float(arr.sum())
    if total > 0:
        yy, xx = np.indices(arr.shape)
        cx = float((arr * xx).sum() / total)
        cy = float((arr * yy).sum() / total)
        dx = int(round(13.5 - cx))
        dy = int(round(13.5 - cy))
        centered = np.zeros_like(arr)
        dst_x0, dst_x1 = max(dx, 0), min(28 + dx, 28)
        dst_y0, dst_y1 = max(dy, 0), min(28 + dy, 28)
        src_x0, src_x1 = max(-dx, 0), min(28 - dx, 28)
        src_y0, src_y1 = max(-dy, 0), min(28 - dy, 28)
        centered[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
        arr = centered

    return np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)


def predict(image_28x28: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Run inference on ONE image; shape is kept (1,784) for batch operations."""
    X = image_28x28.reshape(1, 784)
    Z1 = X @ params["W1"].T + params["B1"]
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ params["W2"].T + params["B2"]
    A2 = np.maximum(0, Z2)
    Z3 = A2 @ params["W3"].T + params["B3"]
    shifted = Z3 - np.max(Z3, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize a photographed handwritten digit using a trained NumPy MNIST model.")
    parser.add_argument("image", type=Path, help="Image path: .jpg, .png, etc.")
    parser.add_argument("--model", type=Path, default=Path("mnist_model.npz"), help="Saved np.savez model file")
    parser.add_argument("--preview", action="store_true", help="Show 28x28 preprocessing preview (requires matplotlib)")
    parser.add_argument("--save-preview", type=Path, default=None, help="Save enlarged 28x28 preview as an image")
    args = parser.parse_args()

    image_28x28 = preprocess_photo(args.image)
    params = load_model(args.model)
    probabilities = predict(image_28x28, params)
    digit = int(np.argmax(probabilities))

    print(f"Predicted digit: {digit}")
    print(f"Softmax confidence: {probabilities[digit] * 100:.2f}%")
    print("All class probabilities:")
    for label, value in enumerate(probabilities):
        print(f"  {label}: {value * 100:6.2f}%")

    preview = Image.fromarray(np.uint8(np.round(image_28x28 * 255)), mode="L")
    if args.save_preview is not None:
        preview.resize((280, 280), Image.Resampling.NEAREST).save(args.save_preview)
        print(f"Preprocessed preview saved to: {args.save_preview}")

    if args.preview:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("Install matplotlib for --preview: pip install matplotlib") from exc
        plt.imshow(image_28x28, cmap="gray", vmin=0, vmax=1)
        plt.title(f"Processed 28x28 digit — prediction: {digit} ({probabilities[digit]*100:.1f}%)")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
