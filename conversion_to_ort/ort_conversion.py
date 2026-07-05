"""Convert Microsoft's Table Transformer model to ONNX, then to ONNX Runtime (.ort) format.

Pipeline:
    Hugging Face Hub (PyTorch) -> ONNX (torch.onnx.export) -> ORT format (onnxruntime)

Extra dependencies beyond requirements.txt:
    pip install onnx onnxruntime

Usage:
    python conversion_to_ort/ort_conversion.py
    python conversion_to_ort/ort_conversion.py --model microsoft/table-transformer-structure-recognition
"""

import argparse
import sys
from pathlib import Path

import torch

DEFAULT_MODEL = "microsoft/table-transformer-detection"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "models"
DEFAULT_IMAGE_SIZE = (800, 1000)  # (height, width) Table Transformer's own processor resizes to
DEFAULT_OPSET = 17


class _ObjectDetectionOnnxWrapper(torch.nn.Module):
    """Reduces the model output to plain tensors so tracing doesn't choke on the HF ModelOutput dataclass."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs.logits, outputs.pred_boxes


def export_to_onnx(model_name: str, onnx_path: Path, image_size: tuple, opset: int) -> None:
    from transformers import AutoModelForObjectDetection

    print(f"Downloading '{model_name}' from the Hugging Face Hub...")
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.eval()
    wrapped_model = _ObjectDetectionOnnxWrapper(model)

    height, width = image_size
    pixel_values = torch.randn(1, 3, height, width)
    pixel_mask = torch.ones(1, height, width, dtype=torch.long)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting '{model_name}' to ONNX ({onnx_path})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (pixel_values, pixel_mask),
            str(onnx_path),
            input_names=["pixel_values", "pixel_mask"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                "pixel_mask": {0: "batch_size", 1: "height", 2: "width"},
                "logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    print("ONNX export complete.")


def convert_onnx_to_ort(onnx_path: Path, optimization_styles: list) -> None:
    from onnxruntime.tools.convert_onnx_models_to_ort import OptimizationStyle, convert_onnx_models_to_ort

    print(f"Converting {onnx_path} to ORT format ({', '.join(optimization_styles)})...")
    convert_onnx_models_to_ort(
        onnx_path,
        optimization_styles=[OptimizationStyle[style] for style in optimization_styles],
    )
    print(f"ORT model(s) written next to {onnx_path}")


def detect_tables(image_path: Path, ort_model_path: Path, model_name: str = DEFAULT_MODEL, score_threshold: float = 0.7) -> list:
    """Detect table regions in the image at `image_path` using a converted .ort model.

    Reuses the Hugging Face image processor for `model_name` so the resize/normalize/box
    de-normalization exactly matches what the model was traced with in `export_to_onnx`.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor

    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_name)
    id2label = AutoConfig.from_pretrained(model_name).id2label

    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"]
    pixel_mask = inputs.get("pixel_mask", np.ones(pixel_values.shape[:1] + pixel_values.shape[2:], dtype=np.int64))

    session = ort.InferenceSession(str(ort_model_path), providers=["CPUExecutionProvider"])
    logits, pred_boxes = session.run(
        ["logits", "pred_boxes"],
        {"pixel_values": pixel_values, "pixel_mask": pixel_mask},
    )

    outputs = argparse.Namespace(logits=torch.from_numpy(logits), pred_boxes=torch.from_numpy(pred_boxes))
    result = processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=[image.size[::-1]]
    )[0]

    return [
        {"label": id2label[label.item()], "score": score.item(), "box": box.tolist()}
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"])
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Table Transformer model to ORT format, or run it.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Export a Table Transformer model to ONNX and ORT format.")
    convert_parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id (default: %(default)s)")
    convert_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the converted model files"
    )
    convert_parser.add_argument(
        "--height", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Dummy input image height used for tracing"
    )
    convert_parser.add_argument(
        "--width", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Dummy input image width used for tracing"
    )
    convert_parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    convert_parser.add_argument(
        "--optimization-style",
        nargs="+",
        default=["Fixed"],
        choices=["Fixed", "Runtime"],
        help="ORT optimization style(s) to produce (default: %(default)s)",
    )

    detect_parser = subparsers.add_parser("detect", help="Detect tables in an image using a converted .ort model.")
    detect_parser.add_argument("image", type=Path, help="Path to the input image")
    detect_parser.add_argument("--ort-model", type=Path, required=True, help="Path to the converted .ort model")
    detect_parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF model id the .ort model was exported from (default: %(default)s)"
    )
    detect_parser.add_argument("--score-threshold", type=float, default=0.7, help="Minimum detection score to keep")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}. Install with:\n  pip install torch transformers onnx onnxruntime")

    if args.command == "convert":
        onnx_path = args.output_dir / (args.model.split("/")[-1] + ".onnx")
        export_to_onnx(args.model, onnx_path, (args.height, args.width), args.opset)
        convert_onnx_to_ort(onnx_path, args.optimization_style)
    elif args.command == "detect":
        detections = detect_tables(args.image, args.ort_model, args.model, args.score_threshold)
        for detection in detections:
            print(f"{detection['label']:<12} score={detection['score']:.3f} box={detection['box']}")


if __name__ == "__main__":
    main()
