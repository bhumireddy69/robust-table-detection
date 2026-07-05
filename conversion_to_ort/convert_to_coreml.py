"""Convert Microsoft's Table Transformer model to Core ML for on-device iOS/macOS inference.

Pipeline:
    Hugging Face Hub (PyTorch) -> traced PyTorch module -> Core ML (.mlpackage)

coremltools has no reader for the ONNX Runtime `.ort` format (or, in current coremltools
versions, for plain ONNX either) so this script does not start from
models/table-transformer-detection.ort or .onnx. Instead it retraces the same Hugging Face
PyTorch model that ort_conversion.py exports to ONNX/.ort and feeds that trace to coremltools
directly. The result is functionally equivalent to models/table-transformer-detection.ort,
just packaged for Core ML / the Apple Neural Engine instead of ONNX Runtime.

Unlike the ONNX/.ort export, this Core ML export uses a FIXED input size baked in at
conversion time (default 800x1000, the same dummy size ort_conversion.py traces with).
coremltools' MIL converter does not reliably support the dynamic height/width this model
needs while computing its sinusoidal position embeddings, so the produced .mlpackage only
accepts exactly that input size. Callers must letterbox/resize images to that size before
inference; see `detect_tables` below for a reference implementation.

Extra dependencies beyond requirements.txt:
    pip install coremltools

Usage:
    python conversion_to_ort/convert_to_coreml.py convert
    python conversion_to_ort/convert_to_coreml.py convert --model microsoft/table-transformer-structure-recognition
    python conversion_to_ort/convert_to_coreml.py detect input/input1.jpeg \
        --mlmodel conversion_to_ort/models/table-transformer-detection.mlpackage
"""

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "microsoft/table-transformer-detection"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "models"
DEFAULT_IMAGE_SIZE = (800, 1000)  # (height, width), matches ort_conversion.py's dummy trace size
DEFAULT_MIN_DEPLOYMENT_TARGET = "iOS16"


class _ObjectDetectionCoreMLWrapper:
    """Built lazily so importing this module doesn't require torch to be installed."""

    @staticmethod
    def build(model):
        import torch

        class Wrapper(torch.nn.Module):
            """Reduces the model output to plain tensors, like ort_conversion.py's wrapper.

            pixel_mask is accepted as int32 (Core ML has no int64 input support) and widened
            to the int64 the Hugging Face model expects before the forward call.
            """

            def __init__(self, inner):
                super().__init__()
                self.model = inner

            def forward(self, pixel_values, pixel_mask):
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask.long())
                return outputs.logits, outputs.pred_boxes

        return Wrapper(model)


def _patch_masking_utils_for_tracing() -> None:
    """Let transformers skip building a padding-aware attention mask while we trace.

    transformers' generic masking utilities (masking_utils.create_bidirectional_mask) can
    take a fast path that returns `None` (full attention, no bias) whenever the padding mask
    is all-ones -- exactly our case, since this export always uses an all-ones pixel_mask for
    a single, unpadded image. But that fast path is unconditionally disabled while tracing
    (masking_utils.is_tracing() short-circuits it), as a conservative default so exported
    graphs stay correct for inputs with real padding. The disabled path instead builds the
    mask via an `and_masks` combinator that calls `Tensor.new_ones`, an op coremltools' torch
    frontend doesn't implement. Since our pixel_mask is provably all-ones here, forcing the
    skip is not an approximation -- it's the exact same result via a traceable path.
    """
    import transformers.masking_utils as masking_utils

    masking_utils.is_tracing = lambda *args, **kwargs: False


def _patch_table_transformer_attention_for_tracing() -> None:
    """Swap TableTransformerAttention's manual bmm-based attention for F.scaled_dot_product_attention.

    The original forward (transformers/models/table_transformer/modeling_table_transformer.py)
    unpacks `batch_size, target_len, embed_dim = hidden_states.size()` and threads `batch_size`
    through `proj_shape = (batch_size * self.num_heads, -1, self.head_dim)` and several
    `.view(*proj_shape)` / size-equality-assertion calls. Under torch.jit.trace those shape
    values come back as traced tensors rather than plain Python ints, and coremltools' torch
    frontend fails converting one of the resulting `aten::Int` casts ("only 0-dimensional
    arrays can be converted to Python scalars"). We only ever trace with a fixed batch_size=1
    and (thanks to _patch_masking_utils_for_tracing) an attention_mask of None, so a plain SDPA
    call is numerically identical and avoids that whole manual reshape/assert chain.
    """
    import torch.nn.functional as F
    from transformers.models.table_transformer import modeling_table_transformer

    def traceable_forward(self, hidden_states, attention_mask=None, object_queries=None,
                           key_value_states=None, spatial_position_embeddings=None,
                           output_attentions=False):
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.shape

        hidden_states_original = hidden_states
        if object_queries is not None:
            hidden_states = self.with_pos_embed(hidden_states, object_queries)

        key_value_states_original = key_value_states
        if spatial_position_embeddings is not None:
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

        query_states = self.q_proj(hidden_states)
        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states_original)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states_original)

        def reshape_heads(tensor):
            return tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = reshape_heads(query_states)
        key_states = reshape_heads(key_states)
        value_states = reshape_heads(value_states)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None

    modeling_table_transformer.TableTransformerAttention.forward = traceable_forward


def export_to_coreml(model_name: str, output_path: Path, image_size: tuple, min_deployment_target: str) -> None:
    import coremltools as ct
    import numpy as np
    import torch
    from transformers import AutoModelForObjectDetection

    print(f"Downloading '{model_name}' from the Hugging Face Hub...")
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.eval()
    wrapped_model = _ObjectDetectionCoreMLWrapper.build(model)

    height, width = image_size
    pixel_values = torch.randn(1, 3, height, width)
    pixel_mask = torch.ones(1, height, width, dtype=torch.int32)

    _patch_masking_utils_for_tracing()
    _patch_table_transformer_attention_for_tracing()

    print(f"Tracing '{model_name}' with fixed input size {height}x{width}...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, (pixel_values, pixel_mask))

    print(f"Converting to Core ML (minimum deployment target: {min_deployment_target})...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="pixel_values", shape=pixel_values.shape, dtype=np.float32),
            ct.TensorType(name="pixel_mask", shape=pixel_mask.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="pred_boxes"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, min_deployment_target),
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.author = model_name
    mlmodel.short_description = (
        f"{model_name} (Table Transformer object detector), traced at a fixed "
        f"{height}x{width} input size. See conversion_to_ort/ort_conversion.py for the "
        "dynamic-shape ONNX/.ort export of the same model."
    )
    mlmodel.input_description["pixel_values"] = f"RGB image, ImageNet-normalized, shape [1,3,{height},{width}]"
    mlmodel.input_description["pixel_mask"] = f"All-ones mask, shape [1,{height},{width}]"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"Core ML model written to {output_path}")


def detect_tables(
    image_path: Path, mlmodel_path: Path, model_name: str = DEFAULT_MODEL, score_threshold: float = 0.7
) -> list:
    """Detect table regions in the image at `image_path` using a converted .mlpackage model.

    The .mlpackage only accepts the fixed input size it was traced with, so the image is
    letterboxed (resized without preserving aspect ratio, then ImageNet-normalized) to that
    size rather than using the Hugging Face processor's dynamic resize. Boxes are not
    rescaled back to the original image, since they're only meaningful relative to the
    letterboxed canvas; only labels/scores are returned.
    """
    import coremltools as ct
    import numpy as np
    from PIL import Image
    from transformers import AutoConfig

    mlmodel = ct.models.MLModel(str(mlmodel_path))
    input_spec = mlmodel.get_spec().description.input
    _, _, height, width = next(i for i in input_spec if i.name == "pixel_values").type.multiArrayType.shape

    image = Image.open(image_path).convert("RGB").resize((width, height), Image.BILINEAR)
    pixel_array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixel_array = (pixel_array - mean) / std
    pixel_values = pixel_array.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> [1,3,H,W]
    pixel_mask = np.ones((1, height, width), dtype=np.int32)

    id2label = AutoConfig.from_pretrained(model_name).id2label

    outputs = mlmodel.predict({"pixel_values": pixel_values, "pixel_mask": pixel_mask})
    logits = outputs["logits"][0]  # [num_queries, num_classes+1]

    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    class_probs = probs[:, :-1]  # drop the "no object" class
    labels = class_probs.argmax(axis=-1)
    scores = class_probs.max(axis=-1)

    return [
        {"label": id2label[int(label)], "score": float(score)}
        for label, score in zip(labels, scores)
        if score >= score_threshold
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Table Transformer model to Core ML, or run it.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Export a Table Transformer model to a .mlpackage.")
    convert_parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id (default: %(default)s)")
    convert_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the converted .mlpackage"
    )
    convert_parser.add_argument(
        "--height", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Fixed input image height (default: %(default)s)"
    )
    convert_parser.add_argument(
        "--width", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Fixed input image width (default: %(default)s)"
    )
    convert_parser.add_argument(
        "--min-deployment-target",
        default=DEFAULT_MIN_DEPLOYMENT_TARGET,
        help="coremltools.target attribute name, e.g. iOS16, iOS17 (default: %(default)s)",
    )

    detect_parser = subparsers.add_parser("detect", help="Detect tables in an image using a converted .mlpackage.")
    detect_parser.add_argument("image", type=Path, help="Path to the input image")
    detect_parser.add_argument("--mlmodel", type=Path, required=True, help="Path to the converted .mlpackage")
    detect_parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF model id the .mlpackage was exported from (default: %(default)s)"
    )
    detect_parser.add_argument("--score-threshold", type=float, default=0.7, help="Minimum detection score to keep")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import coremltools  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}. Install with:\n  pip install torch transformers coremltools")

    if args.command == "convert":
        output_path = args.output_dir / (args.model.split("/")[-1] + ".mlpackage")
        export_to_coreml(args.model, output_path, (args.height, args.width), args.min_deployment_target)
    elif args.command == "detect":
        detections = detect_tables(args.image, args.mlmodel, args.model, args.score_threshold)
        if not detections:
            print("No table detected.")
        for detection in detections:
            print(f"{detection['label']:<12} score={detection['score']:.3f}")


if __name__ == "__main__":
    main()
