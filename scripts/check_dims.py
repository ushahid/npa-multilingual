#!/usr/bin/env python3
"""Print the C-RADIO embedding shape for a dummy image to check the model output."""

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "nvidia/C-RADIOv4-H"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_embedding(outputs):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "summary") and outputs.summary is not None:
        return outputs.summary
    if isinstance(outputs, dict):
        if "pooler_output" in outputs and outputs["pooler_output"] is not None:
            return outputs["pooler_output"]
        if "summary" in outputs and outputs["summary"] is not None:
            return outputs["summary"]
        if "embeddings" in outputs and outputs["embeddings"] is not None:
            return outputs["embeddings"]
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"][:, 0, :]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        t = outputs[0]
        if t.dim() == 3:
            return t[:, 0, :]
        return t
    raise RuntimeError("Could not extract embedding from RADIO output")

def main():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE).eval()

    # one dummy image; size does not matter much for this check
    img = Image.new("RGB", (224, 224), (0, 0, 0))

    try:
        inputs = processor(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)
    except TypeError:
        inputs = processor(images=img, return_tensors="pt")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            outputs = model(**inputs)
        except TypeError:
            # some RADIO remote-code forwards expect positional input instead
            pv = inputs["pixel_values"]
            try:
                outputs = model(pv)
            except Exception:
                outputs = model(x=pv)

    emb = extract_embedding(outputs)

    print("embedding tensor shape:", tuple(emb.shape))
    print("embedding dimension:", int(emb.shape[-1]))

if __name__ == "__main__":
    main()