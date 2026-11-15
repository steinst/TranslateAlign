"""What the two Hugging Face backends share: imports on demand, device, loading settings."""
from __future__ import annotations

from typing import Mapping, Optional

INSTALL = ("translating with a model needs PyTorch and transformers: pip install \"translatealign[models]\" "
           "(or give a ready translation with --translation FILE)")


class RemoteCodeNotAllowed(RuntimeError):
    pass


def imports():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ImportError(f"{INSTALL} [{e}]") from None
    return torch, transformers


def pick_device(device: Optional[str] = None) -> str:
    torch, _ = imports()
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_remote_code(needed: bool, allowed: bool, model: str) -> bool:
    """Code from a model repository is run only if the user asked for it explicitly."""
    if needed and not allowed:
        raise RemoteCodeNotAllowed(
            f"the model {model} runs code downloaded from its repository; this is only done on the explicit "
            f"option --trust-remote-code (Python: trust_remote_code=True). Read that code first.")
    return bool(needed and allowed)


def load_model(model_class, model_id: str, load: Mapping, device: str, trust_remote_code: bool):
    """Load with the [load] settings of a definition: dtype, quantization (4bit / 8bit with bitsandbytes)."""
    torch, transformers = imports()
    kwargs = {}
    dtype = load.get("dtype", "auto")
    if dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, dtype)
    elif load.get("quantization", "none") == "none":
        kwargs["torch_dtype"] = "auto"
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if load.get("subfolder"):
        kwargs["subfolder"] = load["subfolder"]
    if load.get("revision"):
        kwargs["revision"] = load["revision"]
    quant = load.get("quantization", "none")
    if quant != "none":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise ImportError("quantization = \"4bit\" / \"8bit\" needs bitsandbytes and accelerate: "
                              "pip install \"translatealign[quantized]\"") from None
        if quant == "4bit":
            compute = getattr(torch, dtype) if dtype != "auto" else torch.bfloat16
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute)
        else:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": device}
        return model_class.from_pretrained(model_id, **kwargs).eval()
    return model_class.from_pretrained(model_id, **kwargs).to(device).eval()


def free(*objects) -> None:
    import gc
    del objects
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def finish_ids(model, tokenizer) -> set:
    """Token ids with which a finished sequence ends (end of sequence, and the padding after it)."""
    ids = set()
    for source in (getattr(model, "generation_config", None), tokenizer):
        for attr in ("eos_token_id", "pad_token_id"):
            value = getattr(source, attr, None)
            if value is None:
                continue
            ids.update(value if isinstance(value, (list, tuple)) else [value])
    return ids


def hit_limit(rows, finish: set, stop_texts=(), stop_strings=()) -> list:
    """Positions of the sequences that did not end with an end-of-sequence token, i.e. stopped because the
    limit on new tokens was reached. `rows` are the generated token ids per sequence; a sequence that ended at
    a stop string (`stop_texts` are the decoded rows) did not hit the limit."""
    cut = []
    for n, row in enumerate(rows):
        row = list(row)
        if not row or row[-1] in finish:
            continue
        if stop_strings and any(s in stop_texts[n] for s in stop_strings):
            continue
        cut.append(n)
    return cut
