from __future__ import annotations


def disable_hf_safetensors_auto_conversion() -> None:
    """Disable Hugging Face background safetensors auto-conversion threads.

    Some repos (for example `hfl/chinese-roberta-wwm-ext`) have discussions disabled,
    which causes `transformers` to spawn a noisy background thread that fails with 403
    while training continues. We replace the auto-conversion helper with a no-op.
    """

    try:
        from transformers import modeling_utils, safetensors_conversion

        def _noop_auto_conversion(*args, **kwargs):
            return None, None, None

        modeling_utils.auto_conversion = _noop_auto_conversion
        safetensors_conversion.auto_conversion = _noop_auto_conversion
    except Exception:
        # If transformers internals change, we prefer to continue normally rather than fail.
        return

