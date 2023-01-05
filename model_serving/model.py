from transformers import AutoModelForCausalLM


def hf_causal_model_fn(chkpt_path: str, **kwargs):
    """Return a huggingface model

    This function assumes the checkpoint path is a directory following the HF convention.
    See https://huggingface.co/EleutherAI/gpt-neo-125M/tree/main for an example.
    (Needs model config and pytorch_model.bin files at minimum)

    Returns
    -------
    AutoModelForCausalLM
        The HF model from a given input config
    """
    model = AutoModelForCausalLM.from_pretrained(chkpt_path)

    return model
