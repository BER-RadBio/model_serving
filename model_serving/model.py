from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)


def hf_model_fn(**kwargs):
    """Return a huggingface model

    Returns
    -------
    AutoModelForCausalLM
        The HF model from a given input config
    """

    # Get the tokenizer
    tokenizer_file = kwargs["tokenizer_file"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(tokenizer_file))
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Get model config
    model_config_json = kwargs["model_config_json"]
    model_config = AutoConfig.from_pretrained(model_config_json)

    model = AutoModelForCausalLM.from_config(model_config)

    return model
