import torch
from config.config import *
from transformers import AutoTokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class Tokenizer:
    def __init__(self, lm_type):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_type)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        return torch.Tensor(self.tokenizer.convert_tokens_to_ids(tokens))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def encode(self, text: str, add_special_tokens: bool = True, max_length=128) -> List[int]:
        return self.tokenizer.encode_plus(text, add_special_tokens=add_special_tokens, return_tensors="pt", padding="max_length", max_length=max_length)

    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    def mask_token(self):
        return self.tokenizer.mask_token

    def unk_id(self):
        return self.tokenizer.unk_token_id

    def get_vocab_len(self):
        return len(self.tokenizer.get_vocab())

    def pad_token_id(self):
        return self.tokenizer.pad_token_id