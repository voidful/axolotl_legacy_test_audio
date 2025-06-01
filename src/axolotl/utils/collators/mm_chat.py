from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

from axolotl.processing_strategies import ProcessingStrategy

@dataclass
class MultiModalChatDataCollator(DataCollatorMixin):
    """
    Collator for multi-modal chat messages (text/image/audio) for Axolotl
    """

    tokenizer: PreTrainedTokenizerBase
    processing_strategy: ProcessingStrategy
    packing: bool = False
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.packing:
            raise ValueError("Packing is currently not supported.")

    def torch_call(self, examples: list[dict]) -> dict[str, Any]:
        return self.process_rows(examples)

    def process_rows(
        self,
        examples: list[dict],
    ) -> dict[str, Tensor]:
        examples = self.processing_strategy(examples) 

        batch: dict[str, Any] = {}

        for example in examples:
            result = self.processing_strategy.processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
                chat_template=self.processing_strategy.chat_template,
            )

            for key in result.keys():
                if key not in batch:
                    batch[key] = []
                batch[key].append(result[key].squeeze(0))

        # pad_sequence to batch shape
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch["attention_mask"], batch_first=True, padding_value=0
        )

        final_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        final_batch["labels"] = self.processing_strategy.process_labels(final_batch["input_ids"])

        return final_batch
