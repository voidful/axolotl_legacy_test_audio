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
        """
        支援 input_ids, attention_mask, labels, pixel_values, input_audio_embeds,
        audio_embed_sizes, audio_attention_mask 多模態 batch 對齊。
        """
        batch: dict[str, Any] = {}

        # 1. 逐一把每個 example 用 processor 處理，確保所有欄位都正確
        for example in examples:
            # 這裡假設 example["messages"] 是經 dataset 處理後的 message 格式
            result = self.processing_strategy.processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
                chat_template=self.processing_strategy.chat_template,
            )
            for key, value in result.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value.squeeze(0))  # squeeze 去掉 batch 維度

        # 2. text: input_ids, attention_mask, labels padding
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch and batch[key][0] is not None:
                padding_value = self.tokenizer.pad_token_id if key != "labels" else -100
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    batch[key], batch_first=True, padding_value=padding_value
                )

        # 3. image: pixel_values stack and pad
        if "pixel_values" in batch and batch["pixel_values"][0] is not None:
            # 保證 shape = (n_img, 3, H, W)
            for i, pv in enumerate(batch["pixel_values"]):
                if pv is not None and len(pv.shape) == 3:
                    batch["pixel_values"][i] = pv.unsqueeze(0)
            num_imgs_max = max(pv.shape[0] for pv in batch["pixel_values"])
            img_shape = batch["pixel_values"][0].shape[1:]
            padded_pv = []
            for pv in batch["pixel_values"]:
                n = pv.shape[0]
                if n < num_imgs_max:
                    pad = torch.zeros(
                        (num_imgs_max - n, *img_shape), dtype=pv.dtype, device=pv.device
                    )
                    pv = torch.cat([pv, pad], dim=0)
                padded_pv.append(pv)
            batch["pixel_values"] = torch.stack(padded_pv, dim=0)  # (batch, num_imgs_max, 3, H, W)
        else:
            batch.pop("pixel_values", None)

        # 4. audio: input_audio_embeds, audio_embed_sizes, audio_attention_mask
        if "input_audio_embeds" in batch and batch["input_audio_embeds"][0] is not None:
            max_audio_len = max(x.shape[0] for x in batch["input_audio_embeds"])
            feat_dim = batch["input_audio_embeds"][0].shape[1]
            padded_audio = []
            for x in batch["input_audio_embeds"]:
                n = x.shape[0]
                if n < max_audio_len:
                    pad = torch.zeros((max_audio_len - n, feat_dim), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=0)
                padded_audio.append(x)
            batch["input_audio_embeds"] = torch.stack(padded_audio, dim=0)
            # audio_embed_sizes: (batch,)
            batch["audio_embed_sizes"] = torch.stack(batch["audio_embed_sizes"], dim=0)
            # audio_attention_mask: (batch, audio_len)
            if "audio_attention_mask" in batch and batch["audio_attention_mask"][0] is not None:
                padded_mask = []
                for x in batch["audio_attention_mask"]:
                    n = x.shape[0]
                    if n < max_audio_len:
                        pad = torch.zeros((max_audio_len - n,), dtype=x.dtype, device=x.device)
                        x = torch.cat([x, pad], dim=0)
                    padded_mask.append(x)
                batch["audio_attention_mask"] = torch.stack(padded_mask, dim=0)
            else:
                batch["audio_attention_mask"] = torch.ones(
                    (len(examples), max_audio_len), dtype=torch.bool, device=batch["input_audio_embeds"].device
                )
        else:
            batch.pop("input_audio_embeds", None)
            batch.pop("audio_embed_sizes", None)
            batch.pop("audio_attention_mask", None)

        # 5. 返回 batch
        return batch
