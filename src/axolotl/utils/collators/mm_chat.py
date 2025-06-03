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

        # 1. 按欄位收集 batch
        batch: dict[str, Any] = {}
        for key in examples[0].keys():
            batch[key] = [ex[key] for ex in examples]

        # 2. text padding
        pad_keys = ["input_ids", "attention_mask", "labels"]
        for key in pad_keys:
            if key in batch and batch[key][0] is not None:
                padding_value = (
                    self.tokenizer.pad_token_id if key != "labels" else -100
                )
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    batch[key], batch_first=True, padding_value=padding_value
                )

        # 3. image padding
        if "pixel_values" in batch and batch["pixel_values"][0] is not None:
            # 單圖補 batch 維度，多圖確保 shape = (n_img, 3, H, W)
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

        # 4. audio padding
        # input_audio_embeds: (batch, audio_len, feature_dim)
        # audio_embed_sizes: (batch,)
        # audio_attention_mask: (batch, audio_len)
        if "input_audio_embeds" in batch and batch["input_audio_embeds"][0] is not None:
            # 找出 batch 內最長 audio_len，pad_sequence
            max_audio_len = max(x.shape[0] for x in batch["input_audio_embeds"])
            feat_dim = batch["input_audio_embeds"][0].shape[1]
            padded_audio = []
            for x in batch["input_audio_embeds"]:
                n = x.shape[0]
                if n < max_audio_len:
                    pad = torch.zeros((max_audio_len - n, feat_dim), dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=0)
                padded_audio.append(x)
            batch["input_audio_embeds"] = torch.stack(padded_audio, dim=0)  # (batch, audio_len, feat_dim)

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
                # 若沒提供，全部設為 1
                batch["audio_attention_mask"] = torch.ones(
                    (len(examples), max_audio_len), dtype=torch.bool, device=batch["input_audio_embeds"].device
                )
        else:
            # 無 audio 則移除
            batch.pop("input_audio_embeds", None)
            batch.pop("audio_embed_sizes", None)
            batch.pop("audio_attention_mask", None)

        # 5. 返回 batch
        return batch
