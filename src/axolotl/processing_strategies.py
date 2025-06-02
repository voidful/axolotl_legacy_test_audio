from copy import deepcopy
from typing import Optional
from PIL import Image, ImageOps
from PIL.Image import Resampling

from torch import Tensor

import numpy as np
import soundfile as sf
import io
from datasets import Audio

from transformers import ProcessorMixin,AutoProcessor
from transformers.image_utils import load_image

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

class ProcessingStrategy:
    """Base Processing Strategy class with multimodal (image/audio) support."""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None
        self.image_size = image_size
        self.image_resize_algorithm = (
            image_resize_algorithm or Image.Resampling.BILINEAR
        )
        # Try to fetch special image token if exists
        if hasattr(processor, "image_token"):
            self.image_token = processor.image_token
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
                self.image_token
            )

    def __call__(self, examples: list[dict]) -> list[dict]:
        role_mapping = {"human": "user", "gpt": "assistant"}
        def normalize_role(role: str) -> str:
            return role_mapping.get(role, role)
        def convert_legacy_format(example: dict) -> dict:
            messages = [
                {"role": normalize_role(convo["from"]), "content": convo["value"]}
                for convo in example["conversations"]
            ]
            result = deepcopy(example)
            result.pop("conversations")
            result["messages"] = messages
            return result
        def convert_messages_to_multimedia_messages(messages: list[dict]) -> list[dict]:
            new_messages = []
            for message in messages:
                if isinstance(message["content"], str):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": message["content"],
                                }
                            ],
                        }
                    )
                elif isinstance(message["content"], list):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": message["content"],
                        }
                    )
            return new_messages

        processed_examples = []
        for example in examples:
            if not ("messages" in example or "conversations" in example):
                raise ValueError(
                    "Only `messages` and `conversations` message keys are currently supported."
                )

            if "messages" in example and example["messages"] is not None:
                processed_example = example
            else:
                processed_example = convert_legacy_format(example)

            processed_example["messages"] = convert_messages_to_multimedia_messages(
                processed_example["messages"]
            )

            possible_image_keys = ["images", "image"]
            image_key = None
            for key in possible_image_keys:
                if key in processed_example and processed_example[key] is not None:
                    image_key = key
                    break
            if image_key is not None:
                if len(processed_example[image_key]) > 0:
                    LOG.warning(
                        f"Found {len(processed_example[image_key])} images in a sample. Using the first one."
                        "If you are using a dataset with multiple images per sample, please convert it to use multi-content Messages."
                        "See https://docs.axolotl.ai/docs/multimodal.html#dataset-format"
                    )
                image_value = processed_example[image_key][0]
                image_value = load_image(image_value)
                if self.image_size is not None:
                    assert hasattr(image_value, "resize")
                    if isinstance(self.image_size, tuple):
                        image_value = image_value.resize(
                            self.image_size, self.image_resize_algorithm
                        )
                    else:
                        padding_color = (0, 0, 0)
                        image_value = ImageOps.pad(
                            image_value,
                            (self.image_size, self.image_size),
                            method=self.image_resize_algorithm,
                            color=padding_color,
                        )
                ind_to_add = None
                for i, content in enumerate(processed_example["messages"][0]["content"]):
                    if content["type"] == "image" and all(
                        k not in content for k in ["image", "url", "path", "base64"]
                    ):
                        ind_to_add = i
                        break
                if ind_to_add is not None:
                    processed_example["messages"][0]["content"][ind_to_add][
                        "image"
                    ] = image_value
                else:
                    processed_example["messages"][0]["content"].append(
                        {
                            "type": "image",
                            "image": image_value,
                        }
                    )
            possible_audio_keys = ["audios", "audio"]
            audio_key = None
            for key in possible_audio_keys:
                if key in processed_example and processed_example[key] is not None:
                    audio_key = key
                    break
            if audio_key is not None:
                audio_field = processed_example[audio_key]
                # 保證 audio_field 一定為 list
                if not isinstance(audio_field, list):
                    audio_field = [audio_field]
                if len(audio_field) == 0:
                    LOG.warning(f"No audios found in {audio_key} of sample, skipping.")
                else:
                    if len(audio_field) > 1:
                        LOG.warning(
                            f"Found {len(audio_field)} audios in a sample. Using the first one."
                            "If you are using a dataset with multiple audios per sample, please convert it to use multi-content Messages."
                        )
                    audio_value = audio_field[0]
                    waveform = None
                    sampling_rate = 16000
                    if isinstance(audio_value, dict):
                        waveform = audio_value.get("array")
                        sampling_rate = audio_value.get("sampling_rate", 16000)
                    elif isinstance(audio_value, np.ndarray):
                        waveform = audio_value
                    elif isinstance(audio_value, str):
                        waveform, sampling_rate = sf.read(audio_value)
                    elif isinstance(audio_value, bytes):
                        waveform, sampling_rate = sf.read(io.BytesIO(audio_value))
                    else:
                        raise ValueError(f"Unknown audio format: {type(audio_value)}")
        
                    ind_to_add = None
                    for i, content in enumerate(processed_example["messages"][0]["content"]):
                        if content["type"] == "audio" and "audio" not in content:
                            ind_to_add = i
                            break
        
                    audio_content = {
                        "type": "audio",
                        "audio": waveform,
                        "sampling_rate": sampling_rate,
                    }
                    if ind_to_add is not None:
                        processed_example["messages"][0]["content"][ind_to_add].update(audio_content)
                    else:
                        processed_example["messages"][0]["content"].append(audio_content)

            processed_examples.append(processed_example)
        return processed_examples

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = input_ids.clone()
        if hasattr(self.processor.tokenizer, "pad_token_id"):
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        return labels



class Qwen2VLProcessingStrategy(ProcessingStrategy):
    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        self.image_token = "<|image_pad|>"
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(self.image_token)

class Gemma3ProcessingStrategy(ProcessingStrategy):
    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
    ):
        super().__init__(processor, chat_template, image_size, image_resize_algorithm)
        if processor is None:
           processor = AutoProcessor.from_pretrained("voidful/gemma-3-omni-27b-it",trust_remote_code=True) 
        self.image_token = processor.tokenizer.special_tokens_map["boi_token"]
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(self.image_token)

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        labels[labels == 262144] = -100  # <image_soft_token>
        return labels

def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
):
    if chat_template_type == "qwen2_vl":
        return Qwen2VLProcessingStrategy(processor, chat_template, image_size, image_resize_algorithm)
    if chat_template_type == "gemma3":
        return Gemma3ProcessingStrategy(processor, chat_template, image_size, image_resize_algorithm)
    if chat_template_type in [
        "llama3_2_vision", "llama4", "llava", "mistral_v7_tekken", "pixtral",
    ]:
        return ProcessingStrategy(processor, chat_template, image_size, image_resize_algorithm)
    raise ValueError(f"Unsupported chat template type: {chat_template_type}")
