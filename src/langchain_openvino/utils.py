import openvino_genai
from openvino import Tensor
import openvino as ov
import queue
import json
import os
import numpy as np
from typing import Union, Dict, Any, List
from PIL import Image
import librosa
import base64
from io import BytesIO
import requests
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import BaseMessage

REPLACEMENT_CHAR = chr(
    65533
)  # Unicode replacement character for invalid UTF-8 sequences


class IterableStreamer(openvino_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer):
        """
        Initializes the IterableStreamer with the given tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.decoded_lengths = []

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next value from the text queue.

        Returns:
            str: The next decoded text chunk.

        Raises:
            StopIteration: If there are no more elements in the queue.
        """
        # get() will be blocked until a token is available.
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped or cancelled.

        Returns:
            openvino_genai.StreamingStatus: Always returns RUNNING in this implementation.
        """
        return openvino_genai.StreamingStatus.RUNNING

    def write_word(self, word: str):
        """
        Puts a word into the text queue.

        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def write(
        self, token: Union[int, list[int]], delay_n_tokens: int = 3
    ) -> openvino_genai.StreamingStatus:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token (Union[int, list[int]]): The token(s) to process.

        Returns:
            bool: True if generation should be stopped, False otherwise.
        """
        if type(token) is list:
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token) - 1)]
        else:
            self.tokens_cache.append(token)

        text = self.tokenizer.decode(self.tokens_cache)
        self.decoded_lengths.append(len(text))

        word = ""
        if len(text) > self.print_len and "\n" == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len :]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == REPLACEMENT_CHAR:
            # Don't print incomplete text.
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            self.compute_decoded_length_for_position(
                len(self.decoded_lengths) - delay_n_tokens
            )
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text length is increased and text is complete (print_until != -1).
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.write_word(word)

        stop_flag = self.get_stop_flag()
        if stop_flag != openvino_genai.StreamingStatus.RUNNING:
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()

        return stop_flag

    def compute_decoded_length_for_position(self, cache_position: int):
        # decode was performed for this position, skippping
        """
        Computes and updates the decoded text length at a given token cache position.

        Marks the position as incomplete (-1) if decoding ends in a replacement character.

        Args:
            cache_position (int): Index in the token cache.
        """
        if self.decoded_lengths[cache_position] != -2:
            return

        cache_for_position = self.tokens_cache[: cache_position + 1]
        text_for_position = self.tokenizer.decode(cache_for_position)

        if len(text_for_position) > 0 and text_for_position[-1] == REPLACEMENT_CHAR:
            # Mark text as incomplete
            self.decoded_lengths[cache_position] = -1
        else:
            self.decoded_lengths[cache_position] = len(text_for_position)

    def end(self) -> openvino_genai.StreamingStatus:
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len :]
            self.write_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.text_queue.put(None)
        return openvino_genai.StreamingStatus.STOP


class ChunkStreamer(IterableStreamer):
    """
    A streamer that emits tokens in fixed-size chunks.

    This is useful for controlling the granularity of streamed output,
    such as generating one chunk per N tokens.
    """

    def __init__(self, tokenizer, tokens_len):
        """
        Initializes the ChunkStreamer with a tokenizer and chunk size.

        Args:
            tokenizer: Tokenizer used for decoding tokens.
            tokens_len (int): Number of tokens per chunk.
        """
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def write(self, token: Union[int, list[int]]) -> openvino_genai.StreamingStatus:
        """
        Writes a token or list of tokens, emitting output every `tokens_len` tokens.

        Args:
            token (Union[int, list[int]]): Token(s) to add to the cache.

        Returns:
            openvino_genai.StreamingStatus: The current streaming status.
        """
        if (len(self.tokens_cache) + 1) % self.tokens_len == 0:
            return super().write(token)

        if type(token) is list:
            self.tokens_cache += token
            # -2 means no decode was done for this token position
            self.decoded_lengths += [-2 for _ in range(len(token))]
        else:
            self.tokens_cache.append(token)
            self.decoded_lengths.append(-2)

        return openvino_genai.StreamingStatus.RUNNING


def get_model_name(model_path: str) -> str:
    """
    Extracts the model name from the given model path.

    Args:
        model_path (str): The path to the model.

    Returns:
        str: The name of the model.
    """
    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {model_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        return "placeholder_model_name"

    return config.get("_name_or_path", "placeholder_model_name")


class _ParametersModel(BaseModel):
    temperature: float = Field(..., ge=0.0, le=2.0)
    top_p: float = Field(..., ge=0.0, le=1.0)
    top_k: int = Field(..., ge=0)
    max_tokens: int = Field(..., gt=0)
    device: str
    do_sample: bool

    @field_validator("device")
    def validate_device(cls, v):
        if v not in {"CPU", "GPU", "NPU"}:
            raise ValueError(f"Unsupported device: {v}. Supported: CPU, GPU, NPU")
        return v


def validate_parameters(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    device: str,
    do_sample: bool,
):
    """
    Validates the parameters for OpenVINO chat models.

    Args:
        temperature (float): Temperature for sampling.
        top_p (float): Top-p sampling parameter.
        top_k (int): Top-k sampling parameter.
        max_tokens (int): Maximum number of tokens to generate.
        device (str): Device to run the model on.
        do_sample (bool): Whether to use sampling.

    """
    _ParametersModel(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        device=device,
        do_sample=do_sample,
    )


def decrypt_model(model_dir, model_file_name, weights_file_name, dtype=np.uint8):
    """
    Loads and (optionally) decrypts the OpenVINO model and weights files.

    Args:
        model_dir (str): Path to the directory containing model files.
        model_file_name (str): Name of the model structure file (.xml).
        weights_file_name (str): Name of the model weights file (.bin).

    Returns:
        Tuple[str, Tensor]: The model IR (XML as string) and weights (as OpenVINO Tensor).
    """
    with open(model_dir + "/" + model_file_name, "r") as file:
        model = file.read()
    # decrypt model

    with open(model_dir + "/" + weights_file_name, "rb") as file:
        binary_data = file.read()
    # decrypt weights
    weights = np.frombuffer(binary_data, dtype=dtype).astype(dtype)

    return model, Tensor(weights)


def read_tokenizer(model_dir):
    """
    Loads and returns a Tokenizer instance using decrypted tokenizer and detokenizer files.

    Assumes presence of:
        - openvino_tokenizer.xml/bin
        - openvino_detokenizer.xml/bin

    Args:
        model_dir (str): Path to the model directory containing tokenizer files.

    Returns:
        openvino_genai.Tokenizer: An initialized tokenizer instance.
    """
    tokenizer_model_name = "openvino_tokenizer.xml"
    tokenizer_weights_name = "openvino_tokenizer.bin"
    tokenizer_model, tokenizer_weights = decrypt_model(
        model_dir, tokenizer_model_name, tokenizer_weights_name
    )

    detokenizer_model_name = "openvino_detokenizer.xml"
    detokenizer_weights_name = "openvino_detokenizer.bin"
    detokenizer_model, detokenizer_weights = decrypt_model(
        model_dir, detokenizer_model_name, detokenizer_weights_name
    )

    return openvino_genai.Tokenizer(
        tokenizer_model, tokenizer_weights, detokenizer_model, detokenizer_weights
    )


def process_image_content(content_item: Dict[str, Any]) -> ov.Tensor:
    """
    Process image content from message and convert to OpenVINO Tensor.

    Args:
        content_item: Dictionary containing image data (URL, base64, or path)

    Returns:
        ov.Tensor: Image data as OpenVINO tensor
    """
    if content_item["type"] != "image_url":
        raise ValueError("Expected image_url content type")

    image_url = content_item["image_url"]["url"]

    try:
        if image_url.startswith("data:image"):
            header, data = image_url.split(",", 1)
            image_data = base64.b64decode(data)
            image = Image.open(BytesIO(image_data))
        elif image_url.startswith(("http://", "https://")):
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(image_url):
                raise FileNotFoundError(f"Image file not found: {image_url}")
            image = Image.open(image_url)

        image_array = np.array(image)
        return ov.Tensor(image_array)
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}")


def process_audio_content(content_item: Dict[str, Any]) -> ov.Tensor:
    """
    Process audio content.

    Args:
        content_item: Dictionary containing audio data (URL, base64, or path)

    Returns:
        raw audio: Audio data (16kHz, normalized)
    """
    if content_item["type"] != "audio_url":
        raise ValueError("Expected audio_url content type")

    audio_url = content_item["audio_url"]["url"]

    try:
        if audio_url.startswith("data:audio"):
            header, data = audio_url.split(",", 1)
            audio_data = base64.b64decode(data)
            audio_bytes = BytesIO(audio_data)
            raw_speech, _ = librosa.load(audio_bytes, sr=16000)
        elif audio_url.startswith(("http://", "https://")):
            response = requests.get(audio_url)
            response.raise_for_status()
            audio_bytes = BytesIO(response.content)
            raw_speech, _ = librosa.load(audio_bytes, sr=16000)
        else:
            if not os.path.exists(audio_url):
                raise FileNotFoundError(f"Audio file not found: {audio_url}")
            raw_speech, _ = librosa.load(audio_url, sr=16000)

        # Convert to OpenVINO tensor
        return raw_speech
    except Exception as e:
        raise RuntimeError(f"Failed to process audio: {e}")


def extract_data(
    message: BaseMessage, is_vlm: bool, is_whisper: bool
) -> tuple[str, List[ov.Tensor], List]:
    """
    Extract text and images from a message.

    Args:
        message: LangChain message object

    Returns:
        tuple: (text_content, list_of_image_tensors, list_of_audios)
    """
    if isinstance(message, str):
        return message, [], []
    if isinstance(message.content, str):
        return message.content, [], []
    if isinstance(message.content, list):
        text_parts = []
        images = []
        audios = []
        for content_item in message.content:
            if isinstance(content_item, str):
                text_parts.append(content_item)
            elif isinstance(content_item, dict):
                if content_item.get("type") == "text":
                    text_parts.append(content_item["text"])
                elif content_item.get("type") == "image_url":
                    if is_vlm:
                        image_tensor = process_image_content(content_item)
                        images.append(image_tensor)
                    else:
                        raise ValueError("Images not supported for text-only models")
                elif content_item.get("type") == "audio_url":
                    if is_whisper:
                        audio_tensor = process_audio_content(content_item)
                        audios.append(audio_tensor)
                    else:
                        raise ValueError("Audio not supported for text-only models")
        return " ".join(text_parts), images, audios
    raise ValueError(f"Unsupported message content type: {type(message.content)}")


VLM_MODEL_TYPES = {
    "minicpmv",
    "llava",
    "llava_next",
    "internvl_chat",
    "phi3_v",
    "phi4mm",
    "qwen2_vl",
    "qwen2_5_vl",
}

VLM_INDICATORS = [
    "vision",
    "vlm",
    "multimodal",
    "internvl",
    "llava",
    "blip",
    "clip",
    "qwen-vl",
    "cogvlm",
    "minigpt",
    "phi3v",
]

VISION_FILES = [
    "vision_model.xml",
    "image_encoder.xml",
    "visual_encoder.xml",
    "vision_tower.xml",
    "image_projection.xml",
]


def detect_vlm_model(model_path: str) -> bool:
    """
    Detect if the model is a Vision-Language Model.

    Returns:
        bool: True if the model is a VLM, False otherwise.
    """
    processor_config_path = os.path.join(model_path, "processor_config.json")
    if os.path.exists(processor_config_path):
        try:
            processor_config_path = os.path.join(model_path, "processor_config.json")
            if os.path.exists(processor_config_path):
                with open(processor_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                if model_type in VLM_MODEL_TYPES:
                    return True

            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                model_type = config.get("model_type", "").lower()
                architectures = config.get("architectures", [])
                if any(ind in model_type for ind in VLM_INDICATORS):
                    return True
                if any(
                    any(ind in arch.lower() for ind in VLM_INDICATORS)
                    for arch in architectures
                ):
                    return True

            if any(os.path.exists(os.path.join(model_path, f)) for f in VISION_FILES):
                return True
        except Exception as vlm_error:
            print(f"Not a VLM model: {vlm_error}")
    else:
        print("No processor_config.json found - not a VLM model")
    return False


WHISPER_MODEL_TYPES = {
    "whisper",
}

WHISPER_INDICATORS = [
    "whisper",
    "speech",
    "audio",
    "asr",
    "transcription",
    "speech-to-text",
]

WHISPER_FILES = [
    "encoder_model.xml",
    "decoder_model.xml",
    "decoder_model_merged.xml",
    "decoder_with_past_model.xml",
    "audio_encoder.xml",
    "text_decoder.xml",
]

WHISPER_PROCESSOR_CLASSES = {
    "whisperprocessor",
    "whisperfeaturesextractor",
}


def detect_whisper_model(model_path: str) -> bool:
    """
    Detect if the model is a Whisper model.

    Args:
        model_path (str): Path to the model directory

    Returns:
        bool: True if the model is a Whisper model, False otherwise.
    """
    try:
        preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            processor_class = config.get("processor_class", "").lower()
            if processor_class in WHISPER_PROCESSOR_CLASSES:
                return True
            feature_extractor_type = config.get("feature_extractor_type", "").lower()
            if "whisper" in feature_extractor_type:
                return True

        processor_config_path = os.path.join(model_path, "processor_config.json")
        if os.path.exists(processor_config_path):
            with open(processor_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            processor_class = config.get("processor_class", "").lower()
            if processor_class in WHISPER_PROCESSOR_CLASSES:
                return True

            model_type = config.get("model_type", "").lower()
            if model_type in WHISPER_MODEL_TYPES:
                return True

        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            model_type = config.get("model_type", "").lower()
            architectures = config.get("architectures", [])

            if model_type in WHISPER_MODEL_TYPES:
                return True

            if any(ind in model_type for ind in WHISPER_INDICATORS):
                return True

            if any(
                any(ind in arch.lower() for ind in WHISPER_INDICATORS)
                for arch in architectures
            ):
                return True
    except Exception as whisper_error:
        print(f"Error detecting Whisper model: {whisper_error}")
        return False

    return False
