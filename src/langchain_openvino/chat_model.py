from typing import Any, Dict, List, Optional, Iterator
from threading import Thread
import os
import openvino_genai as ov_genai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import PrivateAttr
from .utils import (
    ChunkStreamer,
    get_model_name,
    validate_parameters,
    decrypt_model,
    read_tokenizer,
)


class ChatOpenVINO(BaseChatModel):
    """
    LangChain-compatible chat model wrapper using OpenVINO for efficient inference.

    This class wraps around an OpenVINO-compatible LLM pipeline and exposes
    a LangChain-compatible interface. It supports draft models, LoRA adapters,
    encrypted models, and streaming.

    Attributes:
        model_path (str): Path to the OpenVINO IR model directory.
        device (str): Target device for inference (e.g., "CPU", "GPU").
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
        do_sample (bool): Whether to sample from the distribution or greedy decode.
        use_encrypted_model (bool): Whether the model is encrypted.
        prompt_lookup (bool): Enable prompt-based token reuse or guidance.
        draft_model_path (Optional[str]): Path to draft model for speculative decoding.
        adapter_path (Optional[str]): Path to LoRA adapter.
        adapter_alpha (float): Scaling factor for LoRA adapter.
    """

    model_path: str
    device: str = "CPU"
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    use_encrypted_model: bool = False
    prompt_lookup: bool = False
    draft_model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    adapter_alpha: float = 0.75
    _pipeline: Any = PrivateAttr()

    def __init__(self, **kwargs):
        """
        Initialize the ChatOpenVINO instance and load the model pipeline.

        Raises:
            FileNotFoundError: If the provided model path or draft/adapter path does not exist.
            NotADirectoryError: If the given paths are not directories.
            RuntimeError: If the pipeline fails to initialize.
        """
        super().__init__(**kwargs)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Provided model path does not exist: {self.model_path}"
            )
        if not os.path.isdir(self.model_path):
            raise NotADirectoryError(
                f"Model path must be a directory: {self.model_path}"
            )
        self._validate_parameters()
        if self.use_encrypted_model:
            self.model, self.weights = decrypt_model(
                self.model_path, "openvino_model.xml", "openvino_model.bin"
            )
            self.tokenizer = read_tokenizer(self.model_path)
        try:
            if self.draft_model_path:
                if not os.path.exists(self.draft_model_path):
                    raise FileNotFoundError(
                        f"Draft model path does not exist: {self.draft_model_path}"
                    )
                if not os.path.isdir(self.draft_model_path):
                    raise NotADirectoryError(
                        f"Draft model path must be a directory: {self.draft_model_path}"
                    )
                draft_model = ov_genai.draft_model(self.draft_model_path, self.device)
                self._pipeline = ov_genai.LLMPipeline(
                    self.model_path, self.device, draft_model=draft_model
                )
            elif self.adapter_path:
                if not os.path.exists(self.adapter_path):
                    raise FileNotFoundError(
                        f"Adapter path does not exist: {self.adapter_path}"
                    )
                adapter = ov_genai.Adapter(self.adapter_path)
                adapter_config = ov_genai.AdapterConfig(adapter)
                if self.use_encrypted_model:
                    self._pipeline = ov_genai.LLMPipeline(
                        self.model,
                        self.weights,
                        self.tokenizer,
                        self.device,
                        adapter=adapter_config,
                    )
                else:
                    self._pipeline = ov_genai.LLMPipeline(
                        self.model_path, self.device, adapter=adapter_config
                    )
            else:
                if self.use_encrypted_model:
                    self._pipeline = ov_genai.LLMPipeline(
                        self.model, self.weights, self.tokenizer, self.device
                    )
                else:
                    self._pipeline = ov_genai.LLMPipeline(self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenVINO pipeline: {e}")

    def _validate_parameters(self):
        """
        Validates the current model configuration parameters to ensure compatibility and correctness.

        Raises:
            ValueError: If any parameter is outside the allowed range or invalid for the target device.
        """
        validate_parameters(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            device=self.device,
            do_sample=self.do_sample,
        )

    def with_temperature(self, temperature: float):
        """
        Set the temperature for generation.

        Args:
            temperature (float): Sampling temperature value.

        Returns:
            ChatOpenVINO: Self with updated configuration.
        """
        self.temperature = temperature
        self._validate_parameters()
        return self

    def with_top_p(self, top_p: float):
        """
        Set the top-p value for generation.

        Only tokens with cumulative probability up to `top_p` are considered.

        Args:
            top_p (float): Value between 0 and 1 for sampling.

        Returns:
            ChatOpenVINO: The updated model instance.
        """
        self.top_p = top_p
        self._validate_parameters()
        return self

    def with_top_k(self, top_k: int):
        """
        Set the top-k value for generation.

        This controls how many top tokens are considered when picking the next word.
        Lower values make output more predictable, higher values make it more random.

        Args:
            top_k (int): Number of top tokens to sample from.

        Returns:
            ChatOpenVINO: The updated model instance.
        """
        self.top_k = top_k
        self._validate_parameters()
        return self

    def with_max_tokens(self, max_tokens: int):
        """
        Set the maximum number of tokens to generate.

        Args:
            max_tokens (int): Max tokens in the response.

        Returns:
            ChatOpenVINO: The updated model instance.
        """
        self.max_tokens = max_tokens
        self._validate_parameters()
        return self

    def with_do_sample(self, do_sample: bool):
        """
        Turn sampling on or off.

        If False, the model uses greedy decoding. If True, it samples from possible outputs.

        Args:
            do_sample (bool): Enable or disable sampling.

        Returns:
            ChatOpenVINO: The updated model instance.
        """
        self.do_sample = do_sample
        self._validate_parameters()
        return self

    def with_device(self, device: str):
        """
        Set the device to run the model on.

        Examples: "CPU", "GPU", "AUTO", etc.

        Args:
            device (str): Name of the device.

        Returns:
            ChatOpenVINO: The updated model instance.
        """
        self.device = device
        self._validate_parameters()
        return self

    def with_prompt_lookup(self, prompt_lookup: bool):
        """
        Enable or disable prompt-based token reuse or guidance.

        Args:
            prompt_lookup (bool): Whether to enable prompt lookup.

        Returns:
            ChatOpenVINO: The updated model instance.
        """
        self.prompt_lookup = prompt_lookup
        return self

    def _prepare_generation_config(self, **kwargs: Any):
        """
        Prepare generation configuration based on class attributes and runtime overrides.

        Args:
            **kwargs: Optional runtime overrides for generation parameters.

        Returns:
            Any: An OpenVINO-compatible generation config object.
        """
        config = self._pipeline.get_generation_config()
        config.max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        config.temperature = kwargs.get("temperature", self.temperature)
        config.top_k = kwargs.get("top_k", self.top_k)
        config.top_p = kwargs.get("top_p", self.top_p)
        config.do_sample = kwargs.get("do_sample", self.do_sample)
        if self.prompt_lookup:
            config.num_assistant_tokens = kwargs.get("num_assistant_tokens", 5)
            config.max_ngram_size = kwargs.get("max_ngram_size", 3)
        if self.draft_model_path:
            config.num_assistant_tokens = kwargs.get("num_assistant_tokens", 5)
        if self.adapter_path:
            config.adapters = ov_genai.AdapterConfig(
                (ov_genai.Adapter(self.adapter_path), self.adapter_alpha)
            )
        return config

    def _generate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response to the given list of chat messages.

        Args:
            messages (List[BaseMessage]): Chat history including latest user prompt.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LangChain.
            stop (Optional[List[str]]): Optional stop tokens.
            **kwargs: Optional overrides for generation configuration.

        Returns:
            ChatResult: Generated response wrapped in LangChain's ChatResult.
        """
        if not messages or not messages[-1].content:
            raise ValueError("No input message provided for generation.")
        msg = messages[-1].content
        configuration = self._prepare_generation_config(**kwargs)
        try:
            resp = self._pipeline.generate(
                msg,
                configuration,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
        gen = ChatGeneration(message=AIMessage(content=resp))
        return ChatResult(generations=[gen])

    def _stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream tokens for the response to the given messages.

        Args:
            messages (List[BaseMessage]): Chat history including latest user prompt.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LangChain.
            stop (Optional[List[str]]): Optional stop tokens.
            **kwargs: Optional overrides (e.g., `tokens_len` for streaming granularity).

        Yields:
            Iterator[ChatGenerationChunk]: Stream of response tokens.
        """
        msg = messages[-1].content
        configuration = self._prepare_generation_config(**kwargs)
        tokens_len = kwargs.get("tokens_len", 10)
        try:
            token_streamer = ChunkStreamer(
                self._pipeline.get_tokenizer(), tokens_len=tokens_len
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create token streamer: {e}")

        def generate():
            self._pipeline.generate(msg, configuration, token_streamer)

        generation_thread = Thread(target=generate, daemon=True)
        try:
            generation_thread.start()
        except Exception as e:
            raise RuntimeError(f"Failed to start generation thread: {e}")
        for token_chunk in token_streamer:
            if run_manager:
                run_manager.on_llm_new_token(token_chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=token_chunk))
        generation_thread.join()

    @property
    def _llm_type(self) -> str:
        """Identifier."""
        return "openvino-llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: Dictionary of identifying model parameters.
        """
        return {
            "model_name": get_model_name(self.model_path),
            "model_path": self.model_path,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }
