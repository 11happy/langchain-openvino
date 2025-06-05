from typing import Any, Dict, List, Optional, Iterator
from threading import Thread
import os
import openvino_genai as ov_genai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import PrivateAttr
from .utils import ChunkStreamer, get_model_name


class ChatOpenVINO(BaseChatModel):
    """Chat model using OpenVINO for inference."""
    
    model_path: str
    device: str = "CPU"
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    draft_model_path: Optional[str] = None
    _pipeline: Any = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Provided model path does not exist: {self.model_path}")
        if not os.path.isdir(self.model_path):
            raise NotADirectoryError(f"Model path must be a directory: {self.model_path}")
        self._validate_parameters()
        try:
            if self.draft_model_path:
                if not os.path.exists(self.draft_model_path):
                    raise FileNotFoundError(f"Draft model path does not exist: {self.draft_model_path}")
                if not os.path.isdir(self.draft_model_path):
                    raise NotADirectoryError(f"Draft model path must be a directory: {self.draft_model_path}")
                draft_model = ov_genai.draft_model(self.draft_model_path, self.device)
                self._pipeline = ov_genai.LLMPipeline(self.model_path, self.device, draft_model=draft_model)
            else:
                self._pipeline = ov_genai.LLMPipeline(self.model_path, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenVINO pipeline: {e}")
        
    def _validate_parameters(self):
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")

        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be in [0.0, 1.0], got {self.top_p}")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError(f"top_k must be a non-negative integer, got {self.top_k}")

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {self.max_tokens}")

        if self.device not in {"CPU", "GPU", "NPU"}:
            raise ValueError(f"Unsupported device: {self.device}. Supported: CPU, GPU, NPU")

        if not isinstance(self.do_sample, bool):
            raise TypeError(f"do_sample must be a boolean, got {type(self.do_sample).__name__}")
    
    def _prepare_generation_config(self, **kwargs: Any):
        config = self._pipeline.get_generation_config()
        config.max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        config.temperature = kwargs.get("temperature", self.temperature)
        config.top_k = kwargs.get("top_k", self.top_k)
        config.top_p = kwargs.get("top_p", self.top_p)
        config.do_sample = kwargs.get("do_sample", self.do_sample)
        if self.draft_model_path:
            config.num_assistant_tokens = kwargs.get("num_assistant_tokens", 5)
        return config
    def _generate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response to the given messages."""
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
        """Stream the response to the given messages."""
        msg = messages[-1].content
        configuration = self._prepare_generation_config(**kwargs)
        tokens_len = kwargs.get("tokens_len", 10)
        try:
            token_streamer = ChunkStreamer(self._pipeline.get_tokenizer(), tokens_len=tokens_len)
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
        """Return identifying parameters."""
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

