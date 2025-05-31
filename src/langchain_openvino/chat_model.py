from typing import Any, Dict, List, Optional, Iterator
from queue import Queue
from threading import Thread
from time import perf_counter
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages.ai import UsageMetadata
from pydantic import Field
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
    _pipeline: Any = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = ov_genai.LLMPipeline(self.model_path, self.device)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response to the given messages."""
        msg = messages[-1].content
        
        resp = self._pipeline.generate(
            msg, 
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            do_sample=kwargs.get("do_sample", self.do_sample),
        )

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
        configuration = self._pipeline.get_generation_config()
        configuration.max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        configuration.temperature = kwargs.get("temperature", self.temperature)
        configuration.top_k = kwargs.get("top_k", self.top_k)
        configuration.top_p = kwargs.get("top_p", self.top_p)
        configuration.do_sample = kwargs.get("do_sample", self.do_sample)
        tokens_len = kwargs.get("tokens_len", 10)
        token_streamer = ChunkStreamer(self._pipeline.get_tokenizer(), tokens_len=tokens_len)
        def generate():
            self._pipeline.generate(msg, configuration, token_streamer)
        generation_thread = Thread(target=generate, daemon=True)
        generation_thread.start()
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

