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

from utils import OpenVINOStreamer


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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response to the given messages."""
    
    @property
    def _llm_type(self) -> str:
        """Identifier."""
        return "openvino-llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }

