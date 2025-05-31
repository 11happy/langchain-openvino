from typing import Type
import pytest
from src.langchain_openvino.chat_model import ChatOpenVINO
from langchain_tests.unit_tests import ChatModelUnitTests
from src.langchain_openvino.utils import get_model_name


class TestChatOpenVINOLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOpenVINO]:
        return ChatOpenVINO

    @property
    def chat_model_params(self) -> dict:
        return {
           "model_name": get_model_name(r"D:\newov_model"),
           "model_path": r"D:\newov_model",
           "device": "CPU",
           "max_tokens": 256,
           "temperature": 1.0,
           "top_k": 50,
           "top_p": 0.95,
           "do_sample": True,
        }
    
def test_sample_chat_openvino():
    model = ChatOpenVINO(model_path=r"D:\newov_model")
    assert model is not None
    assert model._llm_type == "openvino-llm"
    assert model._identifying_params == {
        "model_name": get_model_name(r"D:\newov_model"),
        "model_path": r"D:\newov_model",
        "device": "CPU",
        "max_tokens": 256,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    }
    response = model.invoke("Hello, world!")  
    assert response is not None