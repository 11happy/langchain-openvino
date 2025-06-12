from typing import Type
import pytest
import os
from pathlib import Path
from src.langchain_openvino.chat_model import ChatOpenVINO
from langchain_tests.unit_tests import ChatModelUnitTests
from src.langchain_openvino.utils import get_model_name

CUR_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = CUR_DIR.parent.parent / "models" / "newov_model"
MAIN_MODEL_PATH = CUR_DIR.parent.parent / "models" / "main-model"
DRAFT_MODEL_PATH = CUR_DIR.parent.parent / "models" / "draft-model"
MODEL_PATH = Path(os.getenv("OPENVINO_MODEL_PATH", DEFAULT_MODEL_PATH))

class TestChatOpenVINOLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOpenVINO]:
        return ChatOpenVINO

    @property
    def chat_model_params(self) -> dict:
        return {
           "model_name": get_model_name(str(MODEL_PATH)),
           "model_path": str(MODEL_PATH),
           "device": "CPU",
           "max_tokens": 256,
           "temperature": 1.0,
           "top_k": 50,
           "top_p": 0.95,
           "do_sample": True,
        }
    
def test_sample_chat_openvino():
    model = ChatOpenVINO(model_path=str(MODEL_PATH))
    assert model is not None
    assert model._llm_type == "openvino-llm"
    assert model._identifying_params == {
        "model_name": get_model_name(str(MODEL_PATH)),
        "model_path": str(MODEL_PATH),
        "device": "CPU",
        "max_tokens": 256,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    }
    response = model.invoke("Hello, world!")  
    assert response is not None

def test_speculative_chat_openvino():
    model = ChatOpenVINO(model_path=str(MAIN_MODEL_PATH),draft_model_path=str(DRAFT_MODEL_PATH),max_tokens=128)
    assert model is not None
    assert model._llm_type == "openvino-llm"
    assert model._identifying_params == {
        "model_name": get_model_name(str(MAIN_MODEL_PATH)),
        "model_path": str(MAIN_MODEL_PATH),
        "device": "CPU",
        "max_tokens": 128,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    }
    response = model.invoke("Hello, world!")
    assert response is not None