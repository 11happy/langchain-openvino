from typing import Type
import pytest
import os
from pathlib import Path
import openvino as ov
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
    model = ChatOpenVINO(
        model_path=str(MAIN_MODEL_PATH),
        draft_model_path=str(DRAFT_MODEL_PATH),
        max_tokens=128,
    )
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


def test_model_with_method_chaining():
    model = ChatOpenVINO(model_path=str(MODEL_PATH))
    assert model is not None
    assert model._llm_type == "openvino-llm"
    model = (
        model.with_temperature(0.7).with_top_p(0.9).with_top_k(40).with_max_tokens(128)
    )
    assert model.temperature == 0.7
    assert model.top_p == 0.9
    assert model.top_k == 40
    assert model.max_tokens == 128
    response = model.invoke("Hello, world!")
    assert response is not None


def test_model_with_prompt_lookup():
    model = ChatOpenVINO(model_path=str(MODEL_PATH), prompt_lookup=True)
    assert model is not None
    assert model._llm_type == "openvino-llm"
    assert model.prompt_lookup is True
    response = model.invoke("Hello, world!")
    assert response is not None


def test_empty_input():
    model = ChatOpenVINO(model_path=str(MODEL_PATH))
    with pytest.raises(ValueError, match="No input message provided for generation."):
        model.invoke("")


@pytest.mark.parametrize("bad_input", [None, 1234, {}, [], True])
def test_non_string_input(bad_input):
    model = ChatOpenVINO(model_path=str(MODEL_PATH))
    with pytest.raises(Exception):
        model.invoke(bad_input)


def test_long_prompt():
    model = ChatOpenVINO(model_path=str(MODEL_PATH), max_tokens=128)
    long_prompt = " ".join(["longinput"] * 2048)
    response = model.invoke(long_prompt)
    assert response is not None


def test_invalid_device():
    core = ov.Core()
    available_devices = core.available_devices
    if "NPU" in available_devices:
        pytest.skip("Skipping test: 'NPU' is available on this system.")
    with pytest.raises(Exception):
        ChatOpenVINO(model_path=str(MODEL_PATH), device="NPU")


def test_invalid_model_path():
    with pytest.raises(FileNotFoundError):
        ChatOpenVINO(model_path="invalid/path/to/model")
