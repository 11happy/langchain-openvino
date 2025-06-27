from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from src.langchain_openvino.chat_model import ChatOpenVINO
from langchain_tests.unit_tests import ChatModelUnitTests
from src.langchain_openvino.utils import get_model_name
from langchain_core.messages import (
    BaseMessageChunk,
)
from pathlib import Path
import os
from openvino import save_model
from openvino_tokenizers import convert_tokenizer

CUR_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = CUR_DIR.parent.parent / "models" / "ov_model"
MODEL_PATH = Path(os.getenv("OPENVINO_MODEL_PATH", DEFAULT_MODEL_PATH))

model = OVModelForCausalLM.from_pretrained("gpt2", export=True)
model.save_pretrained(str(MODEL_PATH))
tokenizer = AutoTokenizer.from_pretrained("gpt2")
ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
save_model(ov_tokenizer, str(MODEL_PATH / "openvino_tokenizer.xml"))
save_model(ov_detokenizer, str(MODEL_PATH / "openvino_detokenizer.xml"))


def test_invoke():
    model_path = str(MODEL_PATH)
    chat_model = ChatOpenVINO(
        model_name=get_model_name(model_path),
        model_path=model_path,
        device="CPU",
        max_tokens=256,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    response = chat_model.invoke("Hello, world!")
    assert response is not None


def test_stream():
    model_path = str(MODEL_PATH)
    chat_model = ChatOpenVINO(
        model_name=get_model_name(model_path),
        model_path=model_path,
        device="CPU",
        max_tokens=256,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    response = ""
    for chunk in chat_model.stream("Hello, world!"):
        assert isinstance(chunk, BaseMessageChunk)
        assert isinstance(chunk.content, str)
        response += chunk.content
    assert response is not None


import shutil


def test_cleanup():
    model_path = str(MODEL_PATH)
    shutil.rmtree(model_path, ignore_errors=True)
