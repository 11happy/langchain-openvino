from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
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
ADAPTER_PATH = CUR_DIR.parent.parent / "adapter"
MODEL_PATH = Path(os.getenv("OPENVINO_MODEL_PATH", DEFAULT_MODEL_PATH))

model = OVModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B", export=True)
model.save_pretrained(str(MODEL_PATH))
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
save_model(ov_tokenizer, str(MODEL_PATH / "openvino_tokenizer.xml"))
save_model(ov_detokenizer, str(MODEL_PATH / "openvino_detokenizer.xml"))

snapshot_download(
    repo_id="DevQuasar/llama3.2_1b_chat_brainstorm-v3.2.1_adapter",
    local_dir=ADAPTER_PATH,
    local_dir_use_symlinks=False,
)


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


def test_with_lora_adapter():
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
        adapter_path=str(ADAPTER_PATH / "adapter_model.safetensors"),
    )
    response = chat_model.invoke("Hello, world!")
    assert response is not None


import shutil


def test_cleanup():
    model_path = str(MODEL_PATH)
    adapter_path = str(ADAPTER_PATH)
    shutil.rmtree(adapter_path, ignore_errors=True)
    shutil.rmtree(model_path, ignore_errors=True)
