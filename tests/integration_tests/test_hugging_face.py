from transformers import AutoTokenizer, AutoModelForCausalLM
from src.langchain_openvino.chat_model import ChatOpenVINO
from src.langchain_openvino.utils import get_model_name
from langchain_core.messages import HumanMessage
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForCausalLM
from openvino import save_model
from pathlib import Path
import shutil
import os


def test_compare_with_hf():
    MODEL_ID = "gpt2"
    PROMPT = "The capital of France is"
    MAX_NEW_TOKENS = 32
    CUR_DIR = Path(__file__).parent
    MODEL_PATH = CUR_DIR.parent.parent / "models" / "temp_ov_model"
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # hf
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    encoded_input = tokenizer.encode(
        PROMPT, return_tensors="pt", add_special_tokens=False
    )
    hf_output_ids = model.generate(
        encoded_input, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
    )
    hf_output = tokenizer.decode(
        hf_output_ids[0, encoded_input.shape[1] :], skip_special_tokens=True
    ).strip()

    # openvino
    ov_model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    ov_model.save_pretrained(str(MODEL_PATH))
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
    save_model(ov_tokenizer, str(MODEL_PATH / "openvino_tokenizer.xml"))
    save_model(ov_detokenizer, str(MODEL_PATH / "openvino_detokenizer.xml"))
    chat_model = ChatOpenVINO(
        model_path=str(MODEL_PATH),
        device="CPU",
        max_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=False,
    )

    response = chat_model.invoke([HumanMessage(content=PROMPT)])
    ov_output = response.content.strip()
    print(f"HF Output: {hf_output}")
    print(f"OV Output: {ov_output}")
    assert hf_output == ov_output
    shutil.rmtree(str(MODEL_PATH), ignore_errors=True)
