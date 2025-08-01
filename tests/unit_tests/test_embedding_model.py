from huggingface_hub import snapshot_download
from pathlib import Path
import os
from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests
from langchain_openvino.embeddings import OpenVINOEmbeddings

CUR_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = CUR_DIR.parent.parent / "models" / "ov_model"
EMBEDDING_MODEL_PATH = CUR_DIR.parent.parent / "models" / "embedding_model"
MODEL_PATH = Path(os.getenv("OPENVINO_MODEL_PATH", DEFAULT_MODEL_PATH))

snapshot_download(
    repo_id="OpenVINO/bge-base-en-v1.5-int8-ov",
    local_dir=EMBEDDING_MODEL_PATH,
    local_dir_use_symlinks=False,
)


class TestOpenVINOEmbedding(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return OpenVINOEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model_path": str(EMBEDDING_MODEL_PATH),
            "device": "CPU",
            "query_instruction": "Represent this query for retrieval: ",
            "embed_instruction": "Represent this document for retrieval: ",
        }


def test_embedding_initialization() -> None:
    embeddings = OpenVINOEmbeddings(
        str(EMBEDDING_MODEL_PATH),
        "CPU",
    )
    assert embeddings is not None
    response = embeddings.embed_query("Hello, world!")
    assert response is not None
    document_embeddings = embeddings.embed_documents(["Document 1", "Document 2"])
    assert document_embeddings is not None


import shutil


def test_cleanup():
    model_path = str(MODEL_PATH)
    embedding_model_path = str(EMBEDDING_MODEL_PATH)
    shutil.rmtree(embedding_model_path, ignore_errors=True)
    shutil.rmtree(model_path, ignore_errors=True)
