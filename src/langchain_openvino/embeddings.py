from typing import List, Optional
import openvino_genai as ov_genai
from langchain_core.embeddings import Embeddings


class OpenVINOEmbeddings(Embeddings):
    """LangChain-compatible wrapper for OpenVINO GenAI text embedding pipeline."""

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        normalize: Optional[bool] = True,
        pooling_type: Optional[str] = "CLS",
        max_length: Optional[int] = 256,
        query_instruction: Optional[str] = "Represent this query for retrieval: ",
        embed_instruction: Optional[str] = "Represent this document for retrieval: ",
    ):
        try:
            config = ov_genai.TextEmbeddingPipeline.Config(
                normalize=normalize,
                pooling_type=getattr(
                    ov_genai.TextEmbeddingPipeline.PoolingType, pooling_type
                ),
                max_length=max_length,
                query_instruction=query_instruction,
                embed_instruction=embed_instruction,
            )

            self._pipeline = ov_genai.TextEmbeddingPipeline(
                model_path,
                device,
                config,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenVINO TextEmbeddingPipeline: {e}"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                raise ValueError("`texts` list is empty.")
            return self._pipeline.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"Error while embedding documents: {e}")

    def embed_query(self, text: str) -> List[float]:
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("`text` must be a non-empty string.")
            return self._pipeline.embed_query(text)
        except Exception as e:
            raise RuntimeError(f"Error while embedding query: {e}")
