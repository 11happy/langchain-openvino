# OpenVINO LangChain Integration

This project integrates LangChain with OpenVINO GenAI to enable efficient local inference.

## Features

- **OpenVINO Optimized**: Leverages Intel's OpenVINO toolkit for efficient inference
- **Flexible Configuration**: Customizable generation parameters (temperature, top-k, top-p, etc.)

## Installation

This project uses Poetry for dependency management. Make sure you have Poetry installed, then:

```bash
# Clone the repository
git clone https://github.com/11happy/langchain-openvino.git
cd openvino-langchain

# Install dependencies
poetry install

```

### Dependencies

The main dependencies include:
- `openvino`: Intel's OpenVINO toolkit
- `openvino-genai`: OpenVINO generative AI extensions
- `langchain-core`: Core LangChain components
- `numpy`: Numerical computing
- `pydantic`: Data validation

## Usage

### Basic Usage

![final1](https://github.com/user-attachments/assets/fce9ad2b-9dbf-494e-a0fc-7f135ce45ef4)

#### Chat Model
```python
from langchain_openvino import ChatOpenVINO
from langchain_core.messages import HumanMessage

# Initialize the model
llm = ChatOpenVINO(
    model_path="path/to/your/openvino/model",
    device="CPU",
    max_tokens=256,
    temperature=0.7
)

# Generate a response
messages = [HumanMessage(content="Hello, how are you?")]
response = llm.invoke(messages)
print(response.content)

# Streaming Usage
for chunk in llm.stream("explain neural networks in simple terms"):
    print(chunk.content, end="", flush=True)
   
```

#### Embedding Model
```python
from langchain_openvino import OpenVINOEmbeddings

# Initialize embeddings
embeddings = OpenVINOEmbeddings(
    model_path="path/to/embedding/model",
    device="CPU",
    normalize=True,
    pooling_type="CLS"
)

# Embed documents
docs = ["This is a document", "Another document"]
doc_embeddings = embeddings.embed_documents(docs)

# Embed query
query = "What is this about?"
query_embedding = embeddings.embed_query(query)
```

#### Testing
#### Converting Models to OpenVINO Format
```bash

pip install optimum[openvino]

optimum-cli export openvino --model microsoft/DialoGPT-medium --task text-generation-with-past ./openvino-chat-model

optimum-cli export openvino --model sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction ./openvino-embedding-model
```
##### Downloading Sample/Test Models:
To run inference, you need to download an OpenVINO-compatible model:
```bash
# Sample model
huggingface-cli download OpenVINO/Qwen3-0.6B-int8-ov --local-dir models

```
> Make sure to configure the model path and model name in the test files
(e.g., tests/unit_tests/test_chat_model.py, similarly for integration tests).

```bash
# Unit Testing
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# Integration Testing
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/integration_tests

# Run all tests
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto
```

### Configuration Parameters

**Chat Model Parameters**

- `model_path (str)`:Path to the OpenVINO model directory

- `device (str)`:Target device for inference (default`:"CPU")

- `max_tokens (int)`:Maximum number of tokens to generate (default:256)

- `temperature (float)`:Sampling temperature (default:1.0)

- `top_k (int)`:Top-k sampling parameter (default:50)

- `top_p (float)`:Top-p (nucleus) sampling parameter (default:0.95)

- `do_sample (bool)`:Whether to use sampling (default:True)

- `use_encrypted_model (bool)`:Whether the model is encrypted (default:False)

- `prompt_lookup (bool)`:Enable prompt-based token reuse (default:False)

- `draft_model_path (Optional[str])`:Path to draft model for speculative decoding

- `adapter_path (Optional[str])`:Path to LoRA adapter

- `adapter_alpha (float)`:Scaling factor for LoRA adapter (default:0.75)

**Embedding Model Parameters**

- `model_path (str)`:Path to the OpenVINO embedding model directory

- `device (str)`:Target device for inference (default:"CPU")

- `normalize (bool)`:Whether to normalize embeddings (default:True)

- `pooling_type (str)`:Pooling strategy ("CLS", "MEAN", etc.) (default:"CLS")

- `max_length (int)`:Maximum sequence length (default:256)

- `query_instruction (str)`:Instruction prefix for queries

- `embed_instruction (str)`:Instruction prefix for documents

