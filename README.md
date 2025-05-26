# OpenVINO LangChain Integration

This project integrates LangChain with OpenVINO GenAI to enable efficient local inference.

## Features

- **OpenVINO Optimized**: Leverages Intel's OpenVINO toolkit for efficient inference
- **Flexible Configuration**: Customizable generation parameters (temperature, top-k, top-p, etc.)

## Installation

This project uses Poetry for dependency management. Make sure you have Poetry installed, then:

```bash
# Clone the repository
git clone <repository-url>
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

```python
from chat_openvino import ChatOpenVINO
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

# Testing
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests
    
```

### Configuration Parameters

- `model_path` (str): Path to the OpenVINO model directory
- `device` (str): Target device for inference (default: "CPU")
- `max_tokens` (int): Maximum number of tokens to generate (default: 256)
- `temperature` (float): Sampling temperature (default: 1.0)
- `top_k` (int): Top-k sampling parameter (default: 50)
- `top_p` (float): Top-p (nucleus) sampling parameter (default: 0.95)
- `do_sample` (bool): Whether to use sampling (default: True)

