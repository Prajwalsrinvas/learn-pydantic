# Pydantic for LLM Workflows 🔧⚡

- Code files from the DeepLearning.AI [short course on Pydantic for LLM Workflows](https://www.deeplearning.ai/short-courses/pydantic-for-llm-workflows/)
- Instructor: Ryan Keenan, Director of the Learning Experience Lab at DeepLearning.AI
- [Certificate](https://learn.deeplearning.ai/accomplishments/1337b44b-f412-44cc-93fd-c66fc018b640)

## Why Pydantic? 🧩

Pydantic transforms how we build LLM-powered applications by bringing structure, reliability, and validation to AI workflows:

- **Structured Output** - Move beyond free-form LLM responses to predictable, validated data structures
- **Data Validation** - Catch issues like badly formatted emails or missing fields before they cause problems
- **Type Safety** - Use Python type hints to define expected data shapes and get automatic validation
- **LLM Integration** - Work seamlessly with modern LLM providers and frameworks for structured responses
- **Reliability** - Ensure LLM responses are complete, correctly formatted, and ready to use in your applications
- **Ecosystem** - Leverage one of Python's most popular packages (300M+ downloads/month) used by FastAPI, LangChain, PydanticAI, and more

## Core Pydantic Concepts 🛠️

### 1. BaseModel - Define Data Structure
```python
from pydantic import BaseModel, EmailStr

class UserInput(BaseModel):
    name: str
    email: EmailStr
    query: str
```

### 2. Validation - Ensure Data Quality
```python
from pydantic import Field, ValidationError
from typing import Literal, Optional

class CustomerQuery(UserInput):
    priority: str = Field(..., description="Priority level: low, medium, high")
    category: Literal['refund_request', 'information_request', 'other']
    order_id: Optional[int] = Field(None, ge=10000, le=99999)
```

### 3. LLM Integration - Structured API Responses
```python
# OpenAI Structured Output
response = openai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format=CustomerQuery
)

# Anthropic with Instructor
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())
response = client.messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
    response_model=CustomerQuery
)
```

| No. | Concepts | NBSanity | GitHub |
|-----|----------|----------|--------|
| 1 | • Pydantic BaseModel fundamentals<br>• Data validation with type hints<br>• Error handling and validation errors<br>• JSON parsing and serialization<br>• Field constraints and optional fields | [![Open In NBSanity](https://nbsanity.com/assets/icon.png)](https://nbsanity.com/Prajwalsrinvas/learn-pydantic/blob/main/1_pydantic_basics.ipynb) | [![GitHub](https://cdn-icons-png.flaticon.com/32/270/270798.png)](https://github.com/Prajwalsrinvas/learn-pydantic/blob/main/1_pydantic_basics.ipynb) |
| 2 | • Structured LLM output generation<br>• Retry mechanisms for validation errors<br>• Prompt engineering for structured responses<br>• CustomerQuery model inheritance<br>• Validation feedback loops | [![Open In NBSanity](https://nbsanity.com/assets/icon.png)](https://nbsanity.com/Prajwalsrinvas/learn-pydantic/blob/main/2_validating_llm_responses.ipynb) | [![GitHub](https://cdn-icons-png.flaticon.com/32/270/270798.png)](https://github.com/Prajwalsrinvas/learn-pydantic/blob/main/2_validating_llm_responses.ipynb) |
| 3 | • Direct API integration patterns<br>• OpenAI structured output API<br>• Anthropic with Instructor library<br>• PydanticAI agent framework<br>• Multiple LLM provider strategies | [![Open In NBSanity](https://nbsanity.com/assets/icon.png)](https://nbsanity.com/Prajwalsrinvas/learn-pydantic/blob/main/3_structured_llm_output.ipynb) | [![GitHub](https://cdn-icons-png.flaticon.com/32/270/270798.png)](https://github.com/Prajwalsrinvas/learn-pydantic/blob/main/3_structured_llm_output.ipynb) |
| 4 | • Tool calling with Pydantic schemas<br>• Function parameter validation<br>• Customer support automation<br>• FAQ lookup and order status tools<br>• End-to-end workflow integration | [![Open In NBSanity](https://nbsanity.com/assets/icon.png)](https://nbsanity.com/Prajwalsrinvas/learn-pydantic/blob/main/4_tool_calling.ipynb) | [![GitHub](https://cdn-icons-png.flaticon.com/32/270/270798.png)](https://github.com/Prajwalsrinvas/learn-pydantic/blob/main/4_tool_calling.ipynb) |

## Resources 📚

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic GitHub Repository](https://github.com/pydantic/pydantic)
- [PydanticAI Framework](https://ai.pydantic.dev/) - Agent framework for production-grade LLM applications
- [Instructor Library](https://python.useinstructor.com/) - Structured outputs for LLMs
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework built on Pydantic