# ChatPromptTemplate and Language Model Workflow

## Overview
This document explains the usage of `ChatPromptTemplate` with a language model, specifically the `qwen:1.8b` model from Ollama, to generate structured and processed responses for user queries.

---

## Steps to Implement

### 1. Import Required Modules
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
```
- `ChatPromptTemplate`: Creates structured conversation templates.
- `StrOutputParser`: Converts raw model outputs into plain text.
- `Ollama`: Provides access to the qwen:1.8b language model.

### 2. Create a Prompt Template
```python
prompt = ChatPromptTemplate.from_messages(
          [
                    ("system", "You are a helpful assistant. Please respond to the user queries"),
                    ("user", "Question:{question}")
          ]
)
```
- **System Message**: Defines the assistant's behavior as helpful.
- **User Message**: Contains a placeholder `{question}` for dynamic input.

**Example:**
```plaintext
System: "You are a helpful assistant. Please respond to the user queries."
User: "Question: [input text]"
```

### 3. Define User Input
```python
input_text = "overview about python"
```
Replace `"overview about python"` with any user query.

### 4. Initialize the Language Model
```python
llm = Ollama(model="qwen:1.8b")
```
- **Model**: Specifies the qwen:1.8b model to use.

### 5. Add an Output Parser
```python
output_parser = StrOutputParser()
```
Converts the model's response into plain text for easy readability.

### 6. Create a Processing Chain
```python
chain = prompt | llm | output_parser
```
Combines the prompt, language model, and output parser into a single pipeline.

### 7. Execute the Chain
```python
print(chain.invoke({"question": input_text}))
```
- **Invoke**: Replaces `{question}` with `input_text` and processes the prompt through the pipeline.

## Example Workflow

### Input
```plaintext
"overview about python"
```

### Prompt Generated
```plaintext
System: "You are a helpful assistant. Please respond to the user queries."
User: "Question: overview about python"
```

### Model Output (Raw)
```json
{
          "id": "12345",
          "object": "text_completion",
          "created": 1674384675,
          "model": "qwen:1.8b",
          "choices": [
                    {
                              "text": "\nPython is a versatile, high-level programming language known for its simplicity and readability. It is widely used in fields such as web development, data analysis, artificial intelligence, and scientific computing.",
                              "index": 0,
                              "logprobs": null,
                              "finish_reason": "stop"
                    }
          ],
          "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 42,
                    "total_tokens": 54
          }
}
```

### Parsed Output
```plaintext
Python is a versatile, high-level programming language known for its simplicity and readability. It is widely used in fields such as web development, data analysis, artificial intelligence, and scientific computing.
```
