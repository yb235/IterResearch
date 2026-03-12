# Data Schema

> This document describes every data structure used in IterResearch — inputs, outputs, messages, tool calls, and internal objects. If you're wondering "what does that field mean?" or "what shape is this data?", this is the place to look.

---

## Input Schema

### Input JSONL File

Each line is a JSON object with these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ Yes | The research question to investigate |
| `answer` | string | ❌ No | Ground truth answer (for evaluation) |
| `answers` | string | ❌ No | Alternative field name for ground truth |

**Example**:
```jsonc
// One question per line in a .jsonl file
{"question": "What are the latest advances in large language models?", "answer": ""}
{"question": "How does the transformer attention mechanism work?", "answer": ""}
{"question": "What is Markovian State Reconstruction?"}
```

**Notes**:
- If both `answers` and `answer` are present, `answers` takes priority (checked first in code)
- The `answer` field is optional and used only for evaluation — the agent doesn't see it
- Empty lines in the file are safely skipped

**Rationale**: The minimal schema (just `question`) makes it easy to get started. The optional `answer`/`answers` fields support evaluation workflows where you compare the agent's output against a known answer.

---

## Message Schema

IterResearch uses the OpenAI chat message format throughout. Three roles are used:

### User Message

```json
{
    "role": "user",
    "content": "You are a professional problem-solving agent... Question: ..."
}
```

- First turn: Contains the `initial_instruction_prompt` with the question, tools, and date injected
- Later turns: Contains the `instruction_prompt` or `last_instruction_prompt` with the compressed context

### Assistant Message

```json
{
    "role": "assistant",
    "content": "<report>\n### Status Report...\n</report>\n\n<tool_call>\n{\"name\": \"google_search\", \"arguments\": {\"query\": [\"...\"]}}\n</tool_call>"
}
```

The content always contains `<report>` tags and either `<tool_call>` or `<answer>` tags.

May also contain (removed by `clean_reasoning()`):
- `reasoning`: Raw reasoning text (model-specific)
- `reasoning_details`: Detailed reasoning metadata (model-specific)

### Tool Message

```json
{
    "role": "tool",
    "name": "google_search",
    "content": "A Google search for 'latest LLM advances' found 10 results:\n\n## Web Results\n1. [Title](link)..."
}
```

- `name`: The tool that produced this result
- `content`: The tool's output as a formatted string

**Rationale**: Using the standard OpenAI message format means the system works with any OpenAI-compatible LLM server out of the box. The additional `name` field on tool messages lets downstream processing know which tool generated each result.

---

## Tool Call Schema

When the agent wants to use a tool, it produces JSON inside `<tool_call>` tags:

```json
{
    "name": "tool_name",
    "arguments": {
        "param1": "value1",
        "param2": "value2"
    }
}
```

### google_search Tool Call

```json
{
    "name": "google_search",
    "arguments": {
        "query": ["search query 1", "search query 2"]
    }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | array of strings | ✅ | One or more search queries |

### google_scholar Tool Call

```json
{
    "name": "google_scholar",
    "arguments": {
        "query": ["academic search query 1"]
    }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | array of strings | ✅ | One or more academic search queries |

### Visit Tool Call

```json
{
    "name": "Visit",
    "arguments": {
        "url": ["https://example.com/page1", "https://example.com/page2"],
        "goal": "Find information about transformer architecture",
        "parse_type": "html"
    }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | array of strings | ✅ | — | URLs to visit |
| `goal` | string | ✅ | — | What information to extract |
| `parse_type` | `"html"` or `"pdf"` | ❌ | `"html"` | Content type hint |

### PythonInterpreter Tool Call

```json
{
    "name": "PythonInterpreter",
    "arguments": {
        "code": "import math\nresult = math.factorial(10)\nprint(result)"
    }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | ✅ | Python code to execute (use `print()` for output) |

**Rationale**: Each tool uses an array for its primary parameter (`query` for searches, `url` for visits) to enable batch operations in a single tool call. This is more efficient than requiring separate tool calls for each query/URL.

---

## Tool Response Schemas

### Search Response

```
A Google search for 'query text' found N results:

## Web Results
1. [Title](link)
Date published: 2025-01-15
Source: TechReview
Snippet text describing the result...

2. [Another Title](link)
Source: ArXiv
Another snippet...
```

Multiple queries are separated by `\n=======\n`.

### Scholar Response

```
A Google Scholar search for 'query text' found N results:

## Scholar Results
1. [Paper Title](pdfUrl: https://...) or (link: https://...)
publicationInfo: Journal Name, Author et al., 2025
citedBy: 342
Snippet text from the paper...

2. [Another Paper](link: ...)
publicationInfo: ...
citedBy: 56
...
```

### Visit Response

```
The useful information in 'https://example.com' for user goal 'find transformer info':

Evidence in page:
[Extracted evidence text, potentially multiple paragraphs]

Summary:
[Concise summary of the relevant information]
```

Multiple URLs are separated by `\n=======\n`.

### PythonInterpreter Response

```
stdout:
3628800

stderr:
[any error messages]
```

Or on error:
```
[PythonInterpreter Error] TimeoutError: Execution timed out.
```

---

## Output Schema

### Standard Output (JSONL)

Each line of the output file is a JSON object:

```json
{
    "question": "What are the latest advances in LLMs?",
    "answer": "",
    "records": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "tool", "name": "google_search", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "usage": {
        "prompt_tokens": 45678,
        "completion_tokens": 12345
    },
    "conversations": [
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "tool", "name": "...", "content": "..."}
        ],
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | The original question |
| `answer` | string | Ground truth answer (if provided in input) |
| `records` | array of messages | Flat list of ALL messages from all turns |
| `usage` | object | Cumulative token usage across all LLM calls |
| `usage.prompt_tokens` | integer | Total prompt tokens consumed |
| `usage.completion_tokens` | integer | Total completion tokens generated |
| `conversations` | array of arrays | Conversation snapshot at each turn checkpoint |

### Full Output (JSONL)

Same as standard output, plus:

| Additional Field | Type | Description |
|------------------|------|-------------|
| `summary_records` | array of messages | All LLM conversations from the Visit tool's extraction step |

**Example `summary_records`**:
```json
[
    [
        {"role": "user", "content": "Please process the following webpage content..."},
        {"role": "assistant", "content": "{\"rational\": \"...\", \"evidence\": \"...\", \"summary\": \"...\"}"}
    ],
    [
        {"role": "user", "content": "Please process the following webpage content..."},
        {"role": "assistant", "content": "{\"rational\": \"...\", \"evidence\": \"...\", \"summary\": \"...\"}"}
    ]
]
```

**Rationale**: The standard output has everything needed for evaluation and analysis. The full output adds the Summary LLM conversations for debugging — if a webpage extraction seems wrong, you can inspect exactly what the Summary LLM saw and produced.

---

## Output File Naming

```
{input_filename}_{prefix}_{timestamp}.jsonl          # Standard
{input_filename}_{prefix}_{timestamp}_full.jsonl     # Full
```

Example:
```
questions.jsonl_iterresearch_20250115_143022.jsonl
questions.jsonl_iterresearch_20250115_143022_full.jsonl
```

**Rationale**: Including the timestamp prevents accidental overwrites when running multiple experiments. Including the input filename makes it easy to trace which questions produced which results.

---

## Internal Data Structures

### LLM Response (EasyDict)

The raw LLM response is wrapped in `EasyDict` for dot-notation access:

```python
response.choices[0].message.content  # The generated text
response.usage.prompt_tokens         # Token counts
response.usage.completion_tokens
```

### Tool Definitions (TOOLS)

```python
TOOLS = [
    {
        "name": "google_search",
        "description": "Perform Google web searches...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "The list of search queries."
                }
            },
            "required": ["query"]
        }
    },
    # ... more tools
]
```

This JSON schema is serialized to a string (`tool_str`) and injected into the prompts so the LLM knows what tools are available and how to call them.

**Rationale**: Using a JSON Schema-like format for tool definitions is a common pattern in LLM tool-use systems. It gives the model a precise specification of expected parameters, reducing errors in tool calls.

### Visit Tool: Extractor JSON Output

The Summary LLM is asked to return JSON with three fields:

```json
{
    "rational": "Why this information is relevant to the goal",
    "evidence": "Direct quotes and data from the webpage",
    "summary": "A concise paragraph summarizing the key findings"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `rational` | string | Reasoning about why the found content relates to the goal (note: the field name in the codebase is `rational`, not `rationale`) |
| `evidence` | string | Actual extracted text from the webpage (can be multiple paragraphs) |
| `summary` | string | A condensed summary with logical flow |

**Rationale**: Separating `evidence` from `summary` ensures the agent gets both raw facts (for accuracy) and a distilled version (for efficiency). The `rational` field forces the Summary LLM to think about relevance before extracting.

---

## Next Steps

- **Want to understand the external APIs?** → [APIs & Integrations](apis-and-integrations.md)
- **Want to see how tools work internally?** → [Tools Reference](tools-reference.md)
- **Want to configure these schemas?** → [Configuration Guide](configuration-guide.md)
