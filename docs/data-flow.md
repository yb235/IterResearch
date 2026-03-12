# Data Flow

> This document traces the complete journey of data through the IterResearch system — from a raw question in a JSONL file, through every transformation, to the final output file. Every step is explained so you can follow exactly what happens to your data.

---

## The Full Pipeline at a Glance

```
questions.jsonl ──▶ main() ──▶ agentic_loop() ──▶ output.jsonl
                                     │
                              ┌──────┴──────┐
                              │ Turn Loop   │
                              │             │
                              │ Prompt ──▶ LLM ──▶ Parse ──▶ Tool ──▶ Compress
                              │   ▲                                      │
                              │   └──────────────────────────────────────┘
                              └─────────────┘
```

---

## Phase 1: Input Loading

**Where**: `main()` in `run.py`, lines 509–513

```python
all_data = []
with open(args.input_fp, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            all_data.append(json.loads(line))
```

**What happens**:
1. Open the input JSONL file
2. Read it line by line
3. Parse each line as JSON
4. Skip empty lines
5. Collect all questions into a list

**Data at this point**: A list of dictionaries, each with at least a `"question"` key.

```python
# Example
[
    {"question": "What are the latest advances in LLMs?", "answer": ""},
    {"question": "How does attention work in transformers?", "answer": ""}
]
```

**Rationale**: Line-by-line reading with empty-line skipping is robust — it handles files with trailing newlines or accidental blank lines without crashing.

---

## Phase 2: Worker Dispatch

**Where**: `main()` in `run.py`, lines 533–544

```python
with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = [
        executor.submit(agentic_loop, data=data, ...)
        for data in all_data
    ]
```

**What happens**:
1. A thread pool is created with `max_workers` threads
2. Each question is submitted as an independent task
3. Each task runs `agentic_loop()` with one question

**Data at this point**: Each worker receives one dict like `{"question": "...", "answer": "..."}`.

**Rationale**: Using `ThreadPoolExecutor` instead of sequential processing allows multiple questions to be researched simultaneously, which is much faster when you have a capable LLM server.

---

## Phase 3: Initial Prompt Construction

**Where**: `agentic_loop()` in `run.py`, lines 399–404

```python
messages = [{
    "role": "user",
    "content": initial_instruction_prompt
        .replace('{question}', task)
        .replace("{tools}", tool_str)
        .replace('{date_to_use}', date)
}]
```

**What happens**:
1. The `initial_instruction_prompt` template is loaded
2. Three placeholders are filled in:
   - `{question}` → The actual question text
   - `{tools}` → JSON string of all tool definitions
   - `{date_to_use}` → A random date string (for temporal context)
3. This becomes the first message in a standard chat format

**Data at this point**: A list with one message:
```python
[{"role": "user", "content": "You are a professional problem-solving agent... Question: What are the latest advances in LLMs? ..."}]
```

**Rationale**: The random date gives the agent a sense of "when" it is operating, which can help it evaluate the recency of search results. Template-based prompt construction keeps the code clean and the prompts maintainable.

---

## Phase 4: LLM Call

**Where**: `call_llm()` in `run.py`, lines 215–272

```python
payload = {
    "model": "",
    "messages": messages,
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 1.5
}
resp = requests.post(llm_url, headers=headers, json=payload, timeout=300)
response = EasyDict(resp.json())
```

**What happens**:
1. Messages are packed into an OpenAI-compatible payload
2. The payload is POSTed to the LLM endpoint
3. The response is parsed into an `EasyDict` (allows dot-notation access)
4. If `check_format=True`, the response is validated (see below)
5. If validation fails, the call is retried (up to `MAX_FORMAT_RETRIES`)

**Data at this point**: An LLM response object:
```python
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "<report>\n### Status Report...\n</report>\n\n<tool_call>\n{\"name\": \"google_search\", ...}\n</tool_call>"
        }
    }],
    "usage": {"prompt_tokens": 1234, "completion_tokens": 567}
}
```

**Rationale**: `temperature=0.6` provides a balance between deterministic and creative responses. `presence_penalty=1.5` discourages the model from being repetitive. The 300-second timeout accommodates slow models. `EasyDict` makes the response easier to work with in Python.

---

## Phase 5: Response Parsing

**Where**: `agentic_loop()` in `run.py`, lines 436–443

```python
tool_call_str = extract_tags(cur_message['content'], 'tool_call')
tool_call = json.loads(tool_call_str)
```

**What happens**:
1. The `extract_tags()` function uses regex to find content between `<tool_call>` and `</tool_call>` tags
2. The extracted string is parsed as JSON
3. The JSON must contain `"name"` (tool name) and `"arguments"` (tool parameters)

**Data at this point**:
```python
{
    "name": "google_search",
    "arguments": {"query": ["latest advances in LLMs 2025"]}
}
```

If no `<tool_call>` is found but an `<answer>` is present, the agent is finished.

**Rationale**: XML-style tags (`<report>`, `<tool_call>`, `<answer>`) are used instead of JSON-only output because LLMs are more reliable at producing freeform text with tag markers than perfectly structured JSON. The tags also make it easy to extract specific sections with simple regex.

---

## Phase 6: Tool Execution

**Where**: `execute_tool()` in `run.py`, lines 278–311

```python
def execute_tool(tool_name, arguments):
    if tool_name == 'google_search':
        return search_engine.call(arguments), []
    elif tool_name == 'google_scholar':
        # Calls both scholar AND regular search
        scholar_result = scholar_engine.call({"query": query})
        search_result = search_engine.call(arguments)
        return f"{scholar_result}\n\n{search_result}", []
    elif tool_name == 'Visit':
        result, summary_messages = visit_tool.call(arguments)
        return result, summary_messages
    elif tool_name == 'PythonInterpreter':
        return python_executor.call({"code": code}), []
```

**What happens**:
1. The tool name is matched to the appropriate tool object
2. The tool's `call()` method is invoked with the arguments
3. The tool returns a string result (the "observation")
4. For the Visit tool, additional "summary messages" are also returned

**Data at this point**: A string observation from the tool, for example:
```
A Google search for 'latest advances in LLMs 2025' found 10 results:

## Web Results
1. [GPT-5 and Beyond: The State of LLMs in 2025](https://example.com/...)
Date published: Jan 15, 2025
Source: TechReview
Large language models have seen remarkable advances in 2025...
...
```

**Note about `google_scholar`**: When the agent calls `google_scholar`, the system actually runs BOTH a Google Scholar search AND a regular Google search, then combines the results. This gives the agent academic papers plus general web results for a more complete picture.

**Rationale**: Combining scholar + web results gives the agent both academic rigor and practical context. Returning summary messages from the Visit tool preserves the full LLM extraction conversation for debugging.

---

## Phase 7: Context Compression (Markovian State Reconstruction)

**Where**: `format_context()` in `run.py`, lines 317–367

This is the most important data transformation in the system:

```python
def format_context(messages, question, date, is_last=False):
    user, bot, tool = messages[0], messages[1], messages[2]

    report = extract_tags(bot['content'], 'report')     # What the agent learned
    action = extract_tags(bot['content'], 'tool_call')   # What it just did
    tool_response = tool['content']                       # What the tool returned

    observation = observation_prompt.format(tool_response=tool_response)

    # Truncate if too long
    if len(tokens) > MAX_OBSERVATION_TOKENS:
        observation = tokenizer.decode(tokens[:MAX_OBSERVATION_TOKENS])

    # Inject into the instruction prompt template
    return instruction_prompt
        .replace("{question}", question)
        .replace("{report}", report)
        .replace("{action}", action)
        .replace("{observation}", observation)
        .replace("{tools}", tool_str)
        .replace('{date_to_use}', date)
```

**What happens**:
1. From the previous turn's messages, extract:
   - The `<report>` (the agent's accumulated knowledge)
   - The `<tool_call>` (what tool was just called)
   - The tool response (what the tool returned)
2. Truncate the observation if it exceeds `MAX_OBSERVATION_TOKENS` (32,000 tokens)
3. Pack everything into the `instruction_prompt` template
4. Return a single string that becomes the new user message

**Data transformation**:
```
BEFORE (3 messages, potentially very long):
  [user_msg, assistant_msg, tool_msg]

AFTER (1 message, bounded size):
  [new_user_msg containing: question + report + action + observation + tools]
```

Then in the agent loop:
```python
messages = [{"role": "user", "content": cur_turn}]  # Reset to single message!
```

**Rationale**: This is the heart of Markovian State Reconstruction. By discarding the raw conversation and keeping only the compressed report, the context stays bounded. The agent's `<report>` acts as a rolling summary of everything important discovered so far — like a researcher's notebook.

---

## Phase 8: Repeat or Finalize

The loop continues from Phase 4 until one of these happens:
1. The LLM produces an `<answer>` tag instead of a `<tool_call>` → Agent is done
2. The turn counter reaches `max_turn - 2` → Next turn uses `last_instruction_prompt` which forces a final answer
3. The LLM fails to respond → Loop breaks with whatever data was collected

---

## Phase 9: Result Assembly

**Where**: `agentic_loop()` in `run.py`, lines 483–500

```python
result = {
    'question': task,
    'answer': answer,           # Ground truth (if provided)
    'records': records,         # All messages from all turns
    'usage': usage,             # Total token counts
    'conversations': conversations  # Full conversation at each turn
}

full_result = {
    ...same as above...,
    'summary_records': summary_records  # Visit tool's LLM conversations
}
```

**Data at this point**: Two dictionaries — a standard result and a full result with additional debug info.

**Rationale**: Keeping `records` (the flat list of all messages) and `conversations` (the conversation state at each turn checkpoint) serves different purposes. `records` is for analysis and evaluation. `conversations` is for debugging — you can replay any turn's exact context.

---

## Phase 10: Output Saving

**Where**: `main()` in `run.py`, lines 551–555

```python
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")

with open(full_output_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(full_result, ensure_ascii=False) + "\n")
```

**What happens**:
1. The result is JSON-serialized
2. It's appended (not overwritten) to the output file
3. A newline is added (JSONL format)
4. This happens for both the standard and full output files

**Rationale**: Append mode (`'a'`) ensures that results from parallel workers don't overwrite each other. `ensure_ascii=False` preserves non-ASCII characters (important for Chinese text). Writing immediately after each question means no results are lost if the process crashes.

---

## Complete Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                         │
│                                                                    │
│  questions.jsonl                                                   │
│       │                                                            │
│       ▼                                                            │
│  json.loads(line) ──▶ {"question": "...", "answer": "..."}        │
│       │                                                            │
│       ▼                                                            │
│  initial_instruction_prompt.replace(...)                           │
│       │                                                            │
│       ▼                                                            │
│  [{"role":"user", "content":"..."}]  ──▶  LLM API                │
│       │                                         │                  │
│       │                                         ▼                  │
│       │                              Response with <report>        │
│       │                              and <tool_call>               │
│       │                                         │                  │
│       │                                         ▼                  │
│       │                              json.loads(tool_call_json)    │
│       │                                         │                  │
│       │                                         ▼                  │
│       │                              execute_tool(name, args)      │
│       │                                         │                  │
│       │                                         ▼                  │
│       │                              observation (string)          │
│       │                                         │                  │
│       │              ┌──────────────────────────┘                  │
│       │              │                                             │
│       │              ▼                                             │
│       │    format_context()                                       │
│       │    ├── extract <report>                                   │
│       │    ├── extract <tool_call>                                │
│       │    ├── truncate observation                               │
│       │    └── build instruction_prompt                           │
│       │              │                                             │
│       │              ▼                                             │
│       │    messages = [new_user_msg]  (RESET)                     │
│       │              │                                             │
│       │              ▼                                             │
│       │         ┌────────────────┐                                │
│       │         │ Next turn      │◀────────────────┐              │
│       │         │ call LLM again │                  │              │
│       │         └───────┬────────┘                  │              │
│       │                 │                           │              │
│       │                 ▼                           │              │
│       │          Has <answer>?                      │              │
│       │           │         │                       │              │
│       │          YES        NO                      │              │
│       │           │         └── execute_tool ───────┘              │
│       │           ▼                                                │
│       │    Build result dict                                      │
│       │           │                                                │
│       │           ▼                                                │
│       └──▶ json.dumps(result) + "\n"  ──▶  output.jsonl           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

- **Want to know exactly what the data structures look like?** → [Data Schema](data-schema.md)
- **Want to understand the prompt templates?** → [Prompt Engineering](prompt-engineering.md)
- **Want to see how each tool processes data?** → [Tools Reference](tools-reference.md)
