# Architecture Overview

> This document explains the high-level architecture of IterResearch — what the major components are, how they relate to each other, and where each one lives in the codebase.

---

## What Is IterResearch?

IterResearch is a **research agent** — a program that takes a question, autonomously searches for information, reads webpages, runs code, and keeps iterating until it has enough evidence to write a comprehensive answer.

Think of it like a diligent research assistant who:
1. Reads your question
2. Searches Google and Google Scholar
3. Visits and reads the most promising webpages and papers
4. Takes detailed notes after each step
5. Decides what to look up next based on what it's learned so far
6. Repeats steps 2–5 as many times as needed (up to 2048+ times!)
7. Finally writes a thorough, evidence-based answer

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         IterResearch System                        │
│                                                                     │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────────────┐    │
│  │  Input    │────▶│  Agent Loop  │────▶│  Output (JSONL)      │    │
│  │  (JSONL)  │     │  (run.py)    │     │  results + records   │    │
│  └──────────┘     └──────┬───────┘     └──────────────────────┘    │
│                          │                                          │
│              ┌───────────┼───────────┐                              │
│              │           │           │                              │
│              ▼           ▼           ▼                              │
│      ┌─────────┐  ┌──────────┐  ┌──────────┐                      │
│      │ Prompts │  │ LLM API  │  │  Tools   │                      │
│      │ System  │  │ (OpenAI  │  │          │                      │
│      │         │  │ compat.) │  │ ┌──────┐ │                      │
│      │ initial │  │          │  │ │Search│ │                      │
│      │ instruct│  │ Reasoning│  │ ├──────┤ │                      │
│      │ prompt  │  │ LLM      │  │ │Scholr│ │                      │
│      │         │  │          │  │ ├──────┤ │                      │
│      │ instruct│  │ Summary  │  │ │Visit │ │                      │
│      │ prompt  │  │ LLM      │  │ ├──────┤ │                      │
│      │         │  │          │  │ │Python│ │                      │
│      │ last    │  └──────────┘  │ └──────┘ │                      │
│      │ prompt  │                └──────────┘                      │
│      └─────────┘                                                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Configuration (config.py)                  │  │
│  │  LLM URLs · API Keys · Token Limits · Agent Parameters       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Input Layer

**File**: User-provided JSONL file  
**Purpose**: Supplies the questions the agent will research

The system reads a JSONL file (one JSON object per line), where each line has at least a `"question"` field. You can supply hundreds of questions and the system will process them in parallel.

**Rationale**: JSONL is used instead of plain JSON because it is streaming-friendly — you can append results line-by-line without loading the entire file into memory.

---

### 2. Agent Loop (The Brain)

**File**: `run.py`  
**Purpose**: Orchestrates the entire research process

This is the heart of the system. For each question, it:
1. Creates the initial prompt with the question and available tools
2. Calls the LLM to get a plan and first tool call
3. Executes the requested tool
4. Formats the results into context for the next turn
5. Repeats steps 2–4 until the agent decides it has enough info (or hits the turn limit)
6. Saves the result

**Key innovation**: The agent uses **Markovian State Reconstruction** — instead of keeping the full conversation history (which would grow unboundedly), it compresses each turn into just three pieces: the previous report, the previous action, and the current observation. This allows it to make 2048+ tool calls within a 40k token context window.

**Rationale**: A standard chatbot accumulates messages and eventually exceeds its context window. Markovian State Reconstruction breaks this limit by treating each turn as essentially a fresh prompt with a compressed summary of all prior work.

---

### 3. Prompt System

**File**: `prompts/prompts.py`  
**Purpose**: Templates that shape how the LLM reasons and responds

There are four prompt templates:

| Prompt | When Used | What It Does |
|--------|-----------|-------------|
| `initial_instruction_prompt` | First turn only | Sets up the problem, requires a plan and first tool call |
| `instruction_prompt` | Turns 2 through N-1 | Provides previous context, asks for progress update + next action or final answer |
| `last_instruction_prompt` | Final turn | Forces the agent to give a final answer (no more tool calls) |
| `observation_prompt` | After every tool call | Wraps tool output so the LLM can read it |

**Rationale**: Separate prompts for different stages ensure the agent behaves differently at each phase — exploratory at the start, synthesizing at the end. The last prompt prevents the agent from endlessly searching without ever answering.

---

### 4. LLM Interface

**File**: `run.py` (the `call_llm` function)  
**Purpose**: Communicates with the language model

The system uses an **OpenAI-compatible chat completions API**. This means it works with any LLM that exposes the same endpoint format (DeepSeek, Qwen, local models via SGLang/vLLM, etc.).

There are actually **two LLM endpoints**:
- **Reasoning LLM** (`LLM_URL`): The main brain — does all the thinking, planning, and tool selection
- **Summary LLM** (`SUMMARY_LLM_URL`): A helper — reads webpages and extracts relevant information

**Rationale**: Separating the two LLMs allows you to use a powerful (and expensive) model for reasoning while using a lighter (and cheaper) model for the repetitive task of webpage summarization. They can also be the same model if resources are limited.

---

### 5. Tool Layer

**File**: `tools/` directory  
**Purpose**: The agent's hands — how it interacts with the outside world

Four tools are available:

| Tool | File | What It Does |
|------|------|-------------|
| `google_search` | `tools/search.py` | Searches the web via SerpAPI |
| `google_scholar` | `tools/scholar.py` | Searches academic papers via SerpAPI |
| `Visit` | `tools/visit.py` | Reads webpages/PDFs and extracts key info |
| `PythonInterpreter` | `tools/python_interpreter.py` | Runs Python code in a sandbox |

Each tool follows the same interface pattern: it has a `call(params)` method that accepts a dict and returns a string result.

**Rationale**: A uniform tool interface makes it easy to add new tools without changing the agent loop. The agent just needs to know the tool name and parameter format to use any tool.

---

### 6. Configuration Layer

**File**: `config.py`  
**Purpose**: Centralized settings management

Every setting is loaded from environment variables with sensible defaults. This includes API keys, LLM URLs, token limits, and agent behavior parameters.

**Rationale**: Environment variables make the system deployable anywhere (local, cloud, Docker) without code changes. Defaults ensure it works out of the box for common setups.

---

### 7. Output Layer

**File**: `run.py` (within `main()`)  
**Purpose**: Saves research results

Two JSONL files are produced:
- **Standard output**: Question, answer, conversation records, and token usage
- **Full output**: Everything above, plus the summary LLM conversations from the Visit tool

Results are saved incrementally (one line appended after each question completes) so that progress survives crashes.

**Rationale**: Incremental JSONL saving is crash-resilient — if the process is killed after processing 50 of 100 questions, you still have those 50 results. The "full" output includes summary records for debugging and analysis.

---

## How Components Connect

```
User provides questions.jsonl
        │
        ▼
    main() reads the file
        │
        ▼
    ThreadPoolExecutor spawns workers
        │
        ├──▶ agentic_loop(question_1) ──▶ result_1
        ├──▶ agentic_loop(question_2) ──▶ result_2
        └──▶ agentic_loop(question_3) ──▶ result_3
                    │
                    ▼
            For each turn:
            1. call_llm() ──▶ LLM API ──▶ response
            2. Extract <tool_call> from response
            3. execute_tool() ──▶ Tool module ──▶ observation
            4. format_context() ──▶ Compress into next prompt
            5. Repeat or finalize

        Results appended to output .jsonl files
```

---

## Next Steps

- **Want to understand the agent loop in detail?** → [Agent Architecture](agent-architecture.md)
- **Want to trace data through the system?** → [Data Flow](data-flow.md)
- **Want to understand each tool?** → [Tools Reference](tools-reference.md)
