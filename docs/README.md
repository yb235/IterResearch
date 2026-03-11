# IterResearch Documentation

> **IterResearch** is an iterative research agent that performs deep research tasks through multi-turn tool interactions, supporting **2048+ tool calls with only 40k context**. It was accepted at **ICLR 2026**.

This documentation is written so that anyone — even someone unfamiliar with AI agents — can understand how IterResearch works under the hood, why it was built this way, and how all the pieces fit together.

---

## 📚 Table of Contents

| Document | What You'll Learn |
|----------|-------------------|
| [Architecture Overview](architecture-overview.md) | The big picture — every component and how they connect |
| [Agent Architecture](agent-architecture.md) | How the AI agent thinks, plans, acts, and remembers across turns |
| [Data Flow](data-flow.md) | The complete journey of data from input question to final answer |
| [Data Schema](data-schema.md) | Every data structure — inputs, outputs, messages, tool calls |
| [APIs & Integrations](apis-and-integrations.md) | All external services the system talks to and how |
| [Tools Reference](tools-reference.md) | A deep dive into each tool the agent can use |
| [Prompt Engineering](prompt-engineering.md) | The prompt templates that guide the agent's reasoning |
| [Configuration Guide](configuration-guide.md) | Every setting you can tweak and what it controls |
| [Design Rationale](design-rationale.md) | *Why* things are built this way — the thinking behind every major design decision |

---

## 🗂️ Quick File Map

For quick reference, here is the project's source code structure:

```
IterResearch/
├── run.py                    # Main entry point — the agent loop lives here
├── config.py                 # All configuration (env vars with defaults)
├── requirements.txt          # Python dependencies
├── prompts/
│   ├── __init__.py           # Exports prompt templates
│   └── prompts.py            # The 4 prompt templates that guide the agent
└── tools/
    ├── __init__.py           # Exports tool classes
    ├── search.py             # Google Search via SerpAPI
    ├── scholar.py            # Google Scholar via SerpAPI
    ├── visit.py              # Webpage/PDF content extraction + LLM summarization
    ├── python_interpreter.py # Sandboxed Python code execution
    └── pdf_parser.py         # PDF download and text extraction
```

---

## 🚀 Where to Start

- **Want the big picture?** Start with [Architecture Overview](architecture-overview.md).
- **Curious about the AI agent's brain?** Read [Agent Architecture](agent-architecture.md).
- **Want to trace a question from start to finish?** Read [Data Flow](data-flow.md).
- **Setting up the system?** Jump to [Configuration Guide](configuration-guide.md).
- **Want to know *why* things are this way?** Read [Design Rationale](design-rationale.md).
