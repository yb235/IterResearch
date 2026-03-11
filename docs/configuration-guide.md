# Configuration Guide

> This document covers every configurable setting in IterResearch — what it controls, its default value, and how to change it. All settings are managed through environment variables.

---

## Quick Setup

The fastest way to configure IterResearch is to export environment variables before running:

```bash
# Required: LLM endpoints
export LLM_URL="http://127.0.0.1:10086/v1/chat/completions"
export SUMMARY_LLM_URL="http://127.0.0.1:10087/v1/chat/completions"

# Required: Search API
export SEARCH_API_KEY="your-serpapi-key"
export SCHOLAR_API_KEY="your-serpapi-key"

# Recommended: Webpage reading
export JINA_API_KEY="your-jina-api-key"

# Optional extras
export SCRAPER_API_KEY="your-scraper-api-key"
export SANDBOX_ENDPOINTS="http://127.0.0.1:8080"

# Run the agent
python run.py --input_fp questions.jsonl --output_path ./output --max_turn 256
```

---

## Configuration Categories

### 1. LLM Configuration

These control the AI language models that power the agent.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://127.0.0.1:10086/v1/chat/completions` | URL of the main reasoning LLM. This is the "brain" that does all the thinking. |
| `SUMMARY_LLM_URL` | `http://127.0.0.1:10086/v1/chat/completions` | URL of the summary LLM used to extract information from webpages. Can be the same as `LLM_URL` or a separate, lighter model. |
| `SUMMARY_LLM_AUTH` | `""` (empty) | Optional `Authorization` header for the summary LLM. Use this if the summary LLM requires authentication (e.g., `Bearer your-token`). |

**Rationale for separate LLMs**: The reasoning LLM needs to be powerful (it makes complex decisions). The summary LLM just reads and extracts text — a lighter model works fine and costs less. Having separate URLs lets you optimize costs.

**Rationale for defaults pointing to localhost**: The system assumes you're running models locally (common in research). For cloud APIs, override with the provider's URL.

---

### 2. Search API Configuration

These power the Google Search and Google Scholar tools.

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_API_KEY` | `""` (empty) | API key for the web search provider (SerpAPI). |
| `SEARCH_API_URL` | `https://serpapi.com/search` | URL of the search API endpoint. Change this to use a different search provider. |
| `SCHOLAR_API_KEY` | `""` (empty) | API key for the Google Scholar provider. Often the same as `SEARCH_API_KEY` when using SerpAPI. |
| `SCHOLAR_API_URL` | `https://serpapi.com/search` | URL of the Scholar API endpoint. |

**Rationale**: Separating search and scholar API keys allows using different providers or accounts for each. However, SerpAPI provides both services under one key, so they're often identical.

**What happens if not set**: The search/scholar tools will return an error message like `"[Search] Error: SEARCH_API_KEY is not configured."` The agent will still run but cannot search.

---

### 3. Webpage Reading Configuration

These control how the Visit tool fetches and reads web content.

| Variable | Default | Description |
|----------|---------|-------------|
| `JINA_API_KEY` | `""` (empty) | API key for Jina Reader, which converts webpages to clean markdown. |
| `SCRAPER_API_KEY` | `""` (empty) | API key for ScraperAPI, which provides proxy-based web scraping. |

**How they work together in the Visit tool**:
1. **ScraperAPI first** (if key set): Tries proxy-based fetching
2. **Jina Reader next** (if key set): Falls back to Jina for content extraction
3. **Direct HTTP last** (always): Tries a simple GET request as last resort

**What happens if neither is set**: The Visit tool attempts direct HTTP GET requests. This works for simple, public webpages but fails for JavaScript-heavy sites or those with anti-bot measures.

**Rationale**: Making both optional means the system still works in minimal configurations — important for quick testing or environments with limited API access.

---

### 4. Python Sandbox Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_ENDPOINTS` | `http://127.0.0.1:8080` | Comma-separated list of sandbox endpoints for executing Python code. Multiple endpoints enable load balancing. |

**Example with multiple endpoints**:
```bash
export SANDBOX_ENDPOINTS="http://sandbox1:8080,http://sandbox2:8080,http://sandbox3:8080"
```

The system randomly selects an endpoint for each execution and retries on others if one fails.

**What happens if not available**: The PythonInterpreter tool returns `"[Python Interpreter Error]: All attempts failed."` when the sandbox is unreachable. The agent will still run but can't execute code.

**Rationale**: Code execution is sandboxed for security — the agent might generate dangerous code. Running it in an isolated container prevents damage. Multiple endpoints provide resilience.

---

### 5. Tokenizer Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKENIZER_PATH` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model path for the tokenizer used in token counting and truncation. |

**Why it matters**: The tokenizer is used to:
- Count tokens in tool observations (to decide if truncation is needed)
- Truncate content to fit within `MAX_OBSERVATION_TOKENS` or `MAX_WEBPAGE_TOKENS`

**What happens if unavailable**: A warning is printed, and token-based truncation is skipped. Content might exceed token limits, potentially causing LLM errors.

**Rationale**: Token counting must match the model being used. The default is Qwen2.5-7B because that's a common model for this system. If you use a different model family, set this to match for accurate counting.

---

### 6. Agent Behavior Configuration

These control how the agent behaves during research.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TURN` | `25` | Maximum number of turns per question. Higher = more thorough research but slower. |
| `MAX_FORMAT_RETRIES` | `5` | How many times to retry when the LLM produces invalid output format. |
| `MAX_WORKERS` | `1` | Number of questions to process in parallel. |
| `MAX_OBSERVATION_TOKENS` | `32000` | Maximum token count for tool observations injected into the agent's context. Observations exceeding this are truncated. |
| `MAX_WEBPAGE_TOKENS` | `48000` | Maximum token count for webpage content before it's sent to the Summary LLM. |

#### MAX_TURN

- **Low values (5–10)**: Quick but shallow research. Good for simple factual questions.
- **Medium values (25–50)**: Default range. Good balance for most research tasks.
- **High values (100–2048)**: Deep, thorough research. The agent's Markovian State Reconstruction makes this feasible without context overflow.

**Rationale**: The default of 25 balances thoroughness with speed. The system is designed to handle up to 2048+ turns, but most questions don't need that many.

#### MAX_FORMAT_RETRIES

- **Low (1–2)**: Fast but fragile. One bad format = move on.
- **Default (5)**: Robust. Handles occasional LLM formatting mistakes.
- **High (10+)**: Very tolerant. Useful for weaker models that struggle with formatting.

**Rationale**: Format validation retries are cheap (just another LLM call) and prevent the entire agent loop from failing due to a single formatting error.

#### MAX_WORKERS

- **1**: Sequential processing. Safe and predictable. Best when LLM server has limited capacity.
- **2–4**: Moderate parallelism. Good when you have a capable LLM server.
- **8+**: High throughput. Requires a powerful LLM server with sufficient capacity.

**Rationale**: Default of 1 is safest — won't overwhelm a single LLM server. Increase based on your infrastructure capacity.

#### MAX_OBSERVATION_TOKENS vs MAX_WEBPAGE_TOKENS

Why are these different?
- **Observations** (32k) go into the main agent's context alongside the prompt, report, and action. They need to be smaller to leave room.
- **Webpage content** (48k) goes to the Summary LLM in isolation (just the content + extraction prompt). It can be larger because there's less competing context.

---

## Command-Line Arguments

These can also override some settings at runtime:

| Argument | Type | Default | Overrides |
|----------|------|---------|-----------|
| `--input_fp` | str | (required) | — |
| `--output_path` | str | `./output` | — |
| `--prefix` | str | `iterresearch` | — |
| `--max_workers` | int | `MAX_WORKERS` from config | `MAX_WORKERS` env var |
| `--max_format_retries` | int | `MAX_FORMAT_RETRIES` from config | `MAX_FORMAT_RETRIES` env var |
| `--max_turn` | int | `MAX_TURN` from config | `MAX_TURN` env var |
| `--llm_url` | str | `LLM_URL` from config | `LLM_URL` env var |

**Precedence**: Command-line argument > Environment variable > Default value

---

## Recommended Configurations

### Minimal Setup (Testing)

```bash
export LLM_URL="http://127.0.0.1:10086/v1/chat/completions"
export SUMMARY_LLM_URL="http://127.0.0.1:10086/v1/chat/completions"  # Same model
export SEARCH_API_KEY="your-serpapi-key"
export SCHOLAR_API_KEY="your-serpapi-key"

python run.py --input_fp questions.jsonl --max_turn 10
```

### Full Setup (Production)

```bash
# Separate reasoning and summary models
export LLM_URL="http://reasoning-server:10086/v1/chat/completions"
export SUMMARY_LLM_URL="http://summary-server:10087/v1/chat/completions"

# All search capabilities
export SEARCH_API_KEY="your-serpapi-key"
export SCHOLAR_API_KEY="your-serpapi-key"
export JINA_API_KEY="your-jina-key"
export SCRAPER_API_KEY="your-scraper-key"

# Code execution
export SANDBOX_ENDPOINTS="http://sandbox1:8080,http://sandbox2:8080"

# Tuned parameters
export MAX_TURN=256
export MAX_WORKERS=4

python run.py --input_fp questions.jsonl --output_path ./results
```

### Deploying the Recommended Summary Model

```bash
pip install sglang[all]

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --port 10087 \
    --tp 8 \
    --trust-remote-code
```

For lower resources:
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --port 10087 \
    --tp 2 \
    --trust-remote-code
```

---

## Next Steps

- **Want to understand the system architecture?** → [Architecture Overview](architecture-overview.md)
- **Want to know why settings have these defaults?** → [Design Rationale](design-rationale.md)
- **Want to understand the APIs these settings configure?** → [APIs & Integrations](apis-and-integrations.md)
