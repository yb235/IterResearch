# APIs & Integrations

> This document describes every external service that IterResearch communicates with — what each API does, how it's called, what it returns, and why it was chosen. Think of this as your guide to all the "plumbing" that connects IterResearch to the outside world.

---

## Overview of External Services

```
┌──────────────────────────────────────────────────────────┐
│                    IterResearch                           │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  Reasoning   │   │ Summary LLM  │   │   Search     │  │
│  │  LLM         │   │ (Visit tool) │   │   (SerpAPI)  │  │
│  └──────┬──────┘   └──────┬───────┘   └──────┬───────┘  │
│         │                  │                  │          │
└─────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │
          ▼                  ▼                  ▼
   OpenAI-compat.     OpenAI-compat.     serpapi.com
   LLM server         LLM server         (Google Search
   (DeepSeek,         (Qwen, etc.)       & Scholar)
    Qwen, etc.)
          │                                    │
          │         ┌──────────────┐           │
          │         │ Jina Reader  │           │
          │         │ API          │           │
          │         └──────┬───────┘           │
          │                │                   │
          │         ┌──────────────┐           │
          │         │ ScraperAPI   │           │
          │         │ (optional)   │           │
          │         └──────┬───────┘           │
          │                │                   │
          │         ┌──────────────┐           │
          │         │ Sandbox      │           │
          │         │ (Python exec)│           │
          │         └──────────────┘           │
          │                                    │
```

---

## 1. Reasoning LLM (Main Agent Brain)

**What it does**: Powers the agent's thinking — analyzes questions, plans research strategies, decides which tools to use, synthesizes findings, and writes final answers.

**API Type**: OpenAI Chat Completions (compatible)

**Endpoint**: Configured via `LLM_URL` (default: `http://127.0.0.1:10086/v1/chat/completions`)

### Request Format

```http
POST {LLM_URL}
Content-Type: application/json

{
    "model": "",
    "messages": [
        {"role": "user", "content": "You are a professional problem-solving agent..."}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 1.5
}
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `model` | `""` (empty) | Server uses its default model. Allows the server to decide. |
| `temperature` | `0.6` | Moderate creativity — not too random, not too deterministic |
| `top_p` | `0.95` | Nucleus sampling — considers the top 95% probability mass |
| `presence_penalty` | `1.5` | Strongly discourages repeating the same content |
| Timeout | 300 seconds | Long timeout because research responses can be lengthy |

### Response Format

```json
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "<report>...</report>\n<tool_call>...</tool_call>"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 1234,
        "completion_tokens": 567
    }
}
```

### Retry Behavior

- Up to `MAX_FORMAT_RETRIES` attempts (default: 5) if format validation fails
- 2-second delay between retries
- Returns last response if all retries fail

### Compatible LLM Servers

Any server that implements the OpenAI `/v1/chat/completions` API:
- [SGLang](https://github.com/sgl-project/sglang) (recommended)
- [vLLM](https://github.com/vllm-project/vllm)
- [Ollama](https://ollama.ai/)
- [LiteLLM](https://github.com/BerriAI/litellm)
- OpenAI API directly
- DeepSeek API

**Rationale**: The OpenAI chat format is the de facto standard for LLM APIs. By using this format, IterResearch works with virtually any modern LLM without code changes. The empty `model` field means the server picks its default, which is the common case for self-hosted models.

---

## 2. Summary LLM (Webpage Content Extraction)

**What it does**: Reads raw webpage content and extracts only the information relevant to the agent's current research goal. Think of it as a "reader" that highlights the important parts.

**API Type**: OpenAI Chat Completions (compatible) — same as the Reasoning LLM

**Endpoint**: Configured via `SUMMARY_LLM_URL` (default: `http://127.0.0.1:10086/v1/chat/completions`)

### Request Format

```http
POST {SUMMARY_LLM_URL}
Content-Type: application/json
Authorization: {SUMMARY_LLM_AUTH}   (if configured)

{
    "model": "",
    "messages": [
        {
            "role": "user",
            "content": "Please process the following webpage content and user goal to extract relevant information:\n\n## **Webpage Content**\n{content}\n\n## **User Goal**\n{goal}\n\n## **Task Guidelines**\n..."
        }
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 24000
}
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | `0.7` | Slightly higher than reasoning — extraction is more creative |
| `top_p` | `0.8` | Narrower sampling — we want focused extraction |
| `max_tokens` | `24000` | Large limit — webpages can be long and we want thorough extraction |
| Timeout | 120 seconds | Shorter than reasoning since extraction is simpler |

### Expected Output

The Summary LLM should return JSON (potentially wrapped in markdown code blocks):

```json
{
    "rational": "This page contains information about transformer architectures...",
    "evidence": "The original Transformer paper (Vaswani et al., 2017) introduced...",
    "summary": "The transformer architecture uses self-attention mechanisms to..."
}
```

### Retry Behavior

- Up to 5 attempts
- 2-second delay between retries
- Returns `None` if all retries fail

**Rationale**: Using a separate LLM endpoint for summarization allows you to deploy a cheaper/faster model for this task while reserving the expensive model for reasoning. The higher `max_tokens` accommodates extracting long evidence passages from webpages. A separate `SUMMARY_LLM_AUTH` header supports scenarios where the summary model has different authentication.

---

## 3. SerpAPI (Google Search & Scholar)

**What it does**: Provides Google Search and Google Scholar results via a clean JSON API, bypassing the need to scrape Google directly.

**API Type**: REST API (GET requests)

**Endpoint**: Configured via `SEARCH_API_URL` / `SCHOLAR_API_URL` (default: `https://serpapi.com/search`)

### Google Search Request

```http
GET https://serpapi.com/search?api_key={key}&q={query}&num=10&hl=en&gl=US
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `api_key` | `SEARCH_API_KEY` | Your SerpAPI key |
| `q` | Query text | The search query |
| `num` | `10` | Number of results to return |
| `hl` | `en` or `zh-CN` | Language (auto-detected from query) |
| `gl` | `US` or `CN` | Geographic location (matches language) |

### Google Scholar Request

```http
GET https://serpapi.com/search?api_key={key}&engine=google_scholar&q={query}&num=10&hl=en
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `engine` | `google_scholar` | Specifies Google Scholar results |
| `q` | Query text | Academic search query |
| `num` | `10` | Number of results |
| `hl` | `en` | Language (always English for Scholar) |

### Response Format (Search)

```json
{
    "organic_results": [
        {
            "title": "Result Title",
            "link": "https://example.com/result",
            "snippet": "A brief description of the result...",
            "date": "Jan 15, 2025",
            "source": "TechReview"
        }
    ]
}
```

### Response Format (Scholar)

```json
{
    "organic_results": [
        {
            "title": "Paper Title",
            "link": "https://arxiv.org/abs/...",
            "snippet": "Abstract excerpt...",
            "publication_info": {
                "summary": "A Smith, B Jones - Nature, 2025"
            },
            "inline_links": {
                "cited_by": {
                    "total": 342
                }
            },
            "resources": [
                {
                    "file_format": "PDF",
                    "link": "https://arxiv.org/pdf/..."
                }
            ]
        }
    ]
}
```

### Retry & Error Handling

- Search: 5 retries for HTTP errors, 3 retries for empty results (with 1s/2s delays)
- Scholar: 5 retries for HTTP errors (with 1s delay)

**Rationale**: SerpAPI provides structured JSON results from Google without the need to deal with anti-bot measures, CAPTCHAs, or HTML parsing. The language auto-detection (Chinese vs. English) ensures results are relevant to the query's language. Scholar always uses English because academic literature is predominantly in English.

---

## 4. Jina Reader API

**What it does**: Converts any URL into clean, readable text (markdown). It handles JavaScript rendering, cookie banners, ads, and other web clutter.

**API Type**: REST API (GET request through a proxy URL)

**Endpoint**: `https://r.jina.ai/{url}`

### Request

```http
GET https://r.jina.ai/https://example.com/article
Authorization: Bearer {JINA_API_KEY}
```

### Response

Returns plain text/markdown content of the webpage.

### Fallback

If Jina API key is not configured, the Visit tool falls back to a direct HTTP GET with a browser-like User-Agent header:
```http
GET https://example.com/article
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
```

### Retry Behavior

- 3 attempts with 2-second delays

**Rationale**: Many webpages are heavily JavaScript-dependent and return empty or garbled content when fetched with a simple HTTP GET. Jina Reader handles rendering and returns clean content. The fallback to direct fetch ensures the tool still works (with reduced quality) when Jina is unavailable.

---

## 5. ScraperAPI (Optional)

**What it does**: Provides proxy-based web scraping — routes requests through residential proxies to avoid blocking, and can return pages in markdown format.

**API Type**: REST API (GET request)

**Endpoint**: `https://api.scraperapi.com/`

### Request

```http
GET https://api.scraperapi.com/?api_key={key}&url={target_url}&device_type=desktop&country_code=us&output_format=markdown
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `api_key` | `SCRAPER_API_KEY` | Your ScraperAPI key |
| `url` | Target URL | The webpage to scrape |
| `device_type` | `desktop` | Emulate desktop browser |
| `country_code` | `us` | Route through US proxies |
| `output_format` | `markdown` | Return as markdown text |

### Response

Returns the webpage content. The response `Content-Type` header is checked:
- If it contains `pdf` → content is saved as a file and parsed with PyMuPDF
- If it contains `text` → content is used directly as HTML/markdown
- Otherwise → discarded

### Retry Behavior

- 3 attempts with 1-second delays

**Rationale**: ScraperAPI is used as the first content-fetching method in the Visit tool because it handles JavaScript-rendered pages and anti-bot protections better than a simple HTTP GET. The PDF detection via Content-Type allows the same endpoint to handle both webpages and PDF documents transparently.

---

## 6. Sandbox API (Python Code Execution)

**What it does**: Executes arbitrary Python code in a secure, isolated environment and returns the output.

**API Type**: REST API (POST request)

**Endpoint**: Configured via `SANDBOX_ENDPOINTS` (default: `http://127.0.0.1:8080`)

### Request

```http
POST {endpoint}/run
Content-Type: application/json

{
    "code": "import math\nresult = math.factorial(10)\nprint(result)",
    "language": "python",
    "run_timeout": 50
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | string | Python code to execute |
| `language` | string | Always `"python"` |
| `run_timeout` | integer | Maximum execution time in seconds |

### Response

```json
{
    "stdout": "3628800\n",
    "stderr": "",
    "execution_time": 0.05
}
```

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | string | Standard output (from `print()` calls) |
| `stderr` | string | Error messages |
| `execution_time` | float | Time taken to execute (seconds) |

### Retry & Load Balancing

- Randomly selects from available endpoints (supports multiple)
- Up to 8 attempts across endpoints
- 1-second delay between attempts

**Rationale**: Running arbitrary code from an AI agent in the main process would be a massive security risk. A sandbox endpoint isolates code execution in a container, preventing any damage to the host system. Multiple endpoints provide load balancing and redundancy. Compatible with [sandbox-fusion](https://github.com/sandbox-fusion) or similar services.

---

## API Dependency Summary

| API | Required? | What Breaks Without It |
|-----|-----------|----------------------|
| Reasoning LLM | ✅ Required | Nothing works — no agent reasoning |
| Summary LLM | ✅ Required | Visit tool can't extract information from pages |
| SerpAPI (Search) | ✅ Required* | `google_search` tool returns error messages |
| SerpAPI (Scholar) | ✅ Required* | `google_scholar` tool returns error messages |
| Jina Reader | ❌ Optional | Visit tool falls back to direct HTTP fetch |
| ScraperAPI | ❌ Optional | Visit tool skips proxy-based fetching |
| Sandbox | ❌ Optional | `PythonInterpreter` tool returns error messages |

*The agent can technically run without search APIs, but its usefulness would be severely limited.

---

## Authentication Summary

| Service | Auth Method | Config Variable |
|---------|-------------|-----------------|
| Reasoning LLM | None (or server-specific) | `LLM_URL` |
| Summary LLM | Optional `Authorization` header | `SUMMARY_LLM_AUTH` |
| SerpAPI (Search) | API key in query params | `SEARCH_API_KEY` |
| SerpAPI (Scholar) | API key in query params | `SCHOLAR_API_KEY` |
| Jina Reader | Bearer token | `JINA_API_KEY` |
| ScraperAPI | API key in query params | `SCRAPER_API_KEY` |
| Sandbox | None | `SANDBOX_ENDPOINTS` |

---

## Next Steps

- **Want to understand each tool implementation?** → [Tools Reference](tools-reference.md)
- **Want to configure these APIs?** → [Configuration Guide](configuration-guide.md)
- **Want to know why these services were chosen?** → [Design Rationale](design-rationale.md)
