# Tools Reference

> This document is a deep dive into each tool the IterResearch agent can use. For every tool, you'll learn what it does, how it works internally, its error handling, and design decisions behind it.

---

## Tool Architecture

All tools share a common pattern:

```python
class ToolName:
    name = "tool_name"                    # Identifier used in tool calls
    description = "What the tool does"    # Shown to the LLM
    parameters = { ... }                  # JSON Schema for parameters

    def __init__(self, ...):              # Setup (API keys, config)
        ...

    def call(self, params) -> str:        # Main entry point
        # 1. Parse params (handle str or dict)
        # 2. Validate inputs
        # 3. Execute the operation
        # 4. Return formatted string result
```

**Rationale**: A uniform interface makes the agent loop simple — it just needs to call `tool.call(arguments)` regardless of which tool is being used. String inputs/outputs keep things compatible with any LLM.

---

## Tool 1: Google Search (`tools/search.py`)

### Purpose

Searches the web via Google to find current information, news, documentation, and other web content.

### How It Works

```
Agent sends: {"query": ["latest LLM advances 2025", "GPT-5 features"]}
                │
                ▼
        For each query:
        ┌─────────────────────────────────────┐
        │ 1. Detect language (Chinese/English) │
        │ 2. Set locale params (hl, gl)        │
        │ 3. GET serpapi.com/search?...         │
        │ 4. Extract organic_results            │
        │ 5. Format each result as markdown     │
        │ 6. Handle retries for failures        │
        └─────────────────────────────────────┘
                │
                ▼
        Return formatted results (separated by "=======")
```

### Language Detection

The tool checks each query for Chinese characters:

```python
def _contains_chinese(self, text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)
```

- **Chinese detected**: Sets `hl=zh-CN`, `gl=CN` for Chinese locale results
- **No Chinese**: Sets `hl=en`, `gl=US` for English locale results

**Rationale**: Google returns different results based on locale. If a user asks a question in Chinese, they probably want Chinese-language results. This auto-detection ensures relevant results without requiring the user to specify their language.

### Result Formatting

Each result is formatted as:

```
1. [Title](link)
Date published: Jan 15, 2025
Source: TechReview
Snippet text describing the result...
```

The markdown format with links is designed to be readable by the LLM while preserving all useful metadata.

### Retry Logic

```
Attempt 1 → API call → Success? → Return results
                     → HTTP error? → Wait 2s, retry
                     → Empty results? → Wait 1s, retry (max 3 empty retries)
                     → Max retries (5)? → Return error message
```

### Statistics

The tool tracks `total_requests` and `successful_requests` for monitoring.

**Rationale**: Tracking success rates helps diagnose API issues. If the rate drops, it might indicate an API key quota problem or service outage.

---

## Tool 2: Google Scholar (`tools/scholar.py`)

### Purpose

Searches academic literature — papers, citations, and publications — via Google Scholar.

### How It Works

Similar to Google Search but with academic-specific features:

```
Agent sends: {"query": ["transformer attention mechanism", "BERT pretraining"]}
                │
                ▼
        For each query (parallel with ThreadPoolExecutor, 3 workers):
        ┌──────────────────────────────────────┐
        │ 1. GET serpapi.com/search?             │
        │    engine=google_scholar&q=...         │
        │ 2. Extract organic_results             │
        │ 3. For each result, extract:           │
        │    - Title                              │
        │    - Link (prefer PDF link if avail.)   │
        │    - Publication info (journal, authors) │
        │    - Citation count                      │
        │    - Snippet                             │
        │ 4. Format as markdown                   │
        └──────────────────────────────────────┘
                │
                ▼
        Return formatted results
```

### Key Difference: PDF Link Priority

Scholar results often include PDF links in the `resources` field. The tool prefers these:

```python
for res in resources:
    if res.get("file_format") == "PDF":
        pdf_link = res.get("link", "")
        break

link_info = f"pdfUrl: {pdf_link}" if pdf_link else f"link: {link}"
```

**Rationale**: Researchers usually want the PDF of a paper, not just the abstract page. By surfacing PDF links, the agent can later use the Visit tool to read the full paper.

### Parallel Execution

Unlike the Search tool (which processes queries sequentially), Scholar uses a thread pool:

```python
with ThreadPoolExecutor(max_workers=3) as executor:
    responses = list(executor.map(self._search_single, query))
```

**Rationale**: Scholar API calls tend to be slower than regular search. Parallelizing multiple queries reduces total wall-clock time significantly.

### Combined with Regular Search

In `execute_tool()`, calling `google_scholar` actually triggers **both** Scholar AND regular search:

```python
elif tool_name == 'google_scholar':
    scholar_result = scholar_engine.call({"query": query})
    search_result = search_engine.call(arguments)
    return f"{scholar_result}\n\n{search_result}", []
```

**Rationale**: Academic papers don't always show up in regular search, and not all relevant content is in academic papers. Combining both gives the agent the best of both worlds.

---

## Tool 3: Visit (`tools/visit.py`)

### Purpose

The most complex tool. It visits webpages or PDFs, extracts their content, and uses a secondary LLM to pull out only the information relevant to the agent's research goal.

### How It Works (Complete Pipeline)

```
Agent sends: {"url": ["https://arxiv.org/abs/1706.03762"],
              "goal": "How does self-attention work?",
              "parse_type": "html"}
                │
                ▼
        For each URL (parallel with ThreadPoolExecutor, 5 workers):
        ┌─────────────────────────────────────────────┐
        │                                             │
        │  STEP 1: Fetch Content                      │
        │  ┌────────────────────────────────────────┐ │
        │  │ Try ScraperAPI first:                  │ │
        │  │   GET api.scraperapi.com/?url=...      │ │
        │  │   ├─ PDF Content-Type? Save file,      │ │
        │  │   │  parse with PyMuPDF                │ │
        │  │   ├─ Text Content-Type? Use directly   │ │
        │  │   └─ Other? Fall through               │ │
        │  │                                        │ │
        │  │ Then try Jina Reader:                  │ │
        │  │   GET r.jina.ai/{url}                  │ │
        │  │   ├─ Success? Use the text             │ │
        │  │   └─ Fail? Try direct HTTP GET         │ │
        │  └────────────────────────────────────────┘ │
        │                │                            │
        │                ▼                            │
        │  STEP 2: Truncate if Needed                 │
        │  ┌────────────────────────────────────────┐ │
        │  │ Token count = tokenizer.encode(content)│ │
        │  │ If > MAX_WEBPAGE_TOKENS (48,000):      │ │
        │  │   content = tokenizer.decode(           │ │
        │  │     tokens[:MAX_WEBPAGE_TOKENS])        │ │
        │  └────────────────────────────────────────┘ │
        │                │                            │
        │                ▼                            │
        │  STEP 3: LLM Extraction                     │
        │  ┌────────────────────────────────────────┐ │
        │  │ Send to Summary LLM:                   │ │
        │  │   EXTRACTOR_PROMPT with:               │ │
        │  │   - Full webpage content               │ │
        │  │   - Agent's research goal              │ │
        │  │                                        │ │
        │  │ LLM returns JSON:                      │ │
        │  │   {"rational": "...",                   │ │
        │  │    "evidence": "...",                    │ │
        │  │    "summary": "..."}                    │ │
        │  └────────────────────────────────────────┘ │
        │                │                            │
        │                ▼                            │
        │  STEP 4: Format Output                      │
        │  ┌────────────────────────────────────────┐ │
        │  │ "The useful information in '{url}'     │ │
        │  │  for user goal '{goal}':\n\n            │ │
        │  │  Evidence in page:\n{evidence}\n\n      │ │
        │  │  Summary:\n{summary}\n"                 │ │
        │  └────────────────────────────────────────┘ │
        │                                             │
        └─────────────────────────────────────────────┘
                │
                ▼
        Return combined results + summary messages
```

### Content Fetching Strategy (Three-Layer Fallback)

```
1. ScraperAPI (if SCRAPER_API_KEY is set)
   ├─ Handles JS-rendered pages
   ├─ Bypasses anti-bot protections
   └─ Detects PDFs automatically

2. Jina Reader API (if JINA_API_KEY is set)
   ├─ Converts pages to clean markdown
   └─ Handles most modern websites

3. Direct HTTP GET (always available)
   ├─ Simple but limited
   └─ Only works for static, public pages
```

**Rationale**: A three-layer fallback maximizes the chance of successfully reading any webpage. ScraperAPI is tried first because it handles the most edge cases (JavaScript, CAPTCHAs, PDFs). Jina is the second choice for clean content extraction. Direct HTTP is the last resort.

### PDF Handling

When a PDF is detected (via Content-Type header or URL ending in `.pdf`):

1. The file is downloaded to `/tmp/pdf/`
2. PyMuPDF (`fitz`) opens the file
3. Text is extracted page by page
4. Each page is prefixed with `## Page N`

```python
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    text_parts.append(f"## Page {page_num + 1}\n\n{text}")
```

**Rationale**: Page-by-page extraction with headers preserves document structure. If the agent needs information from page 15, it can reference that specifically.

### JSON Parsing (Lenient)

The Summary LLM's output is parsed with multiple fallbacks:

```python
def _parse_json_output(raw):
    # Try 1: Extract from ```json ... ``` markdown blocks
    # Try 2: json5.loads() (lenient — allows trailing commas, etc.)
    # Try 3: json.loads() (strict standard JSON)
    # All fail? Return None
```

**Rationale**: LLMs often produce almost-valid JSON — with trailing commas, single quotes, or wrapped in markdown code blocks. Using `json5` as a fallback handles these common quirks gracefully.

### Return Value

The Visit tool is unique in that it returns **two things**:

```python
return (observation_string, summary_messages)
```

1. `observation_string`: The formatted extraction result (goes to the agent)
2. `summary_messages`: The full conversation with the Summary LLM (saved for debugging)

---

## Tool 4: PythonInterpreter (`tools/python_interpreter.py`)

### Purpose

Executes Python code in a sandboxed environment for calculations, data processing, and any computation the agent needs.

### How It Works

```
Agent sends: {"code": "import math\nprint(math.factorial(20))"}
                │
                ▼
        ┌─────────────────────────────────────┐
        │ 1. Extract code from markdown/tags  │
        │ 2. Validate code is not empty       │
        │ 3. Pick random sandbox endpoint     │
        │ 4. POST {endpoint}/run with code    │
        │ 5. Collect stdout + stderr          │
        │ 6. Check for timeout                │
        │ 7. Return formatted output          │
        └─────────────────────────────────────┘
```

### Code Extraction

The tool handles code wrapped in various formats:

```python
# From markdown triple backticks:
```python
print("hello")
```  → extracts "print(\"hello\")"

# From <code> tags:
<code>print("hello")</code> → extracts "print(\"hello\")"

# Raw code:
print("hello") → used as-is
```

**Rationale**: LLMs frequently wrap code in markdown or HTML tags. The extraction logic handles these common patterns so the agent doesn't need to produce perfectly clean code strings.

### Load Balancing

```python
endpoint = random.choice(self.endpoints)  # Random selection
```

With up to 8 retry attempts across different endpoints. This provides basic load balancing without complex infrastructure.

**Rationale**: Random selection with retries is simple and effective for small-scale deployments. If one endpoint is down, the retries will eventually hit a working one.

### Timeout Detection

```python
execution_time = result.get("execution_time", 0)
if execution_time >= timeout - 1:
    output_parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")
```

The tool checks both the HTTP-level timeout (via the `requests` library) and the sandbox-level timeout (via `execution_time` in the response).

**Rationale**: Double timeout checking catches both network-level hangs and runaway code. The `-1` buffer accounts for timing imprecision.

---

## Tool Comparison Summary

| Feature | Search | Scholar | Visit | Python |
|---------|--------|---------|-------|--------|
| Batch support | ✅ Array of queries | ✅ Array of queries | ✅ Array of URLs | ❌ Single code block |
| Parallel execution | ❌ Sequential | ✅ 3 workers | ✅ 5 workers | ❌ Sequential |
| External API | SerpAPI | SerpAPI | ScraperAPI + Jina + Summary LLM | Sandbox |
| Retry attempts | 5 | 5 | 3 | 8 |
| Returns summary msgs | ❌ | ❌ | ✅ | ❌ |
| Language detection | ✅ (Chinese/English) | ❌ (English only) | ❌ | ❌ |

---

## Adding a New Tool

To add a new tool to IterResearch:

1. **Create the tool file** in `tools/` with the standard interface:
   ```python
   class NewTool:
       name = "new_tool"
       description = "..."
       parameters = { ... }
       def call(self, params): ...
   ```

2. **Export it** from `tools/__init__.py`:
   ```python
   from .new_tool import NewTool
   ```

3. **Add the tool definition** to the `TOOLS` list in `run.py`

4. **Add the execution handler** in `execute_tool()`:
   ```python
   elif tool_name == 'new_tool':
       return new_tool_instance.call(arguments), []
   ```

5. **Initialize the tool** at module level in `run.py`

**Rationale**: The modular design means adding a tool requires changes in only two files (`tools/new_tool.py` and `run.py`), keeping the system extensible.

---

## Next Steps

- **Want to understand the prompts that guide tool use?** → [Prompt Engineering](prompt-engineering.md)
- **Want to configure tool settings?** → [Configuration Guide](configuration-guide.md)
- **Want to know why these tools were chosen?** → [Design Rationale](design-rationale.md)
