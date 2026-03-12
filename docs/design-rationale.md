# Design Rationale

> This document explains *why* IterResearch is built the way it is. For every major design decision, you'll find the problem it solves, the alternatives that were considered (implicitly or explicitly), and why this approach was chosen. This is the document to read if you want to not just understand the system, but understand the *thinking* behind it.

---

## 1. Markovian State Reconstruction: The Core Innovation

### The Problem

Standard LLM agents accumulate conversation history. Each turn adds the user message, the assistant response, and the tool output. After 20 turns of searching and reading webpages, the conversation might be 100k+ tokens — far exceeding most models' context windows.

### The Naive Solutions and Their Problems

| Approach | Problem |
|----------|---------|
| Use a huge context window (100k+) | Expensive, slow, and attention quality degrades over very long contexts |
| Truncate old messages | Loses critical early findings — the agent "forgets" |
| Summarize the conversation periodically | Summarization is lossy; who decides when to summarize and what to keep? |
| Just limit the number of turns | Severely limits research depth |

### The IterResearch Solution

Instead of accumulating messages, **reset the conversation every turn** to a single message containing:
1. The original question (constant)
2. The previous turn's `<report>` (accumulated knowledge)
3. The previous turn's `<tool_call>` (what was just done)
4. The current observation (what was just learned)

The `<report>` acts as a **rolling summary written by the agent itself**. The agent is explicitly instructed to include ALL important findings, making it responsible for its own memory management.

### Why This Works

- **Context stays bounded**: ~40k tokens per turn regardless of turn count
- **Agent owns its memory**: The LLM decides what's important to remember
- **No external summarizer needed**: The report IS the summary
- **Information loss is minimized**: The prompt aggressively demands comprehensive documentation
- **2048+ turns become feasible**: Each turn starts fresh with bounded context

### The Trade-off

Some information loss is inevitable. Details from early turns may be gradually compressed or dropped as the report evolves. The system mitigates this by requiring the agent to document "ALL important information without any omissions," but no compression is perfectly lossless.

---

## 2. XML Tags Instead of JSON or Function Calling

### The Problem

The agent needs to produce structured output (reports, tool calls, answers) that the system can reliably parse.

### Alternatives Considered

| Approach | Problem |
|----------|---------|
| Pure JSON output | LLMs frequently produce invalid JSON (missing quotes, trailing commas) |
| OpenAI function calling | Not supported by all LLM servers; couples to a specific API feature |
| Freeform text with regex | Fragile; hard to extract specific sections |

### Why XML-Style Tags

```
<report>
Free-form text here — the LLM can write naturally
</report>

<tool_call>
{"name": "google_search", "arguments": {"query": ["..."]}}
</tool_call>
```

XML tags provide the best of both worlds:
- **Tags are simple**: LLMs are very reliable at producing open/close tags
- **Content is flexible**: Inside `<report>`, the LLM writes freeform text. Inside `<tool_call>`, it writes JSON. Different tags can have different content formats.
- **Parsing is trivial**: A simple regex (`<tag>(.*?)</tag>`) extracts any section
- **Server-agnostic**: Works with any LLM endpoint — no special function calling support needed

### The Trade-off

XML tags consume a few extra tokens compared to pure JSON. This is negligible given the ~40k token context.

---

## 3. Two Separate LLMs (Reasoning vs. Summary)

### The Problem

The system needs LLM capabilities for two very different tasks:
1. **Reasoning**: Complex multi-step planning, evidence evaluation, decision-making
2. **Extraction**: Reading a webpage and pulling out relevant facts

### Why Separate

- **Cost optimization**: Reasoning needs a large, powerful model. Extraction works with a smaller model. Smaller models are 10–100x cheaper per token.
- **Throughput**: The Visit tool processes multiple URLs in parallel, creating burst demand. A separate server prevents this from blocking the main reasoning LLM.
- **Specialization**: Different temperature and sampling parameters can be tuned per task.

### The Default: Same Endpoint

Both `LLM_URL` and `SUMMARY_LLM_URL` default to the same endpoint. This means a single-model setup works out of the box, and users can split them later for optimization.

### The Trade-off

Having two LLM servers increases infrastructure complexity. The single-endpoint default mitigates this for simple setups.

---

## 4. Combining Scholar + Web Search

### The Problem

When the agent calls `google_scholar`, should it get only academic results?

### The Decision

Calling `google_scholar` actually triggers BOTH Scholar and regular web search:

```python
elif tool_name == 'google_scholar':
    scholar_result = scholar_engine.call({"query": query})
    search_result = search_engine.call(arguments)
    return f"{scholar_result}\n\n{search_result}", []
```

### Why

- Academic papers provide authoritative, peer-reviewed information
- But not all relevant information is in papers (blog posts, documentation, news articles)
- The agent might miss important context if limited to only Scholar
- The cost of an extra search call is minimal compared to the value of comprehensive results

### The Trade-off

The observation is longer (both search results combined), which uses more context tokens. But `MAX_OBSERVATION_TOKENS` truncation prevents this from being a problem.

---

## 5. Three-Layer Content Fetching Fallback

### The Problem

Webpages are incredibly diverse — some are static HTML, some are JavaScript-heavy SPAs, some require authentication, some block bots, some are PDFs.

### The Solution: ScraperAPI → Jina → Direct HTTP

```
Try ScraperAPI (proxy + JS rendering)
  │ Failed?
  ▼
Try Jina Reader (specialized content extraction)
  │ Failed?
  ▼
Try Direct HTTP GET (basic, but always works for simple pages)
```

### Why This Order

1. **ScraperAPI first**: Most capable — handles JS, anti-bot, PDFs, returns markdown
2. **Jina Reader second**: Excellent content extraction but doesn't handle all sites
3. **Direct HTTP last**: No dependencies, works for the simplest cases

### Why Not Just Use One

No single method handles all webpages. The combination maximizes coverage.

### The Trade-off

Three API calls for a single URL in the worst case. But this only happens on failure — successful fetching stops immediately.

---

## 6. Random Date Generation

### The Problem

The agent needs to know "what today's date is" for evaluating the freshness of search results.

### The Curious Choice

```python
def random_date(start=datetime(2023, 1, 1), end=datetime(2025, 12, 31)):
    delta = end - start
    random_days = random.randint(0, delta.days)
    final_date = start + timedelta(days=random_days)
    return final_date.strftime("%Y-%m-%d")
```

Instead of using the actual current date, the system generates a **random date between 2023 and 2025**.

### Why

This is a **training/evaluation consideration**. The system is designed for:
- **Benchmarking**: When evaluating against a dataset, the "correct" answer might depend on a specific time period. A random date prevents temporal bias.
- **Reproducibility**: Real dates would make results non-reproducible (different dates = different search results evaluations).
- **Robustness**: Training the agent to work with various dates makes it less sensitive to temporal assumptions.

### The Trade-off

The agent might think it's 2023 when it's actually 2026, leading to slight temporal confusion. For production use, you might want to modify this to use the actual date.

---

## 7. Incremental JSONL Output

### The Problem

Processing 100 questions might take hours. What if the process crashes at question 78?

### The Solution

Results are saved immediately after each question completes:

```python
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
```

### Why JSONL (Not JSON Array)

- **JSONL (one object per line)**: Can be appended to safely — just write a new line
- **JSON array**: Would require reading the entire file, parsing it, adding the result, and rewriting — risky and slow

### Why Append Mode

- Thread-safe for parallel workers (each `write()` of a single line is atomic on most systems)
- Crash-resilient — completed results survive process termination
- Simple — no file locking or coordination needed

### The Trade-off

The output file might have results in a non-deterministic order (due to parallel processing). If order matters, sort by question after processing.

---

## 8. Format Validation with Retries

### The Problem

LLMs don't always produce perfectly formatted output. A single missing `<report>` tag would crash the agent loop.

### The Solution

```python
if check_format:
    is_valid, reason = check_report_action(response)
    if not is_valid:
        failed_call += 1
        raise Exception(reason)  # Triggers retry
```

With up to `MAX_FORMAT_RETRIES` (default: 5) attempts.

### Why Not Just Parse Whatever Comes Back

- Missing `<report>` means the agent lost its memory (report is essential for state reconstruction)
- Invalid JSON in `<tool_call>` means we can't execute the requested tool
- Missing both `<answer>` and `<tool_call>` means we don't know what the agent wants to do

A format error is not recoverable — it's better to retry.

### Why 5 Retries

- 1–2 retries: Too few — transient failures are common
- 5 retries: Handles typical error rates (most LLMs succeed within 2–3 attempts)
- 10+ retries: Diminishing returns — if the model can't format correctly in 5 attempts, it's unlikely to improve

### The Trade-off

Each retry costs one additional LLM call (time and tokens). But this is far cheaper than a failed research session.

---

## 9. Forcing First-Turn Tool Use

### The Problem

An LLM might try to answer the question immediately from its training data without doing any research.

### The Solution

The `initial_instruction_prompt` requires:

```
You MUST raise one and only one function call
```

No `<answer>` tag is accepted on the first turn.

### Why

- LLM training data may be outdated — research should use current sources
- Even for questions where the LLM "knows" the answer, verification improves quality
- The agent's value proposition is thorough research, not quick guesses
- First-turn tool calls establish the research trajectory

### The Trade-off

Simple questions that the LLM could answer correctly from memory still require at least one tool call. This adds latency but improves accuracy and reliability.

---

## 10. Lazy Tokenizer Loading

### The Problem

Loading a HuggingFace tokenizer takes several seconds and requires the `transformers` library.

### The Solution

```python
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
    return _tokenizer
```

The tokenizer is loaded only when first needed, and the import is inside the function.

### Why

- **Startup speed**: If the tokenizer is never needed (short observations), no time is wasted loading it
- **Graceful degradation**: If `transformers` isn't installed or the model isn't cached, the system continues without truncation instead of crashing
- **Import flexibility**: The `from transformers import AutoTokenizer` inside the function means the module loads even if `transformers` is not installed

### The Trade-off

First tokenizer use has a one-time latency hit. This is negligible compared to LLM call times.

---

## 11. `presence_penalty: 1.5` — Unusually High

### The Problem

In long research sessions, LLMs tend to repeat themselves — restating the same findings, using the same phrases, making the same searches.

### Why 1.5

Standard chat applications use `presence_penalty` of 0–0.6. IterResearch uses **1.5**, which is quite aggressive.

- **The report requires comprehensive documentation**, which naturally leads to repetitive language
- **High penalty encourages diverse vocabulary** and prevents the agent from getting stuck in loops
- **Research benefits from varied search queries** — high penalty pushes toward different phrasings

### The Trade-off

Very high presence penalty can occasionally make the language awkward or force unusual word choices. The value 1.5 was likely tuned empirically to balance diversity and coherence.

---

## 12. Environment Variables for All Configuration

### The Problem

Different deployment environments (local development, Docker, cloud) need different settings.

### Why Environment Variables Instead of Config Files

- **No file management**: No config files to ship, mount, or protect
- **Docker-friendly**: `docker run -e LLM_URL=... -e SEARCH_API_KEY=...` works naturally
- **Cloud-native**: Kubernetes secrets, AWS Parameter Store, etc., all inject env vars
- **Secure for secrets**: API keys don't end up in git-committed config files

### Why Defaults for Everything

Every env var has a default:
```python
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:10086/v1/chat/completions")
```

This means:
- The system runs without any configuration (with sensible local defaults)
- Users only need to set what they want to change
- Quick testing doesn't require configuration ceremony

### The Trade-off

Environment variables can be harder to document and discover than a config file. This documentation (and `config.py` comments) mitigate that.

---

## Summary: Design Philosophy

IterResearch's design can be summarized by a few key principles:

1. **Bounded context, unbounded research** — Markovian State Reconstruction enables infinite-depth research in finite context
2. **Graceful degradation** — Missing APIs, failed tools, and format errors don't crash the system
3. **Modularity** — Each tool is independent; new tools can be added with minimal changes
4. **Transparency** — The agent must document everything; nothing is hidden in conversation history
5. **Simplicity** — Standard formats (OpenAI API, JSONL, environment variables) over custom solutions

---

## Next Steps

- **Want the big picture?** → [Architecture Overview](architecture-overview.md)
- **Want to understand the agent in detail?** → [Agent Architecture](agent-architecture.md)
- **Ready to configure and run?** → [Configuration Guide](configuration-guide.md)
