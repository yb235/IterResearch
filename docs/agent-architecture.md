# Agent Architecture

> This document takes you deep inside the IterResearch agent — how it thinks, decides, remembers, and iterates. This is where the core innovation lives.

---

## The Core Idea

Most chatbots work like this: you send a message, the AI responds, the full conversation history grows, and eventually the history exceeds the model's context window and things break.

IterResearch solves this with **Markovian State Reconstruction** — a technique that lets the agent run for **2048+ turns** within only **40k tokens** of context. The key insight is:

> The agent doesn't need to remember every word of every past conversation. It only needs to know: (1) what it has learned so far, (2) what it just did, and (3) what it just observed.

This is analogous to the Markov property in mathematics — the future state depends only on the present state, not on the entire history of states.

---

## Agent Loop Step-by-Step

Here is exactly what happens when the agent researches a single question:

### Turn 0: The First Move

```
┌──────────────────────────────────────────────┐
│ 1. Build initial_instruction_prompt with:    │
│    - The question                            │
│    - Available tools (JSON schema)           │
│    - A random date (for the agent's context) │
│                                              │
│ 2. Send to LLM                               │
│                                              │
│ 3. LLM responds with:                        │
│    <report>                                  │
│      Problem analysis and plan               │
│    </report>                                 │
│    <tool_call>                               │
│      {"name": "google_search",               │
│       "arguments": {"query": ["..."]}}       │
│    </tool_call>                              │
│                                              │
│ 4. Execute the tool → get observation        │
└──────────────────────────────────────────────┘
```

**What the LLM must produce on the first turn**:
- A `<report>` tag containing its analysis of the problem
- A `<tool_call>` tag with exactly one tool invocation (JSON)
- The first turn always requires a tool call — no answer is accepted yet

**Rationale**: Forcing the first turn to always make a tool call ensures the agent starts by gathering information rather than guessing an answer. The `<report>` tag forces it to think before acting.

---

### Turns 1 through N-2: The Iterative Research Phase

This is where the magic happens. After the first turn, every subsequent turn follows this pattern:

```
┌─────────────────────────────────────────────────────┐
│ 1. Take the previous turn's 3 messages:             │
│    [user_msg, assistant_msg, tool_msg]              │
│                                                     │
│ 2. Extract from assistant_msg:                      │
│    - <report> content (what the agent learned)      │
│    - <tool_call> content (what it just did)         │
│                                                     │
│ 3. Build instruction_prompt with:                   │
│    - Original question                              │
│    - Previous report (compressed knowledge)         │
│    - Previous action (what tool was called)         │
│    - Observation (what the tool returned)           │
│    - Available tools                                │
│                                                     │
│ 4. RESET messages to just this one new user message │
│                                                     │
│ 5. Send to LLM                                      │
│                                                     │
│ 6. LLM responds with EITHER:                        │
│    Option A: <report> + <tool_call> (keep going)    │
│    Option B: <report> + <answer> (done!)            │
│                                                     │
│ 7. If Option A: execute tool, go to step 1          │
│    If Option B: save result, stop                   │
└─────────────────────────────────────────────────────┘
```

**This is Markovian State Reconstruction in action.** Notice step 4 — the message history is reset to a single message every turn. The agent's accumulated knowledge is compressed into the `<report>` field, which is injected into the next prompt. The LLM never sees the raw conversation from two turns ago.

**Rationale**: Without this compression, the context window would grow linearly with each turn. With 2048 turns of search results and webpage content, you'd need millions of tokens of context. By compressing to report + action + observation, context stays bounded at approximately 40k tokens regardless of how many turns have passed.

---

### Turn N-1: The Final Turn

When the turn counter reaches `max_turn - 1`, the system switches to the `last_instruction_prompt`:

```
┌──────────────────────────────────────────────────┐
│ 1. Build last_instruction_prompt with:           │
│    - Same inputs as intermediate turns           │
│    - But NO tool list (no more tools available)  │
│                                                  │
│ 2. LLM MUST respond with:                        │
│    <report> + <answer>                           │
│    (tool calls are NOT allowed)                  │
│                                                  │
│ 3. Save the final answer                         │
└──────────────────────────────────────────────────┘
```

**Rationale**: The last turn forces a final answer. Without this, an agent could get stuck in an infinite loop of "I need one more search." The hard cutoff ensures every question gets an answer, even if imperfect.

---

## Format Validation

Every LLM response is validated before being accepted:

```python
def check_report_action(response):
    # 1. Must contain <report> tags
    # 2. Must contain either <tool_call> or <answer>
    # 3. If <tool_call>, the JSON must be valid with "name" and "arguments"
    # Returns: (is_valid, error_message)
```

If validation fails, the system retries up to `MAX_FORMAT_RETRIES` times (default: 5).

**Rationale**: LLMs are not perfectly reliable at producing structured output. Format validation with retries ensures the agent loop doesn't break due to a one-off formatting error. The retry count is limited to prevent infinite loops on persistent failures.

---

## The Agent's Decision Framework

The instruction prompt forces the agent to follow a structured decision process:

```
Has the agent gathered enough information?
    │
    ├── YES → Output <answer> with comprehensive response
    │         (must synthesize ALL collected evidence)
    │
    └── NO  → Output <tool_call> with next research action
              (must explain WHY more info is needed in <report>)
```

Within the `<report>`, the agent must clearly distinguish between:

1. **Confirmed facts** — Evidence directly observed from tool results
2. **Highly credible inferences** — Conclusions strongly supported by evidence
3. **Hypotheses to verify** — Ideas that still need checking

**Rationale**: This classification prevents the agent from treating guesses as facts. By forcing explicit uncertainty tracking, the agent is more likely to verify important claims before finalizing its answer.

---

## Parallel Question Processing

The system can process multiple questions simultaneously:

```
main()
  │
  ▼
ThreadPoolExecutor(max_workers=N)
  │
  ├──▶ Worker 1: agentic_loop(question_1)
  ├──▶ Worker 2: agentic_loop(question_2)
  ├──▶ Worker 3: agentic_loop(question_3)
  ...
  └──▶ Worker N: agentic_loop(question_N)
```

Each worker runs its own independent agent loop. Results are saved incrementally as each worker completes.

**Rationale**: Research questions are independent — one doesn't affect another. Parallel processing dramatically reduces total time when you have multiple questions and sufficient LLM/API capacity.

---

## Token Management

The agent has two token truncation points to prevent context overflow:

### 1. Observation Truncation (in `format_context()`)

```python
if len(tokens) > MAX_OBSERVATION_TOKENS:  # Default: 32,000
    observation = tokenizer.decode(tokens[:MAX_OBSERVATION_TOKENS])
```

When a tool returns a very long result (e.g., a search with many results), the observation is truncated before being injected into the next prompt.

### 2. Webpage Truncation (in `Visit._visit_single()`)

```python
if len(tokens) > MAX_WEBPAGE_TOKENS:  # Default: 48,000
    content = tokenizer.decode(tokens[:MAX_WEBPAGE_TOKENS])
```

When a webpage is very long, the raw content is truncated before being sent to the Summary LLM.

Both use the Hugging Face `AutoTokenizer` (default: Qwen2.5-7B-Instruct) for accurate token counting.

**Rationale**: Character-based truncation would cut words mid-token, producing garbled text. Token-based truncation respects word boundaries and accurately controls context size. Two separate limits allow fine-tuning — webpage content can be longer because it goes to a separate LLM, while observations need to fit within the main agent's context.

---

## Statistics Tracking

The system tracks two global counters:

```python
total_call = 0    # Total LLM calls made
failed_call = 0   # Calls that failed format validation
```

At the end of processing, it reports:
```
LLM call success rate: 95.23%
```

Each tool also tracks its own statistics (e.g., `Search` tracks `total_requests` and `successful_requests`).

**Rationale**: These metrics help diagnose problems. A low success rate might indicate the LLM is struggling with the output format, suggesting you need a more capable model or adjusted prompts.

---

## Summary: Why This Architecture Works

| Challenge | Solution | Benefit |
|-----------|----------|---------|
| Context window overflow | Markovian State Reconstruction | 2048+ turns in 40k tokens |
| LLM format errors | Validation + retries | Robust against formatting mistakes |
| Agent never stopping | Last turn prompt + turn limit | Every question gets an answer |
| Slow batch processing | ThreadPoolExecutor | Parallel question processing |
| Long tool outputs | Token-based truncation | Context stays within bounds |
| Knowledge loss between turns | `<report>` accumulates findings | No critical information lost |

---

## Next Steps

- **Want to trace the exact data flow?** → [Data Flow](data-flow.md)
- **Want to understand the prompts in detail?** → [Prompt Engineering](prompt-engineering.md)
- **Want to know why these design choices were made?** → [Design Rationale](design-rationale.md)
