# Prompt Engineering

> This document explains the prompt system that guides the IterResearch agent — what each prompt does, why it's structured the way it is, and how the prompts enforce the agent's reasoning discipline.

---

## Why Prompts Matter So Much

In IterResearch, the prompts are not just instructions — they are the **architecture of the agent's mind**. The agent has no hardcoded logic for "what to research next" or "when to stop searching." All of that behavior emerges from how the prompts are written.

The prompt system enforces:
1. **Structured output** — XML-style tags ensure parseable responses
2. **Evidence-driven reasoning** — The agent must document facts vs. hypotheses
3. **Progressive knowledge accumulation** — Each turn builds on the previous report
4. **Graceful termination** — The agent knows when to stop and answer

---

## The Four Prompts

```
Turn 0           Turn 1 ... N-2         Turn N-1
  │                    │                    │
  ▼                    ▼                    ▼
initial_          instruction_         last_instruction_
instruction_      prompt               prompt
prompt                │                    │
  │                    │                    │
  ▼                    ▼                    ▼
<report>          <report>             <report>
<tool_call>       <tool_call>          <answer>
                  OR <answer>          (MUST answer)
```

---

## Prompt 1: `initial_instruction_prompt`

**When used**: First turn only  
**Source**: `prompts/prompts.py`, lines 10–60  

### What It Tells the Agent

1. **Identity**: "You are a professional problem-solving agent with rigorous information verification capabilities and deep analytical thinking."
2. **Output format**: Must produce `<report>` AND `<tool_call>` tags (no `<answer>` allowed on the first turn)
3. **Input data**: The question, available tools (JSON), and current date
4. **Report structure**: Must analyze the "Problem Essence" and create a "Problem-Solving Plan"
5. **Working principles**: Deep thinking, evidence-driven reasoning, detailed documentation

### Key Design Decisions

**No answer on the first turn**: The agent MUST make a tool call on turn 0. It cannot immediately answer.

```
You MUST raise one and only one function call
```

**Rationale**: This prevents the agent from guessing answers without doing any research. Even for simple questions, the agent must at least do one search to verify its knowledge. This dramatically improves answer quality and accuracy.

**Problem Essence Analysis**: The agent is required to analyze the "core elements, complexity levels, and potential pitfalls" before doing anything.

**Rationale**: This forces the agent to think before acting. Without this, agents tend to make their first search query based on a superficial reading of the question, leading to irrelevant results. By analyzing the problem first, the agent makes better initial tool choices.

### Template Variables

| Variable | What It's Replaced With |
|----------|------------------------|
| `{question}` | The user's research question |
| `{tools}` | JSON string of all tool definitions |
| `{date_to_use}` | A date string for temporal context |

---

## Prompt 2: `instruction_prompt`

**When used**: Turns 1 through N-2 (the main research phase)  
**Source**: `prompts/prompts.py`, lines 62–155  

### What It Tells the Agent

This is the most detailed prompt. It provides:

1. **Previous context**: The last report, last tool call, and last tool response
2. **Output options**: Either `<report>` + `<answer>` OR `<report>` + `<tool_call>` (never both)
3. **Decision point**: Explicitly asks "Are you certain that no further verification or information gathering is needed?"
4. **Answer format guidelines**: Detailed instructions on how to format the final answer

### The Report Section: Progressive Knowledge Accumulation

This is the most important part of the prompt:

```
**Progress Achieved:**
[Based on the Last Status Report and Deep Analysis and Last Tool Response
provided in the input, compile a comprehensive and complete documentation
of all currently collected information, conclusions, data, and findings.
This section must capture ALL important information without any omissions...]
```

The agent is required to produce a comprehensive report that includes:
1. All factual data and evidence collected
2. All analytical conclusions and insights derived
3. All source materials and their verification status
4. All uncertainties, limitations, or gaps identified
5. Complete integration of previous progress with new findings

**Rationale**: This is the mechanism that makes Markovian State Reconstruction work. Since the message history is reset each turn, the `<report>` is the ONLY way knowledge persists between turns. If the agent fails to document something in the report, that information is lost forever. By requiring comprehensive documentation, the prompt ensures minimal information loss.

### The Decision Point: When to Stop

```
**Decision Point**: Are you certain that no further verification
or information gathering is needed to provide the final answer?

**If YES - Information is sufficient:**
    <answer>...</answer>

**If NO - Further action needed:**
    <tool_call>...</tool_call>
```

**Rationale**: Making the stop/continue decision explicit (rather than implicit) forces the agent to consciously evaluate whether it has enough information. This prevents two common failure modes:
- **Premature stopping**: Answering with insufficient evidence
- **Infinite searching**: Never deciding that enough is enough

### Answer Format Guidelines

The prompt gives detailed instructions for formatting the final answer:

1. Match the language of the question (Chinese question → Chinese answer)
2. Include as much relevant content as possible
3. Organize into paragraphs to avoid overly long sections
4. Don't include URLs or images from reasoning
5. Start with a direct answer, then elaborate
6. Use markdown tables for statistical data (3+ items)
7. For research questions, target 1000+ words with subheadings

**Rationale**: These guidelines produce high-quality, well-structured answers. The language matching ensures the output language matches the input. The markdown formatting makes answers readable. The word count guidance prevents overly brief responses for complex questions.

### Template Variables

| Variable | What It's Replaced With |
|----------|------------------------|
| `{question}` | The original research question |
| `{tools}` | JSON string of all tool definitions |
| `{date_to_use}` | Date string |
| `{report}` | Previous turn's `<report>` content |
| `{action}` | Previous turn's `<tool_call>` content |
| `{observation}` | Tool response (possibly truncated) |

---

## Prompt 3: `last_instruction_prompt`

**When used**: Final turn only (turn N-1)  
**Source**: `prompts/prompts.py`, lines 157–232  

### What It Tells the Agent

Almost identical to `instruction_prompt`, but with two key differences:

1. **Must answer**: `<report>` AND `<answer>` are both required
2. **No tools available**: The tool list is NOT included — the agent cannot make any more tool calls

```
You MUST follow this exact format. Every response must contain:
1. <report>...</report> (always required)
2. <answer>...</answer> (always required)
```

**Rationale**: The last turn acts as a safety net. Even if the agent would prefer to keep researching, it's forced to synthesize everything it knows into a final answer. Removing the tool list from the prompt reinforces that no more tools are available.

### Template Variables

Same as `instruction_prompt` but WITHOUT `{tools}`.

---

## Prompt 4: `observation_prompt`

**When used**: After every tool execution  
**Source**: `prompts/prompts.py`, lines 235–236  

### The Simplest Prompt

```python
observation_prompt = '''**Tool results**: 
{tool_response}'''
```

This is a minimal wrapper that labels the tool output so the LLM knows it's a tool response, not user text.

### Template Variable

| Variable | What It's Replaced With |
|----------|------------------------|
| `{tool_response}` | The raw output from the tool |

**Rationale**: The observation prompt is intentionally minimal to avoid consuming unnecessary context tokens. The tool output is often very long (search results, webpage content), so the wrapper should add minimal overhead.

---

## Prompt 5: `EXTRACTOR_PROMPT` (Visit Tool)

**Where**: `tools/visit.py`, lines 42–56  
**When used**: By the Summary LLM when processing webpage content

### What It Does

This prompt instructs the Summary LLM to process a webpage and extract relevant information:

```
Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. Content Scanning for Rational: Locate specific sections related to the goal
2. Key Extraction for Evidence: Extract the most relevant information with full context
3. Summary Output for Summary: Organize into a concise paragraph

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
```

### The Three-Part Output

| Field | Purpose | Typical Content |
|-------|---------|-----------------|
| `rational` | Why this content matters | "This section discusses the attention mechanism which directly relates to the user's question about..." |
| `evidence` | Raw extracted information | Full paragraphs from the webpage (can be very long) |
| `summary` | Distilled takeaway | 1-2 paragraph summary of key findings |

**Rationale**: The three-part structure ensures the extraction is thorough:
- `rational` forces the Summary LLM to think about relevance before extracting
- `evidence` preserves the original text for accuracy
- `summary` provides a digestible version for the main agent

Asking for "full original context" in the evidence field prevents the common problem of LLMs over-summarizing and losing important details.

---

## Prompt Design Principles

### 1. Structured Output via XML Tags

All prompts require XML-style tags (`<report>`, `<tool_call>`, `<answer>`).

**Why not JSON?** LLMs are more reliable at producing text wrapped in simple tags than perfectly formatted JSON. Tags also allow the content inside to be freeform text (like the report) or JSON (like the tool call), supporting mixed formats.

**Why not function calling?** IterResearch works with any OpenAI-compatible endpoint, and not all of them support function calling. XML tags in the text output are universally supported.

### 2. Explicit Constraints

The prompts repeat key constraints multiple times:

```
You MUST follow this exact format.
You MUST output this section enclosed with <report></report> tags!
You MUST raise one and only one function call
```

**Rationale**: LLMs sometimes "forget" instructions in long prompts. Repetition at the beginning and end increases compliance. The `## FORMAT REMINDER` section at the bottom of each prompt serves as a final reinforcement.

### 3. Role-Based Framing

```
You are a professional problem-solving agent with rigorous information
verification capabilities and deep analytical thinking.
```

**Rationale**: Setting a professional, rigorous identity encourages the LLM to be thorough and evidence-based rather than casual or speculative.

### 4. Language Adaptation

```
The language of your output must be consistent with the language of the question.
If the question is in Chinese, output in Chinese; if the question is in English, output in English.
```

**Rationale**: Since IterResearch supports both Chinese and English questions (the paper originated from a Chinese research group), the prompts explicitly instruct language matching.

### 5. Documentation Mandate

```
You are required to maintain detailed documentation in all your reports
and actions, providing sufficient information for others to fully grasp
your progress and effectively continue or modify the research trajectory
based on your contributions.
```

**Rationale**: This isn't just about quality — it's architecturally necessary. Since Markovian State Reconstruction discards the raw conversation history, the report is the ONLY persistent memory. Poor documentation = lost knowledge.

---

## How Prompts Flow Through the System

```
Turn 0:
  initial_instruction_prompt(question, tools, date) → LLM → response

Turn 1:
  instruction_prompt(question, tools, date, report₀, action₀, observation₀) → LLM → response

Turn 2:
  instruction_prompt(question, tools, date, report₁, action₁, observation₁) → LLM → response

  ...

Turn N-1:
  last_instruction_prompt(question, date, report_{N-2}, action_{N-2}, observation_{N-2}) → LLM → response

Each "observation" comes from:
  observation_prompt(tool_response) [then truncated if needed]

Each "Visit" tool internally uses:
  EXTRACTOR_PROMPT(webpage_content, goal) → Summary LLM → JSON
```

---

## Next Steps

- **Want to see how format validation works?** → [Agent Architecture](agent-architecture.md)
- **Want to configure prompt parameters?** → [Configuration Guide](configuration-guide.md)
- **Want to know why these prompts were designed this way?** → [Design Rationale](design-rationale.md)
