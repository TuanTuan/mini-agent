# s06: Compact

> A three-layer compression pipeline lets the agent work indefinitely by strategically forgetting old tool results, auto-summarizing when tokens exceed a threshold, and allowing manual compression on demand.

## The Problem

The context window is finite. After enough tool calls, the messages array
exceeds the model's context limit and the API call fails. Even before
hitting the hard limit, performance degrades: the model becomes slower,
less accurate, and starts ignoring earlier messages.

A 200,000 token context window sounds large, but a single `read_file` on
a 1000-line source file consumes ~4000 tokens. After reading 30 files and
running 20 bash commands, you are at 100,000+ tokens. The agent cannot
work on large codebases without some form of compression.

The three-layer pipeline addresses this with increasing aggressiveness:
Layer 1 (micro-compact) silently replaces old tool results every turn.
Layer 2 (auto-compact) triggers a full summarization when tokens exceed
a threshold. Layer 3 (manual compact) lets the model trigger compression
itself.

Teaching simplification: the token estimation here uses a rough
characters/4 heuristic. Production systems use proper tokenizer
libraries for accurate counts.

## The Solution

```
Every turn:
+------------------+
| Tool call result |
+------------------+
        |
        v
[Layer 1: micro_compact]        (silent, every turn)
  Replace tool_result > 3 turns old
  with "[Previous: used {tool_name}]"
        |
        v
[Check: tokens > 50000?]
   |               |
   no              yes
   |               |
   v               v
continue    [Layer 2: auto_compact]
              Save transcript to .transcripts/
              LLM summarizes conversation.
              Replace all messages with [summary].
                    |
                    v
            [Layer 3: compact tool]
              Model calls compact explicitly.
              Same summarization as auto_compact.
```

## How It Works

1. **Layer 1 -- micro_compact**: Before each LLM call, find all
   tool_result entries older than the last 3 and replace their content.

```python
def micro_compact(messages: list) -> list:
    tool_results = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for j, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((i, j, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, part in to_clear:
        if len(part.get("content", "")) > 100:
            tool_id = part.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            part["content"] = f"[Previous: used {tool_name}]"
    return messages
```

2. **Layer 2 -- auto_compact**: When estimated tokens exceed 50,000,
   save the full transcript and ask the LLM to summarize.

```python
def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity..."
            + json.dumps(messages, default=str)[:80000]}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    return [
        {"role": "user", "content": f"[Compressed]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. Continuing."},
    ]
```

3. **Layer 3 -- manual compact**: The `compact` tool triggers the same
   summarization on demand.

```python
if manual_compact:
    messages[:] = auto_compact(messages)
```

4. The agent loop integrates all three layers.

```python
def agent_loop(messages: list):
    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)
        response = client.messages.create(...)
        # ... tool execution ...
        if manual_compact:
            messages[:] = auto_compact(messages)
```

## Key Code

The three-layer pipeline (from `agents/s06_compression.py`,
lines 67-93 and 189-223):

```python
THRESHOLD = 50000
KEEP_RECENT = 3

def micro_compact(messages):
    # Replace old tool results with placeholders
    ...

def auto_compact(messages):
    # Save transcript, LLM summarize, replace messages
    ...

def agent_loop(messages):
    while True:
        micro_compact(messages)          # Layer 1
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)  # Layer 2
        response = client.messages.create(...)
        # ...
        if manual_compact:
            messages[:] = auto_compact(messages)  # Layer 3
```

## What Changed From s05

| Component      | Before (s05)     | After (s06)                |
|----------------|------------------|----------------------------|
| Tools          | 5                | 5 (base + compact)         |
| Context mgmt   | None             | Three-layer compression    |
| Micro-compact  | None             | Old results -> placeholders|
| Auto-compact   | None             | Token threshold trigger    |
| Manual compact | None             | `compact` tool             |
| Transcripts    | None             | Saved to .transcripts/     |
| Skills         | load_skill       | Removed (different focus)  |

## Production Reference

Claude Code implements all three layers. Micro-compaction happens
automatically -- old tool results are truncated or replaced with
summaries. Auto-compaction triggers when the context approaches the
model's limit, summarizing the conversation and resetting. The user can
also trigger manual compaction with `/compact`. Transcripts are saved so
the full history is never truly lost, just moved out of the active
context. The production version also preserves "pinned" messages that
should never be compressed (like CLAUDE.md instructions).

## Try It

```sh
cd learn-claude-code
python agents/s06_compression.py
```

Example prompts to try:

1. `Read every Python file in the agents/ directory one by one`
   (watch micro-compact replace old results)
2. `Keep reading files until compression triggers automatically`
3. `Use the compact tool to manually compress the conversation`
