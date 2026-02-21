# s04: Subagents

> A subagent runs with a fresh messages list, shares the filesystem with the parent, and returns only a summary -- keeping the parent context clean.

## The Problem

As the agent works, its messages array grows. Every tool call, every file
read, every bash output accumulates. After 20-30 tool calls, the context
window is crowded with irrelevant history. Reading a 500-line file to
answer a quick question permanently adds 500 lines to the context.

This is particularly bad for exploratory tasks. "What testing framework
does this project use?" might require reading 5 files, but the parent
agent does not need all 5 file contents in its history -- it just needs
the answer: "pytest with conftest.py configuration."

The solution is process isolation: spawn a child agent with `messages=[]`.
The child explores, reads files, runs commands. When it finishes, only its
final text response returns to the parent. The child's entire message
history is discarded.

## The Solution

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ---------->| while tool_use:  |
|   prompt="..."   |            |   call tools     |
|                  |  summary   |   append results |
|   result = "..." | <--------- | return last text |
+------------------+             +------------------+
          |
Parent context stays clean.
Subagent context is discarded.
```

## How It Works

1. The parent agent gets a `task` tool that triggers subagent spawning.
   The child gets all base tools except `task` (no recursive spawning).

```python
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "task",
     "description": "Spawn a subagent with fresh context.",
     "input_schema": {
         "type": "object",
         "properties": {
             "prompt": {"type": "string"},
             "description": {"type": "string"},
         },
         "required": ["prompt"],
     }},
]
```

2. The subagent starts with a fresh messages list containing only
   the delegated prompt. It shares the same filesystem.

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({
            "role": "assistant", "content": response.content
        })
        if response.stop_reason != "tool_use":
            break
        # execute tools, append results...
```

3. Only the final text returns to the parent. The child's 30+ tool
   call history is discarded.

```python
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

4. The parent receives this summary as a normal tool_result.

```python
if block.name == "task":
    output = run_subagent(block.input["prompt"])
results.append({
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": str(output),
})
```

## Key Code

The subagent function (from `agents/s04_context_isolation.py`,
lines 110-128):

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant",
                             "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input)
                results.append({"type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

## What Changed From s03

| Component      | Before (s03)     | After (s04)               |
|----------------|------------------|---------------------------|
| Tools          | 5                | 5 (base) + task (parent)  |
| Context        | Single shared    | Parent + child isolation  |
| Subagent       | None             | `run_subagent()` function |
| Return value   | N/A              | Summary text only         |
| Todo system    | TodoManager      | Removed (not needed here) |

## Production Reference

Claude Code implements this as the Task tool with `spawn()`. When the
model calls Task, a child process is created with `messages=[]`. The
child inherits the filesystem and tool definitions but gets its own
context window. Claude Code also supports subagent descriptions that
appear in the parent's UI, so users can see what the child is doing
without polluting the parent context. The production version also limits
subagent depth (no recursive Task-from-Task) and enforces a max
iteration count.

## Try It

```sh
cd learn-claude-code
python agents/s04_context_isolation.py
```

Example prompts to try:

1. `Use a subtask to find what testing framework this project uses`
2. `Delegate: read all .py files and summarize what each one does`
3. `Use a task to create a new module, then verify it from here`
