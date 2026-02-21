# s03: TodoWrite

> A TodoManager lets the agent track its own progress, and a nag reminder injection forces it to keep updating when it forgets.

## The Problem

When an agent works on a multi-step task, it often loses track of what it
has done and what remains. Without explicit planning, the model might repeat
work, skip steps, or wander off on tangents. The user has no visibility
into the agent's internal plan.

This is worse than it sounds. Long conversations cause the model to "drift"
-- the system prompt fades in influence as the context window fills with
tool results. A 10-step refactoring task might complete steps 1-3, then
the model starts improvising because it forgot steps 4-10 existed.

The solution is structured state: a TodoManager that the model writes to
explicitly. The model creates a plan, marks items in_progress as it works,
and marks them completed when done. A nag reminder injects a nudge if the
model goes 3+ rounds without updating its todos.

Teaching simplification: the nag threshold of 3 rounds is set low for
teaching visibility. Production Claude Code uses a threshold around 10
to avoid over-prompting.

## The Solution

```
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> | Tools   |
|  prompt  |      |       |      | + todo  |
+----------+      +---+---+      +----+----+
                      ^               |
                      |   tool_result |
                      +---------------+
                            |
                +-----------+-----------+
                | TodoManager state     |
                | [ ] task A            |
                | [>] task B  <- doing  |
                | [x] task C            |
                +-----------------------+
                            |
                if rounds_since_todo >= 3:
                  inject <reminder> into tool_result
```

## How It Works

1. The TodoManager validates and stores a list of items with statuses.
   Only one item can be `in_progress` at a time.

```python
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated = []
        in_progress_count = 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({
                "id": item["id"],
                "text": item["text"],
                "status": status,
            })
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress")
        self.items = validated
        return self.render()
```

2. The `todo` tool is added to the dispatch map like any other tool.

```python
TOOL_HANDLERS = {
    "bash":  lambda **kw: run_bash(kw["command"]),
    # ...other tools...
    "todo":  lambda **kw: TODO.update(kw["items"]),
}
```

3. The nag reminder injects a `<reminder>` tag into the tool_result
   messages when the model goes 3+ rounds without calling `todo`.

```python
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if (last["role"] == "user"
                    and isinstance(last.get("content"), list)):
                last["content"].insert(0, {
                    "type": "text",
                    "text": "<reminder>Update your todos.</reminder>",
                })
        # ... rest of loop ...
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
```

4. The system prompt instructs the model to use todos for planning.

```python
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks.
Mark in_progress before starting, completed when done.
Prefer tools over prose."""
```

## Key Code

The TodoManager and nag injection (from `agents/s03_structured_planning.py`,
lines 51-85 and 158-187):

```python
class TodoManager:
    def update(self, items: list) -> str:
        validated = []
        in_progress_count = 0
        for item in items:
            status = item.get("status", "pending")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({
                "id": item["id"],
                "text": item["text"],
                "status": status,
            })
        if in_progress_count > 1:
            raise ValueError("Only one in_progress")
        self.items = validated
        return self.render()

# In agent_loop:
if rounds_since_todo >= 3:
    last["content"].insert(0, {
        "type": "text",
        "text": "<reminder>Update your todos.</reminder>",
    })
```

## What Changed From s02

| Component      | Before (s02)     | After (s03)              |
|----------------|------------------|--------------------------|
| Tools          | 4                | 5 (+todo)                |
| Planning       | None             | TodoManager with statuses|
| Nag injection  | None             | `<reminder>` after 3 rounds|
| Agent loop     | Simple dispatch  | + rounds_since_todo counter|

## Production Reference

Claude Code implements the same pattern through TodoWrite and TodoRead
tools. The model creates structured todo lists with statuses (pending,
in_progress, completed). The nag reminder concept appears as "system
reminders" -- injected messages that steer the model back on track when
it drifts. The production TodoManager supports priorities and persistence
across sessions. The constraint of "one in_progress at a time" forces
sequential focus, preventing the model from context-switching too rapidly.

## Try It

```sh
cd learn-claude-code
python agents/s03_structured_planning.py
```

Example prompts to try:

1. `Refactor the file hello.py: add type hints, docstrings, and a main guard`
2. `Create a Python package with __init__.py, utils.py, and tests/test_utils.py`
3. `Review all Python files and fix any style issues`
