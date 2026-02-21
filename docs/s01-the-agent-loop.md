# s01: The Agent Loop

> The entire secret of AI coding agents is a while loop that feeds tool results back to the model until the model decides to stop.

## The Problem

Why can't a language model just answer a coding question? Because coding
requires _interaction with the real world_. The model needs to read files,
run tests, check errors, and iterate. A single prompt-response pair cannot
do this.

Without the agent loop, you would have to copy-paste outputs back into the
model yourself. The user becomes the loop. The agent loop automates this:
call the model, execute whatever tools it asks for, feed the results back,
repeat until the model says "I'm done."

Consider a simple task: "Create a Python file that prints hello." The model
needs to (1) decide to write a file, (2) write it, (3) verify it works.
That is three tool calls minimum. Without a loop, each one requires manual
human intervention.

## The Solution

```
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> |  Tool   |
|  prompt  |      |       |      | execute |
+----------+      +---+---+      +----+----+
                      ^               |
                      |   tool_result |
                      +---------------+
                      (loop continues)

The loop terminates when stop_reason != "tool_use".
That single condition is the entire control flow.
```

## How It Works

1. The user provides a prompt. It becomes the first message.

```python
history.append({"role": "user", "content": query})
```

2. The messages array is sent to the LLM along with the tool definitions.

```python
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
```

3. The assistant response is appended to messages.

```python
messages.append({"role": "assistant", "content": response.content})
```

4. We check the stop reason. If the model did not call a tool, the loop
   ends. This is the only exit condition.

```python
if response.stop_reason != "tool_use":
    return
```

5. For each tool_use block in the response, execute the tool (bash in this
   session) and collect results.

```python
for block in response.content:
    if block.type == "tool_use":
        output = run_bash(block.input["command"])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

6. The results are appended as a user message, and the loop continues.

```python
messages.append({"role": "user", "content": results})
```

## Key Code

The minimum viable agent -- the entire pattern in under 30 lines
(from `agents/s01_agent_loop.py`, lines 66-86):

```python
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

## What Changed

This is session 1 -- the starting point. There is no prior session.

| Component     | Before     | After                          |
|---------------|------------|--------------------------------|
| Agent loop    | (none)     | `while True` + stop_reason     |
| Tools         | (none)     | `bash` (one tool)              |
| Messages      | (none)     | Accumulating list              |
| Control flow  | (none)     | `stop_reason != "tool_use"`    |

## Production Reference

In real Claude Code, this same loop runs inside the main REPL. The core
loop logic lives in `agent.ts` and follows the identical pattern: send
messages to the API, check `stop_reason`, execute tools, append results.
The production version adds error handling, token counting, streaming,
and retry logic, but the fundamental structure is unchanged. Every Claude
Code session is this while loop.

## Try It

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

Example prompts to try:

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
