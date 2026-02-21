# s11: Autonomous Agents

> An idle cycle with task board polling lets teammates find and claim work themselves, with identity re-injection after context compression.

## The Problem

In s09-s10, teammates only work when explicitly told to. The lead must
spawn each teammate with a specific prompt. If the task board has 10
unclaimed tasks, the lead must manually assign each one. This does not
scale.

True autonomy means teammates find work themselves. When a teammate
finishes its current task, it should scan the task board for unclaimed
work, claim a task, and start working -- without any instruction from
the lead.

But autonomous agents face a subtlety: after context compression, the
agent might forget who it is. If the messages are summarized, the
original system prompt identity ("you are alice, role: coder") fades.
Identity re-injection solves this by inserting an identity block at the
start of compressed contexts.

Teaching simplification: the token estimation used here is rough
(characters / 4). Production systems use proper tokenizer libraries.
The nag threshold of 3 rounds (from s03) is set low for teaching
visibility; production Claude Code uses a threshold around 10.

## The Solution

```
Teammate lifecycle with idle cycle:

+-------+
| spawn |
+---+---+
    |
    v
+-------+   tool_use     +-------+
| WORK  | <------------- |  LLM  |
+---+---+                +-------+
    |
    | stop_reason != tool_use
    | (or idle tool called)
    v
+--------+
|  IDLE  |  poll every 5s for up to 60s
+---+----+
    |
    +---> check inbox --> message? ----------> WORK
    |
    +---> scan .tasks/ --> unclaimed? -------> claim -> WORK
    |
    +---> 60s timeout ----------------------> SHUTDOWN

Identity re-injection after compression:
  if len(messages) <= 3:
    messages.insert(0, identity_block)
    "You are 'alice', role: coder, team: my-team"
```

## How It Works

1. The teammate loop has two phases: WORK and IDLE. WORK runs the
   standard agent loop. When the LLM stops calling tools (or calls
   the `idle` tool), the teammate enters the IDLE phase.

```python
def _loop(self, name, role, prompt):
    while True:
        # -- WORK PHASE --
        messages = [{"role": "user", "content": prompt}]
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    self._set_status(name, "shutdown")
                    return
                messages.append(...)
            response = client.messages.create(...)
            if response.stop_reason != "tool_use":
                break
            # execute tools...
            if idle_requested:
                break

        # -- IDLE PHASE --
        self._set_status(name, "idle")
        resume = self._idle_poll(name, messages)
        if not resume:
            self._set_status(name, "shutdown")
            return
        self._set_status(name, "working")
```

2. The idle phase polls the inbox and task board in a loop.

```python
def _idle_poll(self, name, messages):
    polls = IDLE_TIMEOUT // POLL_INTERVAL  # 60s / 5s = 12
    for _ in range(polls):
        time.sleep(POLL_INTERVAL)
        # Check inbox for new messages
        inbox = BUS.read_inbox(name)
        if inbox:
            messages.append({"role": "user",
                "content": f"<inbox>{inbox}</inbox>"})
            return True
        # Scan task board for unclaimed tasks
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task = unclaimed[0]
            claim_task(task["id"], name)
            messages.append({"role": "user",
                "content": f"<auto-claimed>Task #{task['id']}: "
                           f"{task['subject']}</auto-claimed>"})
            return True
    return False  # timeout -> shutdown
```

3. Task board scanning looks for pending, unowned, unblocked tasks.

```python
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed

def claim_task(task_id: int, owner: str):
    path = TASKS_DIR / f"task_{task_id}.json"
    task = json.loads(path.read_text())
    task["status"] = "in_progress"
    task["owner"] = owner
    path.write_text(json.dumps(task, indent=2))
```

4. Identity re-injection inserts an identity block when the context
   is too short, indicating compression has occurred.

```python
def make_identity_block(name, role, team_name):
    return {"role": "user",
            "content": f"<identity>You are '{name}', "
                       f"role: {role}, team: {team_name}. "
                       f"Continue your work.</identity>"}

# Before resuming work after idle:
if len(messages) <= 3:
    messages.insert(0, make_identity_block(
        name, role, team_name))
    messages.insert(1, {"role": "assistant",
        "content": f"I am {name}. Continuing."})
```

5. The `idle` tool lets the teammate explicitly signal it has no more
   work, entering the idle polling phase early.

```python
{"name": "idle",
 "description": "Signal that you have no more work. "
                "Enters idle polling phase.",
 "input_schema": {"type": "object", "properties": {}}},
```

## Key Code

The autonomous loop (from `agents/s11_autonomous.py`):

```python
def _loop(self, name, role, prompt):
    while True:
        # WORK PHASE
        for _ in range(50):
            response = client.messages.create(...)
            if response.stop_reason != "tool_use":
                break
            for block in response.content:
                if block.name == "idle":
                    idle_requested = True
            if idle_requested:
                break

        # IDLE PHASE
        self._set_status(name, "idle")
        for _ in range(IDLE_TIMEOUT // POLL_INTERVAL):
            time.sleep(POLL_INTERVAL)
            inbox = BUS.read_inbox(name)
            if inbox: resume = True; break
            unclaimed = scan_unclaimed_tasks()
            if unclaimed:
                claim_task(unclaimed[0]["id"], name)
                resume = True; break
        if not resume:
            self._set_status(name, "shutdown")
            return
        self._set_status(name, "working")
```

## What Changed From s10

| Component      | Before (s10)     | After (s11)                |
|----------------|------------------|----------------------------|
| Tools          | 12               | 14 (+idle, +claim_task)    |
| Autonomy       | Lead-directed    | Self-organizing            |
| Idle phase     | None             | Poll inbox + task board    |
| Task claiming  | Manual only      | Auto-claim unclaimed tasks |
| Identity       | System prompt    | + re-injection after compress|
| Timeout        | None             | 60s idle -> auto shutdown  |

## Production Reference

Claude Code implements autonomous teammates with the same idle/poll/claim
cycle. The production version uses a TaskUpdate tool for status changes
and an idle notification system. Teammates automatically discover work
from the task board and claim it without lead intervention. Identity
re-injection ensures teammates maintain their role and context after
compaction. The production idle timeout is configurable per teammate.
The key insight: the lead creates the task board, teammates self-organize
to complete it.

## Try It

```sh
cd learn-claude-code
python agents/s11_autonomous.py
```

Example prompts to try:

1. `Create 3 tasks on the board, then spawn alice and bob. Watch them auto-claim.`
2. `Spawn a coder teammate and let it find work from the task board itself`
3. `Create tasks with dependencies. Watch teammates respect the blocked order.`
4. Type `/tasks` to see the task board with owners
5. Type `/team` to monitor who is working vs idle
