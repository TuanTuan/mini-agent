# s09: Agent Teams

> Persistent teammates with JSONL inboxes turn isolated agents into a communicating team -- spawn, message, broadcast, and drain.

## The Problem

Subagents (s04) are disposable: spawn, work, return summary, die. They
have no identity, no memory between invocations, and no way to receive
follow-up instructions. Background tasks (s08) run shell commands but
cannot make LLM-guided decisions or communicate findings.

For real teamwork you need three things: (1) persistent agents that
survive beyond a single prompt, (2) identity and lifecycle management,
and (3) a communication channel between agents. Without messaging, even
persistent teammates are deaf and mute -- they can work in parallel but
never coordinate.

The solution combines a TeammateManager for spawning persistent named
agents with a MessageBus using JSONL inbox files. Each teammate runs
its own agent loop in a thread, checks its inbox before every LLM call,
and can send messages to any other teammate or the lead.

Note on the s06-to-s07 bridge: TodoManager items from s03 die with
compression (s06). File-based tasks (s07) survive compression because
they live on disk. Teams build on this same principle -- config.json and
inbox files persist outside the context window.

## The Solution

```
Teammate lifecycle:
  spawn -> WORKING -> IDLE -> WORKING -> ... -> SHUTDOWN

Communication:
  .team/
    config.json           <- team roster + statuses
    inbox/
      alice.jsonl         <- append-only, drain-on-read
      bob.jsonl
      lead.jsonl

                +--------+    send("alice","bob","...")    +--------+
                | alice  | -----------------------------> |  bob   |
                | loop   |    bob.jsonl << {json_line}    |  loop  |
                +--------+                                +--------+
                     ^                                         |
                     |        BUS.read_inbox("alice")          |
                     +---- alice.jsonl -> read + drain ---------+

5 message types:
+-------------------------+------------------------------+
| message                 | Normal text between agents   |
| broadcast               | Sent to all teammates        |
| shutdown_request        | Request graceful shutdown     |
| shutdown_response       | Approve/reject shutdown      |
| plan_approval_response  | Approve/reject plan          |
+-------------------------+------------------------------+
```

## How It Works

1. The TeammateManager maintains config.json with the team roster.
   Each member has a name, role, and status.

```python
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
```

2. `spawn()` creates a teammate and starts its agent loop in a thread.
   Re-spawning an idle teammate reactivates it.

```python
def spawn(self, name: str, role: str, prompt: str) -> str:
    member = self._find_member(name)
    if member:
        if member["status"] not in ("idle", "shutdown"):
            return f"Error: '{name}' is currently {member['status']}"
        member["status"] = "working"
    else:
        member = {"name": name, "role": role, "status": "working"}
        self.config["members"].append(member)
    self._save_config()
    thread = threading.Thread(
        target=self._teammate_loop,
        args=(name, role, prompt), daemon=True)
    self.threads[name] = thread
    thread.start()
    return f"Spawned teammate '{name}' (role: {role})"
```

3. The MessageBus handles JSONL inbox files. `send()` appends a JSON
   line; `read_inbox()` reads all lines and drains the file.

```python
class MessageBus:
    def send(self, sender, to, content,
             msg_type="message", extra=None):
        msg = {"type": msg_type, "from": sender,
               "content": content,
               "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name):
        path = self.dir / f"{name}.jsonl"
        if not path.exists():
            return "[]"
        msgs = [json.loads(l)
                for l in path.read_text().strip().splitlines()
                if l]
        path.write_text("")  # drain
        return json.dumps(msgs, indent=2)
```

4. Each teammate checks its inbox before every LLM call and injects
   received messages into the conversation context.

```python
def _teammate_loop(self, name, role, prompt):
    sys_prompt = f"You are '{name}', role: {role}, at {WORKDIR}."
    messages = [{"role": "user", "content": prompt}]
    for _ in range(50):
        inbox = BUS.read_inbox(name)
        if inbox != "[]":
            messages.append({"role": "user",
                "content": f"<inbox>{inbox}</inbox>"})
            messages.append({"role": "assistant",
                "content": "Noted inbox messages."})
        response = client.messages.create(
            model=MODEL, system=sys_prompt,
            messages=messages, tools=TOOLS)
        messages.append({"role": "assistant",
                         "content": response.content})
        if response.stop_reason != "tool_use":
            break
        # execute tools, append results...
    self._find_member(name)["status"] = "idle"
    self._save_config()
```

5. `broadcast()` sends the same message to all teammates except the
   sender.

```python
def broadcast(self, sender, content, teammates):
    count = 0
    for name in teammates:
        if name != sender:
            self.send(sender, name, content, "broadcast")
            count += 1
    return f"Broadcast to {count} teammates"
```

## Key Code

The TeammateManager + MessageBus core (from `agents/s09_team_messaging.py`):

```python
class TeammateManager:
    def spawn(self, name, role, prompt):
        member = self._find_member(name) or {
            "name": name, "role": role, "status": "working"
        }
        member["status"] = "working"
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt), daemon=True)
        thread.start()
        return f"Spawned '{name}'"

class MessageBus:
    def send(self, sender, to, content,
             msg_type="message", extra=None):
        msg = {"type": msg_type, "from": sender,
               "content": content, "timestamp": time.time()}
        if extra: msg.update(extra)
        with open(self.dir / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")

    def read_inbox(self, name):
        path = self.dir / f"{name}.jsonl"
        if not path.exists(): return "[]"
        msgs = [json.loads(l)
                for l in path.read_text().strip().splitlines()
                if l]
        path.write_text("")
        return json.dumps(msgs, indent=2)
```

## What Changed From s08

| Component      | Before (s08)     | After (s09)                |
|----------------|------------------|----------------------------|
| Tools          | 6                | 9 (+spawn/send/read_inbox) |
| Agents         | Single           | Lead + N teammates         |
| Persistence    | None             | config.json + JSONL inboxes|
| Threads        | Background cmds  | Full agent loops per thread|
| Lifecycle      | Fire-and-forget  | idle -> working -> idle    |
| Communication  | None             | 5 message types + broadcast|

Teaching simplification: this implementation does not use lock files
for inbox access. In production, concurrent append from multiple writers
would need file locking or atomic rename. The single-writer-per-inbox
pattern used here is safe for the teaching scenario.

## Production Reference

Claude Code implements teammate management through the TeammateManager
and messaging through the SendMessage tool with JSONL inbox files. The
production system supports the same 5 message types and uses the same
append/drain pattern. Messages are delivered asynchronously -- the
sender does not block waiting for the recipient to read. The inbox
check before each LLM call ensures messages arrive at a natural
breakpoint. Production adds message deduplication, size limits, and
proper file locking.

## Try It

```sh
cd learn-claude-code
python agents/s09_team_messaging.py
```

Example prompts to try:

1. `Spawn alice (coder) and bob (tester). Have alice send bob a message.`
2. `Broadcast "status update: phase 1 complete" to all teammates`
3. `Check the lead inbox for any messages`
4. Type `/team` to see the team roster with statuses
5. Type `/inbox` to manually check the lead's inbox
