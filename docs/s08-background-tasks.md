# s08: Background Tasks

> A BackgroundManager runs commands in separate threads and drains a notification queue before each LLM call, so the agent never blocks on long-running operations.

## The Problem

Some commands take minutes: `npm install`, `pytest`, `docker build`. With
a blocking agent loop, the model sits idle waiting for the subprocess to
finish. It cannot do anything else. If the user asked "install dependencies
and while that runs, create the config file," the agent would install
first, _then_ create the config -- sequentially, not in parallel.

The agent needs concurrency. Not full multi-threading of the agent loop
itself, but the ability to fire off a long command and continue working
while it runs. When the command finishes, its result should appear
naturally in the conversation.

The solution is a BackgroundManager that runs commands in daemon threads
and collects results in a notification queue. Before each LLM call, the
queue is drained and results are injected into the messages.

## The Solution

```
Main thread                Background thread
+-----------------+        +-----------------+
| agent loop      |        | task executes   |
| ...             |        | ...             |
| [LLM call] <---+------- | enqueue(result) |
|  ^drain queue   |        +-----------------+
+-----------------+

Timeline:
Agent --[spawn A]--[spawn B]--[other work]----
             |          |
             v          v
          [A runs]   [B runs]      (parallel)
             |          |
             +-- notification queue --+
                                      |
                           [results injected before
                            next LLM call]
```

## How It Works

1. The BackgroundManager tracks tasks and maintains a thread-safe
   notification queue.

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()
```

2. `run()` starts a daemon thread and returns a task_id immediately.

```python
def run(self, command: str) -> str:
    task_id = str(uuid.uuid4())[:8]
    self.tasks[task_id] = {
        "status": "running",
        "result": None,
        "command": command,
    }
    thread = threading.Thread(
        target=self._execute,
        args=(task_id, command),
        daemon=True,
    )
    thread.start()
    return f"Background task {task_id} started"
```

3. The thread target `_execute` runs the subprocess and pushes
   results to the notification queue.

```python
def _execute(self, task_id: str, command: str):
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=300)
        output = (r.stdout + r.stderr).strip()[:50000]
        status = "completed"
    except subprocess.TimeoutExpired:
        output = "Error: Timeout (300s)"
        status = "timeout"
    self.tasks[task_id]["status"] = status
    self.tasks[task_id]["result"] = output
    with self._lock:
        self._notification_queue.append({
            "task_id": task_id,
            "status": status,
            "result": output[:500],
        })
```

4. `drain_notifications()` returns and clears pending results.

```python
def drain_notifications(self) -> list:
    with self._lock:
        notifs = list(self._notification_queue)
        self._notification_queue.clear()
    return notifs
```

5. The agent loop drains notifications before each LLM call.

```python
def agent_loop(messages: list):
    while True:
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: "
                f"{n['result']}" for n in notifs
            )
            messages.append({"role": "user",
                "content": f"<background-results>"
                           f"\n{notif_text}\n"
                           f"</background-results>"})
            messages.append({"role": "assistant",
                "content": "Noted background results."})
        response = client.messages.create(...)
```

## Key Code

The BackgroundManager (from `agents/s08_background.py`, lines 49-107):

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running",
                               "result": None,
                               "command": command}
        thread = threading.Thread(
            target=self._execute,
            args=(task_id, command), daemon=True)
        thread.start()
        return f"Background task {task_id} started"

    def _execute(self, task_id, command):
        # run subprocess, push to queue
        ...

    def drain_notifications(self) -> list:
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs
```

## What Changed From s07

| Component      | Before (s07)     | After (s08)                |
|----------------|------------------|----------------------------|
| Tools          | 8                | 6 (base + background_run + check)|
| Execution      | Blocking only    | Blocking + background threads|
| Notification   | None             | Queue drained per loop     |
| Concurrency    | None             | Daemon threads             |
| Task system    | File-based CRUD  | Removed (different focus)  |

## Production Reference

Claude Code runs background tasks as separate processes. Long-running
commands like test suites or builds are launched asynchronously, and
their results are delivered to the agent through a notification system.
The production version adds process groups for cleanup, progress
reporting, and cancellation support. The drain-before-LLM-call pattern
ensures background results arrive at a natural point in the conversation
rather than interrupting mid-thought.

## Try It

```sh
cd learn-claude-code
python agents/s08_background.py
```

Example prompts to try:

1. `Run "sleep 5 && echo done" in the background, then create a file while it runs`
2. `Start 3 background tasks: "sleep 2", "sleep 4", "sleep 6". Check their status.`
3. `Run pytest in the background and keep working on other things`
