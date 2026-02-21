# s07: Tasks

> Tasks persist as JSON files on the filesystem with a dependency graph, so they survive context compression and can be shared across agents.

## The Problem

In-memory state like TodoManager (s03) is lost when the context is
compressed (s06). After auto_compact replaces messages with a summary,
the todo list is gone. The agent has to reconstruct it from the summary
text, which is lossy and error-prone.

This is the critical s06-to-s07 bridge: TodoManager items die with
compression; file-based tasks don't. Moving state to the filesystem
makes it compression-proof.

More fundamentally, in-memory state is invisible to other agents.
When we eventually build teams (s09+), teammates need a shared task
board. In-memory data structures are process-local.

The solution is to persist tasks as JSON files in `.tasks/`. Each task
is a separate file with an ID, subject, status, and dependency graph.
Completing task 1 automatically unblocks task 2 if task 2 has
`blockedBy: [1]`. The file system becomes the source of truth.

## The Solution

```
.tasks/
  task_1.json  {"id":1, "status":"completed", ...}
  task_2.json  {"id":2, "blockedBy":[1], "status":"pending"}
  task_3.json  {"id":3, "blockedBy":[2], "status":"pending"}

Dependency resolution:
+----------+     +----------+     +----------+
| task 1   | --> | task 2   | --> | task 3   |
| complete |     | blocked  |     | blocked  |
+----------+     +----------+     +----------+
     |                ^
     +--- completing task 1 removes it from
          task 2's blockedBy list
```

## How It Works

1. The TaskManager provides CRUD operations. Each task is a JSON file.

```python
class TaskManager:
    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)
```

2. When a task is marked completed, `_clear_dependency` removes its ID
   from all other tasks' `blockedBy` lists.

```python
def _clear_dependency(self, completed_id: int):
    for f in self.dir.glob("task_*.json"):
        task = json.loads(f.read_text())
        if completed_id in task.get("blockedBy", []):
            task["blockedBy"].remove(completed_id)
            self._save(task)
```

3. The `update` method handles status changes and bidirectional dependency
   wiring.

```python
def update(self, task_id, status=None,
           add_blocked_by=None, add_blocks=None):
    task = self._load(task_id)
    if status:
        task["status"] = status
        if status == "completed":
            self._clear_dependency(task_id)
    if add_blocks:
        task["blocks"] = list(set(task["blocks"] + add_blocks))
        for blocked_id in add_blocks:
            blocked = self._load(blocked_id)
            if task_id not in blocked["blockedBy"]:
                blocked["blockedBy"].append(task_id)
                self._save(blocked)
    self._save(task)
```

4. Four task tools are added to the dispatch map.

```python
TOOL_HANDLERS = {
    # ...base tools...
    "task_create": lambda **kw: TASKS.create(kw["subject"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"],
                       kw.get("status")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}
```

## Key Code

The TaskManager with dependency graph (from `agents/s07_file_tasks.py`,
lines 46-123):

```python
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject, description=""):
        task = {"id": self._next_id, "subject": subject,
                "status": "pending", "blockedBy": [],
                "blocks": [], "owner": ""}
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id):
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)
```

## What Changed From s06

| Component      | Before (s06)     | After (s07)                |
|----------------|------------------|----------------------------|
| Tools          | 5                | 8 (+task_create/update/list/get)|
| State storage  | In-memory only   | JSON files in .tasks/      |
| Dependencies   | None             | blockedBy + blocks graph   |
| Compression    | Three-layer      | Removed (different focus)  |
| Persistence    | Lost on compact  | Survives compression       |

## Production Reference

Claude Code implements file-based tasks through the Task tool and
`.tasks/` directory. Each task is a JSON file with status tracking and
dependency resolution. The production version adds priority levels,
assignee fields, and integration with the team messaging system. Tasks
are the bridge between context compression and long-running work: the
agent can forget conversation details but always has the task board to
remind it what needs doing. The dependency graph ensures tasks execute
in the correct order even when the agent's context has been compressed.

## Try It

```sh
cd learn-claude-code
python agents/s07_file_tasks.py
```

Example prompts to try:

1. `Create 3 tasks: "Setup project", "Write code", "Write tests". Make them depend on each other in order.`
2. `List all tasks and show the dependency graph`
3. `Complete task 1 and then list tasks to see task 2 unblocked`
4. `Create a task board for refactoring: parse -> transform -> emit -> test`
