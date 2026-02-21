# s07: Tasks

> タスクはファイルシステム上にJSON形式で依存グラフ付きで永続化され、コンテキスト圧縮後も生き残り、複数エージェント間で共有できる。

## 問題

インメモリの状態であるTodoManager(s03)は、コンテキストが圧縮(s06)されると失われる。auto_compactがメッセージを要約で置換した後、todoリストは消える。エージェントは要約テキストからそれを再構成しなければならないが、これは不正確でエラーが起きやすい。

これがs06からs07への重要な橋渡しだ: TodoManagerのアイテムは圧縮と共に死ぬが、ファイルベースのタスクは死なない。状態をファイルシステムに移すことで、圧縮に対する耐性が得られる。

さらに根本的な問題として、インメモリの状態は他のエージェントからは見えない。最終的にチーム(s09以降)を構築する際、チームメイトには共有のタスクボードが必要だ。インメモリのデータ構造はプロセスローカルだ。

解決策はタスクを`.tasks/`にJSON形式で永続化すること。各タスクはID、件名、ステータス、依存グラフを持つ個別のファイルだ。タスク1を完了すると、タスク2が`blockedBy: [1]`を持つ場合、自動的にタスク2のブロックが解除される。ファイルシステムが信頼できる情報源となる。

## 解決策

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

## 仕組み

1. TaskManagerがCRUD操作を提供する。各タスクは1つのJSONファイル。

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

2. タスクが完了とマークされると、`_clear_dependency`がそのIDを他のすべてのタスクの`blockedBy`リストから除去する。

```python
def _clear_dependency(self, completed_id: int):
    for f in self.dir.glob("task_*.json"):
        task = json.loads(f.read_text())
        if completed_id in task.get("blockedBy", []):
            task["blockedBy"].remove(completed_id)
            self._save(task)
```

3. `update`メソッドがステータス変更と双方向の依存関係の結線を処理する。

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

4. 4つのタスクツールがディスパッチマップに追加される。

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

## 主要コード

依存グラフ付きTaskManager(`agents/s07_file_tasks.py` 46-123行目):

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

## s06からの変更点

| Component      | Before (s06)     | After (s07)                |
|----------------|------------------|----------------------------|
| Tools          | 5                | 8 (+task_create/update/list/get)|
| State storage  | In-memory only   | JSON files in .tasks/      |
| Dependencies   | None             | blockedBy + blocks graph   |
| Compression    | Three-layer      | Removed (different focus)  |
| Persistence    | Lost on compact  | Survives compression       |

## 本番環境との対応

Claude CodeではTaskツールと`.tasks/`ディレクトリを通じてファイルベースのタスクを実装している。各タスクはステータス追跡と依存関係解決を持つJSONファイルだ。本番バージョンでは優先度レベル、担当者フィールド、チームメッセージングシステムとの統合が追加されている。タスクはコンテキスト圧縮と長時間作業の橋渡しとなる: エージェントは会話の詳細を忘れても、タスクボードが常に何をすべきかを思い出させてくれる。依存グラフにより、エージェントのコンテキストが圧縮された後でもタスクは正しい順序で実行される。

## 試してみる

```sh
cd learn-claude-code
python agents/s07_file_tasks.py
```

試せるプロンプト例:

1. `Create 3 tasks: "Setup project", "Write code", "Write tests". Make them depend on each other in order.`
2. `List all tasks and show the dependency graph`
3. `Complete task 1 and then list tasks to see task 2 unblocked`
4. `Create a task board for refactoring: parse -> transform -> emit -> test`
