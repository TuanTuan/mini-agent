# s08: Background Tasks

> BackgroundManagerがコマンドを別スレッドで実行し、各LLM呼び出しの前に通知キューをドレインすることで、エージェントは長時間実行操作でブロックされなくなる。

## 問題

一部のコマンドは数分かかる: `npm install`、`pytest`、`docker build`。ブロッキングのagent loopでは、モデルはサブプロセスの終了を待って待機する。他のことは何もできない。ユーザーが「依存関係をインストールして、その間にconfigファイルを作成して」と言った場合、エージェントはまずインストールを行い、その後configを作成する -- 並列ではなく逐次的に。

エージェントには並行性が必要だ。agent loop自体の完全なマルチスレッディングではなく、長いコマンドを発射して実行中に作業を続ける能力だ。コマンドが終了したら、その結果は自然に会話に現れるべきだ。

解決策は、BackgroundManagerがコマンドをデーモンスレッドで実行し、結果を通知キューに収集すること。各LLM呼び出しの前にキューがドレインされ、結果がメッセージに注入される。

## 解決策

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

## 仕組み

1. BackgroundManagerがタスクを追跡し、スレッドセーフな通知キューを維持する。

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()
```

2. `run()`がデーモンスレッドを開始し、task_idを即座に返す。

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

3. スレッドのターゲットである`_execute`がサブプロセスを実行し、結果を通知キューにプッシュする。

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

4. `drain_notifications()`が保留中の結果を返してクリアする。

```python
def drain_notifications(self) -> list:
    with self._lock:
        notifs = list(self._notification_queue)
        self._notification_queue.clear()
    return notifs
```

5. agent loopが各LLM呼び出しの前に通知をドレインする。

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

## 主要コード

BackgroundManager(`agents/s08_background.py` 49-107行目):

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

## s07からの変更点

| Component      | Before (s07)     | After (s08)                |
|----------------|------------------|----------------------------|
| Tools          | 8                | 6 (base + background_run + check)|
| Execution      | Blocking only    | Blocking + background threads|
| Notification   | None             | Queue drained per loop     |
| Concurrency    | None             | Daemon threads             |
| Task system    | File-based CRUD  | Removed (different focus)  |

## 本番環境との対応

Claude Codeではバックグラウンドタスクを別プロセスとして実行する。テストスイートやビルドなどの長時間実行コマンドは非同期で起動され、結果は通知システムを通じてエージェントに配信される。本番バージョンではクリーンアップのためのプロセスグループ、進捗報告、キャンセルサポートが追加されている。LLM呼び出し前のドレインパターンにより、バックグラウンド結果は思考の途中で割り込むのではなく、会話の自然な切れ目で到着する。

## 試してみる

```sh
cd learn-claude-code
python agents/s08_background.py
```

試せるプロンプト例:

1. `Run "sleep 5 && echo done" in the background, then create a file while it runs`
2. `Start 3 background tasks: "sleep 2", "sleep 4", "sleep 6". Check their status.`
3. `Run pytest in the background and keep working on other things`
