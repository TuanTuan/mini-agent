# s08: Background Tasks (后台任务)

> BackgroundManager 在独立线程中运行命令, 在每次 LLM 调用前排空通知队列, 使智能体永远不会因长时间运行的操作而阻塞。

## 问题

有些命令需要几分钟: `npm install`、`pytest`、`docker build`。在阻塞式的 agent loop 中, 模型只能干等子进程结束, 什么也做不了。如果用户要求 "安装依赖, 同时创建配置文件", 智能体会先安装, 然后才创建配置 -- 串行执行, 而非并行。

智能体需要并发能力。不是将 agent loop 本身完全多线程化, 而是能够发起一个长时间命令然后继续工作。当命令完成时, 结果自然地出现在对话中。

解决方案是一个 BackgroundManager, 它在守护线程中运行命令, 将结果收集到通知队列中。每次 LLM 调用前, 队列被排空, 结果注入到消息中。

## 解决方案

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

## 工作原理

1. BackgroundManager 追踪任务并维护一个线程安全的通知队列。

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()
```

2. `run()` 启动一个守护线程并立即返回 task_id。

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

3. 线程目标函数 `_execute` 运行子进程并将结果推入通知队列。

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

4. `drain_notifications()` 返回并清空待处理的结果。

```python
def drain_notifications(self) -> list:
    with self._lock:
        notifs = list(self._notification_queue)
        self._notification_queue.clear()
    return notifs
```

5. Agent loop 在每次 LLM 调用前排空通知。

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

## 核心代码

BackgroundManager (来自 `agents/s08_background.py`, 第 49-107 行):

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

## 相对 s07 的变更

| 组件           | 之前 (s07)       | 之后 (s08)                         |
|----------------|------------------|------------------------------------|
| Tools          | 8                | 6 (基础 + background_run + check)  |
| 执行方式       | 仅阻塞           | 阻塞 + 后台线程                    |
| 通知机制       | 无               | 每轮排空的队列                     |
| 并发           | 无               | 守护线程                           |
| 任务系统       | 基于文件的 CRUD  | 已移除 (非本节重点)                |

## 生产环境参考

Claude Code 以独立进程运行后台任务。测试套件或构建等长时间运行的命令被异步启动, 其结果通过通知系统传递给智能体。生产版本增加了进程组清理、进度报告和取消支持。"在 LLM 调用前排空" 的模式确保后台结果在对话的自然间断点到达, 而不是打断思考过程。

## 试一试

```sh
cd learn-claude-code
python agents/s08_background.py
```

可以尝试的提示:

1. `Run "sleep 5 && echo done" in the background, then create a file while it runs`
2. `Start 3 background tasks: "sleep 2", "sleep 4", "sleep 6". Check their status.`
3. `Run pytest in the background and keep working on other things`
