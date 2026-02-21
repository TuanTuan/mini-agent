# s11: Autonomous Agents (自治智能体)

> 带任务看板轮询的空闲循环让队友能自己发现和认领工作, 上下文压缩后通过身份重注入保持角色认知。

## 问题

在 s09-s10 中, 队友只在被明确指示时才工作。领导必须用特定的 prompt 生成每个队友。如果任务看板上有 10 个未认领的任务, 领导必须手动分配每一个。这无法扩展。

真正的自治意味着队友自己寻找工作。当一个队友完成当前任务后, 它应该扫描任务看板寻找未认领的工作, 认领一个任务, 然后开始工作 -- 不需要领导的任何指令。

但自治智能体面临一个微妙问题: 上下文压缩后, 智能体可能忘记自己是谁。如果消息被摘要化, 原始系统提示中的身份 ("你是 alice, 角色: coder") 就会淡化。身份重注入通过在压缩后的上下文开头插入身份块来解决这个问题。

教学简化说明: 这里的 token 估算比较粗糙 (字符数 / 4)。生产系统使用专业的 tokenizer 库。s03 中的 nag 阈值 3 轮是为教学可见性设的低值; 生产环境的 Claude Code 使用约 10 轮的阈值。

## 解决方案

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

## 工作原理

1. 队友循环有两个阶段: WORK 和 IDLE。WORK 阶段运行标准的 agent loop。当 LLM 停止调用工具 (或调用了 `idle` 工具) 时, 队友进入 IDLE 阶段。

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

2. 空闲阶段循环轮询收件箱和任务看板。

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

3. 任务看板扫描查找 pending 状态、无 owner、未被阻塞的任务。

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

4. 身份重注入: 当上下文过短时插入身份块, 表明发生了压缩。

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

5. `idle` 工具让队友显式地表示没有更多工作, 提前进入空闲轮询阶段。

```python
{"name": "idle",
 "description": "Signal that you have no more work. "
                "Enters idle polling phase.",
 "input_schema": {"type": "object", "properties": {}}},
```

## 核心代码

自治循环 (来自 `agents/s11_autonomous.py`):

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

## 相对 s10 的变更

| 组件           | 之前 (s10)       | 之后 (s11)                       |
|----------------|------------------|----------------------------------|
| Tools          | 12               | 14 (+idle, +claim_task)          |
| 自治性         | 领导指派         | 自组织                           |
| 空闲阶段       | 无               | 轮询收件箱 + 任务看板           |
| 任务认领       | 仅手动           | 自动认领未认领任务               |
| 身份           | 系统提示         | + 压缩后重注入                   |
| 超时           | 无               | 60 秒空闲 -> 自动关机            |

## 生产环境参考

Claude Code 以相同的 idle/poll/claim 循环实现自治队友。生产版本使用 TaskUpdate 工具进行状态变更, 并有空闲通知系统。队友自动从任务看板发现工作并认领, 无需领导干预。身份重注入确保队友在压缩后保持其角色和上下文。生产环境的空闲超时可按队友配置。核心洞察: 领导创建任务看板, 队友自组织完成它。

## 试一试

```sh
cd learn-claude-code
python agents/s11_autonomous.py
```

可以尝试的提示:

1. `Create 3 tasks on the board, then spawn alice and bob. Watch them auto-claim.`
2. `Spawn a coder teammate and let it find work from the task board itself`
3. `Create tasks with dependencies. Watch teammates respect the blocked order.`
4. 输入 `/tasks` 查看带 owner 的任务看板
5. 输入 `/team` 监控谁在工作、谁在空闲
