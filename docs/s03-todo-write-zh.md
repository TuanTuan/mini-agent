# s03: TodoWrite (待办写入)

> TodoManager 让智能体能追踪自己的进度, 而 nag reminder 注入机制在它忘记更新时强制提醒。

## 问题

当智能体处理多步骤任务时, 它经常丢失对已完成和待办事项的追踪。没有显式的计划, 模型可能重复工作、跳过步骤或跑偏。用户也无法看到智能体内部的计划。

这个问题比听起来更严重。长对话会导致模型 "漂移" -- 随着上下文窗口被工具结果填满, 系统提示的影响力逐渐减弱。一个 10 步的重构任务可能完成了 1-3 步, 然后模型就开始即兴发挥, 因为它忘了第 4-10 步的存在。

解决方案是结构化状态: 一个模型显式写入的 TodoManager。模型创建计划, 工作时将项目标记为 in_progress, 完成后标记为 completed。nag reminder 机制在模型连续 3 轮以上不更新待办时注入提醒。

教学简化说明: 这里 nag 阈值设为 3 轮是为了教学可见性。生产环境的 Claude Code 使用约 10 轮的阈值以避免过度提醒。

## 解决方案

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

## 工作原理

1. TodoManager 验证并存储一组带状态的项目。同一时间只允许一个项目处于 `in_progress` 状态。

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

2. `todo` 工具和其他工具一样添加到 dispatch map 中。

```python
TOOL_HANDLERS = {
    "bash":  lambda **kw: run_bash(kw["command"]),
    # ...other tools...
    "todo":  lambda **kw: TODO.update(kw["items"]),
}
```

3. nag reminder 在模型连续 3 轮以上不调用 `todo` 时, 向 tool_result 消息中注入 `<reminder>` 标签。

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

4. 系统提示指导模型使用 todo 进行规划。

```python
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks.
Mark in_progress before starting, completed when done.
Prefer tools over prose."""
```

## 核心代码

TodoManager 和 nag 注入 (来自 `agents/s03_structured_planning.py`,
第 51-85 行和第 158-187 行):

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

## 相对 s02 的变更

| 组件           | 之前 (s02)       | 之后 (s03)                   |
|----------------|------------------|--------------------------|
| Tools          | 4                | 5 (+todo)                |
| 规划           | 无               | 带状态的 TodoManager     |
| Nag 注入       | 无               | 3 轮后注入 `<reminder>`  |
| Agent loop     | 简单分发         | + rounds_since_todo 计数器|

## 生产环境参考

Claude Code 通过 TodoWrite 和 TodoRead 工具实现了同样的模式。模型创建带状态 (pending, in_progress, completed) 的结构化待办列表。nag reminder 概念在生产中表现为 "系统提醒" (system reminders) -- 当模型偏离方向时注入的引导消息。生产版 TodoManager 支持优先级和跨会话持久化。"同一时间只允许一个 in_progress" 的约束强制顺序聚焦, 防止模型过快地切换上下文。

## 试一试

```sh
cd learn-claude-code
python agents/s03_structured_planning.py
```

可以尝试的提示:

1. `Refactor the file hello.py: add type hints, docstrings, and a main guard`
2. `Create a Python package with __init__.py, utils.py, and tests/test_utils.py`
3. `Review all Python files and fix any style issues`
