# s04: Subagent (子智能体)

> 子智能体使用全新的消息列表运行, 与父智能体共享文件系统, 仅返回摘要 -- 保持父上下文的整洁。

## 问题

随着智能体工作, 它的消息数组不断增长。每次工具调用、每次文件读取、每次 bash 输出都在累积。20-30 次工具调用后, 上下文窗口充满了无关的历史。为了回答一个简单问题而读取的 500 行文件, 会永久占据上下文中的 500 行空间。

这对探索性任务尤其糟糕。"这个项目用了什么测试框架?" 可能需要读取 5 个文件, 但父智能体的历史中并不需要这 5 个文件的全部内容 -- 它只需要答案: "pytest, 使用 conftest.py 配置。"

解决方案是进程隔离: 以 `messages=[]` 启动一个子智能体。子智能体进行探索、读取文件、运行命令。完成后, 只有最终的文本响应返回给父智能体。子智能体的全部消息历史被丢弃。

## 解决方案

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ---------->| while tool_use:  |
|   prompt="..."   |            |   call tools     |
|                  |  summary   |   append results |
|   result = "..." | <--------- | return last text |
+------------------+             +------------------+
          |
Parent context stays clean.
Subagent context is discarded.
```

## 工作原理

1. 父智能体拥有一个 `task` 工具用于触发子智能体的生成。子智能体获得除 `task` 外的所有基础工具 (不允许递归生成)。

```python
PARENT_TOOLS = CHILD_TOOLS + [
    {"name": "task",
     "description": "Spawn a subagent with fresh context.",
     "input_schema": {
         "type": "object",
         "properties": {
             "prompt": {"type": "string"},
             "description": {"type": "string"},
         },
         "required": ["prompt"],
     }},
]
```

2. 子智能体以全新的消息列表启动, 仅包含委派的 prompt。它共享相同的文件系统。

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({
            "role": "assistant", "content": response.content
        })
        if response.stop_reason != "tool_use":
            break
        # execute tools, append results...
```

3. 只有最终文本返回给父智能体。子智能体 30+ 次工具调用的历史被丢弃。

```python
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

4. 父智能体将此摘要作为普通的 tool_result 接收。

```python
if block.name == "task":
    output = run_subagent(block.input["prompt"])
results.append({
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": str(output),
})
```

## 核心代码

子智能体函数 (来自 `agents/s04_context_isolation.py`, 第 110-128 行):

```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant",
                             "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input)
                results.append({"type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"
```

## 相对 s03 的变更

| 组件           | 之前 (s03)       | 之后 (s04)                    |
|----------------|------------------|---------------------------|
| Tools          | 5                | 5 (基础) + task (仅父端)  |
| 上下文         | 单一共享         | 父 + 子隔离               |
| Subagent       | 无               | `run_subagent()` 函数     |
| 返回值         | 不适用           | 仅摘要文本                |
| Todo 系统      | TodoManager      | 已移除 (非本节重点)       |

## 生产环境参考

Claude Code 通过 Task 工具和 `spawn()` 实现了这一机制。当模型调用 Task 时, 会创建一个 `messages=[]` 的子进程。子进程继承文件系统和工具定义, 但拥有自己的上下文窗口。Claude Code 还支持子智能体描述, 显示在父端的 UI 中, 让用户能看到子智能体在做什么而不会污染父上下文。生产版本还限制了子智能体深度 (不允许从 Task 中递归调用 Task) 并强制执行最大迭代次数限制。

## 试一试

```sh
cd learn-claude-code
python agents/s04_context_isolation.py
```

可以尝试的提示:

1. `Use a subtask to find what testing framework this project uses`
2. `Delegate: read all .py files and summarize what each one does`
3. `Use a task to create a new module, then verify it from here`
