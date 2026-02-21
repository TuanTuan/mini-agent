# s01: Agent Loop (智能体循环)

> AI 编程智能体的全部秘密就是一个 while 循环 -- 把工具执行结果反馈给模型, 直到模型决定停止。

## 问题

为什么语言模型不能直接回答编程问题? 因为编程需要**与真实世界交互**。模型需要读取文件、运行测试、检查错误、反复迭代。单次的提示-响应交互无法做到这些。

没有 agent loop, 你就得手动把输出复制粘贴回模型。用户自己变成了那个循环。Agent loop 将这个过程自动化: 调用模型, 执行它要求的工具, 把结果送回去, 重复 -- 直到模型说 "我完成了"。

考虑一个简单任务: "创建一个打印 hello 的 Python 文件。" 模型需要 (1) 决定写文件, (2) 写入文件, (3) 验证是否正常工作。至少三次工具调用。没有循环的话, 每一次都需要人工干预。

## 解决方案

```
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> |  Tool   |
|  prompt  |      |       |      | execute |
+----------+      +---+---+      +----+----+
                      ^               |
                      |   tool_result |
                      +---------------+
                      (loop continues)

The loop terminates when stop_reason != "tool_use".
That single condition is the entire control flow.
```

## 工作原理

1. 用户提供一个 prompt, 成为第一条消息。

```python
history.append({"role": "user", "content": query})
```

2. 消息数组连同工具定义一起发送给 LLM。

```python
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
```

3. 助手的响应被追加到消息列表中。

```python
messages.append({"role": "assistant", "content": response.content})
```

4. 检查 stop_reason。如果模型没有调用工具, 循环结束。这是唯一的退出条件。

```python
if response.stop_reason != "tool_use":
    return
```

5. 对响应中的每个 tool_use 块, 执行工具 (本节课中是 bash) 并收集结果。

```python
for block in response.content:
    if block.type == "tool_use":
        output = run_bash(block.input["command"])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

6. 结果作为 user 消息追加, 循环继续。

```python
messages.append({"role": "user", "content": results})
```

## 核心代码

最小可行智能体 -- 不到 30 行代码实现整个模式
(来自 `agents/s01_agent_loop.py`, 第 66-86 行):

```python
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

## 变更内容

这是第 1 节课 -- 起点。没有前置课程。

| 组件          | 之前       | 之后                           |
|---------------|------------|--------------------------------|
| Agent loop    | (无)       | `while True` + stop_reason     |
| Tools         | (无)       | `bash` (单一工具)              |
| Messages      | (无)       | 累积式消息列表                 |
| Control flow  | (无)       | `stop_reason != "tool_use"`    |

## 生产环境参考

在真实的 Claude Code 中, 同样的循环运行在主 REPL 内。核心循环逻辑位于 `agent.ts`, 遵循完全相同的模式: 发送消息到 API, 检查 `stop_reason`, 执行工具, 追加结果。生产版本增加了错误处理、token 计数、流式输出和重试逻辑, 但基本结构没有变化。每一个 Claude Code 会话本质上都是这个 while 循环。

## 试一试

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

可以尝试的提示:

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
