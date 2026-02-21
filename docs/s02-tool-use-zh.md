# s02: Tools (工具)

> 一个分发映射表 (dispatch map) 将工具调用路由到处理函数 -- 循环本身完全不需要改动。

## 问题

只有 `bash` 时, 智能体所有操作都通过 shell: 读文件、写文件、编辑文件。这能用但很脆弱。`cat` 的输出会被不可预测地截断。`sed` 替换遇到特殊字符就会失败。模型浪费大量 token 构造 shell 管道, 而一个直接的函数调用会简单得多。

更重要的是, bash 是一个安全攻击面。每次 bash 调用都能做 shell 能做的一切。有了专用工具如 `read_file` 和 `write_file`, 你可以在工具层面强制路径沙箱化, 阻止危险模式, 而不是寄希望于模型自觉回避。

关键洞察: 添加工具不需要修改循环。s01 的循环保持不变。你只需在工具数组中添加条目, 编写处理函数, 然后通过 dispatch map 把它们关联起来。

## 解决方案

```
+----------+      +-------+      +------------------+
|   User   | ---> |  LLM  | ---> | Tool Dispatch    |
|  prompt  |      |       |      | {                |
+----------+      +---+---+      |   bash: run_bash |
                      ^          |   read: run_read |
                      |          |   write: run_wr  |
                      +----------+   edit: run_edit |
                      tool_result| }                |
                                 +------------------+

The dispatch map is a dict: {tool_name: handler_function}
One lookup replaces any if/elif chain.
```

## 工作原理

1. 为每个工具定义处理函数。每个函数接受与工具 input_schema 对应的关键字参数, 返回字符串结果。

```python
def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]
```

2. 创建 dispatch map, 将工具名映射到处理函数。

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}
```

3. 在 agent loop 中, 按名称查找处理函数, 而不是硬编码。

```python
for block in response.content:
    if block.type == "tool_use":
        handler = TOOL_HANDLERS.get(block.name)
        output = handler(**block.input)
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
```

4. 路径沙箱化防止模型逃逸出工作区。

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

## 核心代码

dispatch 模式 (来自 `agents/s02_multi_tool.py`, 第 93-129 行):

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
}

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
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler \
                    else f"Unknown tool: {block.name}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

## 相对 s01 的变更

| 组件           | 之前 (s01)         | 之后 (s02)                     |
|----------------|--------------------|----------------------------|
| Tools          | 1 (仅 bash)        | 4 (bash, read, write, edit)|
| Dispatch       | 硬编码 bash 调用   | `TOOL_HANDLERS` 字典       |
| 路径安全       | 无                 | `safe_path()` 沙箱         |
| Agent loop     | 不变               | 不变                       |

## 生产环境参考

Claude Code 使用相同的 dispatch map 模式。工具注册在工具注册表中, 主循环按名称查找处理函数。生产工具集要大得多 (Bash, Read, Write, Edit, Glob, Grep, WebFetch 等), 但每个工具遵循相同的契约: 接受符合 schema 的输入, 返回字符串结果。给 Claude Code 添加新工具意味着添加一个处理函数和一个 schema 条目 -- 循环不需要改动。

## 试一试

```sh
cd learn-claude-code
python agents/s02_multi_tool.py
```

可以尝试的提示:

1. `Read the file requirements.txt`
2. `Create a file called greet.py with a greet(name) function`
3. `Edit greet.py to add a docstring to the function`
4. `Read greet.py to verify the edit worked`
5. `Run the greet function with bash: python -c "from greet import greet; greet('World')"`
