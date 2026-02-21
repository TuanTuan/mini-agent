# s06: Compact (上下文压缩)

> 三层压缩管道让智能体可以无限期工作: 策略性地遗忘旧的工具结果, token 超过阈值时自动摘要, 以及支持手动触发压缩。

## 问题

上下文窗口是有限的。工具调用积累到足够多时, 消息数组会超过模型的上下文限制, API 调用直接失败。即使在到达硬限制之前, 性能也会下降: 模型变慢、准确率降低, 开始忽略早期消息。

200,000 token 的上下文窗口听起来很大, 但一次 `read_file` 读取 1000 行源文件就消耗约 4000 token。读取 30 个文件、运行 20 条 bash 命令后, 你就已经用掉 100,000+ token 了。没有某种压缩机制, 智能体无法在大型代码库上工作。

三层管道以递增的激进程度来应对这个问题:
第一层 (micro-compact) 每轮静默替换旧的工具结果。
第二层 (auto-compact) 在 token 超过阈值时触发完整摘要。
第三层 (manual compact) 让模型自己触发压缩。

教学简化说明: 这里的 token 估算使用粗略的 字符数/4 启发式方法。生产系统使用专业的 tokenizer 库进行精确计数。

## 解决方案

```
Every turn:
+------------------+
| Tool call result |
+------------------+
        |
        v
[Layer 1: micro_compact]        (silent, every turn)
  Replace tool_result > 3 turns old
  with "[Previous: used {tool_name}]"
        |
        v
[Check: tokens > 50000?]
   |               |
   no              yes
   |               |
   v               v
continue    [Layer 2: auto_compact]
              Save transcript to .transcripts/
              LLM summarizes conversation.
              Replace all messages with [summary].
                    |
                    v
            [Layer 3: compact tool]
              Model calls compact explicitly.
              Same summarization as auto_compact.
```

## 工作原理

1. **第一层 -- micro_compact**: 每次 LLM 调用前, 找到最近 3 条之前的所有 tool_result 条目, 替换其内容。

```python
def micro_compact(messages: list) -> list:
    tool_results = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for j, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((i, j, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, part in to_clear:
        if len(part.get("content", "")) > 100:
            tool_id = part.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            part["content"] = f"[Previous: used {tool_name}]"
    return messages
```

2. **第二层 -- auto_compact**: 当估算 token 超过 50,000 时, 保存完整对话记录并请求 LLM 进行摘要。

```python
def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity..."
            + json.dumps(messages, default=str)[:80000]}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    return [
        {"role": "user", "content": f"[Compressed]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. Continuing."},
    ]
```

3. **第三层 -- manual compact**: `compact` 工具按需触发相同的摘要机制。

```python
if manual_compact:
    messages[:] = auto_compact(messages)
```

4. Agent loop 整合了全部三层。

```python
def agent_loop(messages: list):
    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)
        response = client.messages.create(...)
        # ... tool execution ...
        if manual_compact:
            messages[:] = auto_compact(messages)
```

## 核心代码

三层管道 (来自 `agents/s06_compression.py`, 第 67-93 行和第 189-223 行):

```python
THRESHOLD = 50000
KEEP_RECENT = 3

def micro_compact(messages):
    # Replace old tool results with placeholders
    ...

def auto_compact(messages):
    # Save transcript, LLM summarize, replace messages
    ...

def agent_loop(messages):
    while True:
        micro_compact(messages)          # Layer 1
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)  # Layer 2
        response = client.messages.create(...)
        # ...
        if manual_compact:
            messages[:] = auto_compact(messages)  # Layer 3
```

## 相对 s05 的变更

| 组件           | 之前 (s05)       | 之后 (s06)                     |
|----------------|------------------|----------------------------|
| Tools          | 5                | 5 (基础 + compact)         |
| 上下文管理     | 无               | 三层压缩                   |
| Micro-compact  | 无               | 旧结果 -> 占位符           |
| Auto-compact   | 无               | token 阈值触发             |
| Manual compact | 无               | `compact` 工具             |
| Transcripts    | 无               | 保存到 .transcripts/       |
| Skills         | load_skill       | 已移除 (非本节重点)        |

## 生产环境参考

Claude Code 实现了全部三层。Micro-compaction 自动进行 -- 旧的工具结果被截断或替换为摘要。Auto-compaction 在上下文接近模型限制时触发, 摘要对话并重置。用户也可以通过 `/compact` 手动触发压缩。对话记录被保存, 因此完整历史永远不会真正丢失, 只是从活跃上下文中移出。生产版本还会保留 "固定" 消息, 这些消息永远不应被压缩 (比如 CLAUDE.md 指令)。

## 试一试

```sh
cd learn-claude-code
python agents/s06_compression.py
```

可以尝试的提示:

1. `Read every Python file in the agents/ directory one by one`
   (观察 micro-compact 替换旧的结果)
2. `Keep reading files until compression triggers automatically`
3. `Use the compact tool to manually compress the conversation`
