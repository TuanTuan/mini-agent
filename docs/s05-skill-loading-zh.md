# s05: Skills (技能加载)

> 两层技能注入避免了系统提示膨胀: 在系统提示中放技能名称 (低成本), 在 tool_result 中按需放入完整技能内容。

## 问题

你希望智能体针对不同领域遵循特定的工作流: git 约定、测试模式、代码审查清单。简单粗暴的做法是把所有内容都塞进系统提示。但系统提示的有效注意力是有限的 -- 文本太多, 模型就会开始忽略其中一部分。

如果你有 10 个技能, 每个 2000 token, 那就是 20,000 token 的系统提示。模型关注开头和结尾, 但会略过中间部分。更糟糕的是, 这些技能中大部分与当前任务无关。文件编辑任务不需要 git 工作流说明。

两层方案解决了这个问题: 第一层在系统提示中放入简短的技能描述 (每个技能约 100 token)。第二层只在模型调用 `load_skill` 时, 才将完整的技能内容加载到 tool_result 中。模型知道有哪些技能可用 (低成本), 按需加载它们 (只在相关时)。

## 解决方案

```
System prompt (Layer 1 -- always present):
+--------------------------------------+
| You are a coding agent.              |
| Skills available:                    |
|   - git: Git workflow helpers        |  ~100 tokens/skill
|   - test: Testing best practices     |
+--------------------------------------+

When model calls load_skill("git"):
+--------------------------------------+
| tool_result (Layer 2 -- on demand):  |
| <skill name="git">                   |
|   Full git workflow instructions...  |  ~2000 tokens
|   Step 1: ...                        |
|   Step 2: ...                        |
| </skill>                             |
+--------------------------------------+
```

## 工作原理

1. 技能文件以 Markdown 格式存放在 `.skills/` 目录中, 带有 YAML frontmatter。

```
.skills/
  git.md       # ---\n description: Git workflow\n ---\n ...
  test.md      # ---\n description: Testing patterns\n ---\n ...
```

2. SkillLoader 解析 frontmatter, 分离元数据和正文。

```python
class SkillLoader:
    def _parse_frontmatter(self, text: str) -> tuple:
        match = re.match(
            r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL
        )
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()
```

3. 第一层: `get_descriptions()` 返回简短描述, 用于系统提示。

```python
def get_descriptions(self) -> str:
    lines = []
    for name, skill in self.skills.items():
        desc = skill["meta"].get("description", "No description")
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Skills available:
{SKILL_LOADER.get_descriptions()}"""
```

4. 第二层: `get_content()` 返回用 `<skill>` 标签包裹的完整正文。

```python
def get_content(self, name: str) -> str:
    skill = self.skills.get(name)
    if not skill:
        return f"Error: Unknown skill '{name}'."
    return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
```

5. `load_skill` 工具只是 dispatch map 中的又一个条目。

```python
TOOL_HANDLERS = {
    # ...base tools...
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}
```

## 核心代码

SkillLoader 类 (来自 `agents/s05_knowledge_loading.py`, 第 51-97 行):

```python
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        for f in sorted(skills_dir.glob("*.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            self.skills[f.stem] = {
                "meta": meta, "body": body
            }

    def get_descriptions(self) -> str:
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'."
        return (f"<skill name=\"{name}\">\n"
                f"{skill['body']}\n</skill>")
```

## 相对 s04 的变更

| 组件           | 之前 (s04)       | 之后 (s05)                     |
|----------------|------------------|----------------------------|
| Tools          | 5 (基础 + task)  | 5 (基础 + load_skill)      |
| 系统提示       | 静态字符串       | + 技能描述列表             |
| 知识库         | 无               | .skills/*.md 文件          |
| 注入方式       | 无               | 两层 (系统提示 + result)   |
| Subagent       | `run_subagent()` | 已移除 (非本节重点)        |

## 生产环境参考

Claude Code 通过 Skill 工具实现了这一机制。技能以 SKILL.md 文件存储在技能目录中。系统提示包含一个紧凑的可用技能列表。当模型调用 `Skill("skill-name")` 时, 完整的 SKILL.md 内容被注入到 tool_result 中。这种模式也用于 CLAUDE.md 项目指令 -- 它们按需加载到上下文中, 而不是让系统提示膨胀。两层方案可以扩展到数十个技能而不会降低模型的注意力质量。

## 试一试

```sh
cd learn-claude-code
python agents/s05_knowledge_loading.py
```

可以尝试的提示:

1. `What skills are available?`
2. `Load the agent-builder skill and follow its instructions`
3. `I need to do a code review -- load the relevant skill first`
4. `Build an MCP server using the mcp-builder skill`
