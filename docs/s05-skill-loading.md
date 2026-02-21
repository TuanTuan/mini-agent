# s05: Skills

> Two-layer skill injection avoids system prompt bloat by putting skill names in the system prompt (cheap) and full skill bodies in tool_result (on demand).

## The Problem

You want the agent to follow specific workflows for different domains:
git conventions, testing patterns, code review checklists. The naive
approach is to put everything in the system prompt. But the system prompt
has limited effective attention -- too much text and the model starts
ignoring parts of it.

If you have 10 skills at 2000 tokens each, that is 20,000 tokens of system
prompt. The model pays attention to the beginning and end but skims the
middle. Worse, most of those skills are irrelevant to any given task. A
file editing task does not need the git workflow instructions.

The two-layer approach solves this: Layer 1 puts short skill descriptions
in the system prompt (~100 tokens per skill). Layer 2 loads the full skill
body into a tool_result only when the model calls `load_skill`. The model
learns what skills exist (cheap) and loads them on demand (only when
relevant).

## The Solution

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

## How It Works

1. Skill files live in `.skills/` as Markdown with YAML frontmatter.

```
.skills/
  git.md       # ---\n description: Git workflow\n ---\n ...
  test.md      # ---\n description: Testing patterns\n ---\n ...
```

2. The SkillLoader parses frontmatter and separates metadata from body.

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

3. Layer 1: `get_descriptions()` returns short lines for the system prompt.

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

4. Layer 2: `get_content()` returns the full body wrapped in `<skill>` tags.

```python
def get_content(self, name: str) -> str:
    skill = self.skills.get(name)
    if not skill:
        return f"Error: Unknown skill '{name}'."
    return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
```

5. The `load_skill` tool is just another entry in the dispatch map.

```python
TOOL_HANDLERS = {
    # ...base tools...
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}
```

## Key Code

The SkillLoader class (from `agents/s05_knowledge_loading.py`,
lines 51-97):

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

## What Changed From s04

| Component      | Before (s04)     | After (s05)                |
|----------------|------------------|----------------------------|
| Tools          | 5 (base + task)  | 5 (base + load_skill)      |
| System prompt  | Static string    | + skill descriptions       |
| Knowledge      | None             | .skills/*.md files         |
| Injection      | None             | Two-layer (system + result)|
| Subagent       | `run_subagent()` | Removed (different focus)  |

## Production Reference

Claude Code implements this as the Skill tool. Skills are stored as
SKILL.md files in skill directories. The system prompt includes a compact
list of available skills. When the model calls `Skill("skill-name")`,
the full SKILL.md content is injected into the tool_result. This pattern
is also used for CLAUDE.md project instructions -- they are loaded into
the context on demand rather than bloating the system prompt. The two-layer
approach scales to dozens of skills without degrading model attention.

## Try It

```sh
cd learn-claude-code
python agents/s05_knowledge_loading.py
```

Example prompts to try:

1. `What skills are available?`
2. `Load the agent-builder skill and follow its instructions`
3. `I need to do a code review -- load the relevant skill first`
4. `Build an MCP server using the mcp-builder skill`
