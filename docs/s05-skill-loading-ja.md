# s05: Skills

> 2層のスキル注入により、スキル名をシステムプロンプトに(低コスト)、スキル本体をtool_resultに(オンデマンド)配置することで、システムプロンプトの肥大化を回避する。

## 問題

エージェントに特定のドメインのワークフローを遵守させたい: gitの規約、テストパターン、コードレビューのチェックリストなど。単純なアプローチはすべてをシステムプロンプトに入れることだ。しかしシステムプロンプトの実効的な注意力は有限であり、テキストが多すぎるとモデルはその一部を無視し始める。

10個のスキルが各2000トークンあれば、20,000トークンのシステムプロンプトになる。モデルは先頭と末尾に注意を払い、中間部分は飛ばし読みする。さらに悪いことに、ほとんどのスキルは任意のタスクに対して無関係だ。ファイル編集のタスクにgitワークフローの指示は不要だ。

2層アプローチがこれを解決する: 第1層はシステムプロンプトにスキルの短い説明を置く(スキルあたり約100トークン)。第2層はモデルが`load_skill`を呼び出した時だけ、スキル本体の全文をtool_resultに読み込む。モデルはどのスキルが存在するかを知り(低コスト)、必要な時だけ読み込む(関連する時のみ)。

## 解決策

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

## 仕組み

1. スキルファイルは`.skills/`にYAMLフロントマター付きMarkdownとして配置される。

```
.skills/
  git.md       # ---\n description: Git workflow\n ---\n ...
  test.md      # ---\n description: Testing patterns\n ---\n ...
```

2. SkillLoaderがフロントマターを解析し、メタデータと本体を分離する。

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

3. 第1層: `get_descriptions()`がシステムプロンプト用の短い行を返す。

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

4. 第2層: `get_content()`が`<skill>`タグで囲まれた本体全文を返す。

```python
def get_content(self, name: str) -> str:
    skill = self.skills.get(name)
    if not skill:
        return f"Error: Unknown skill '{name}'."
    return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
```

5. `load_skill`ツールはディスパッチマップの単なる一エントリだ。

```python
TOOL_HANDLERS = {
    # ...base tools...
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}
```

## 主要コード

SkillLoaderクラス(`agents/s05_knowledge_loading.py` 51-97行目):

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

## s04からの変更点

| Component      | Before (s04)     | After (s05)                |
|----------------|------------------|----------------------------|
| Tools          | 5 (base + task)  | 5 (base + load_skill)      |
| System prompt  | Static string    | + skill descriptions       |
| Knowledge      | None             | .skills/*.md files         |
| Injection      | None             | Two-layer (system + result)|
| Subagent       | `run_subagent()` | Removed (different focus)  |

## 本番環境との対応

Claude CodeではこれをSkillツールとして実装している。スキルはスキルディレクトリ内のSKILL.mdファイルとして保存される。システムプロンプトには利用可能なスキルのコンパクトなリストが含まれる。モデルが`Skill("skill-name")`を呼び出すと、SKILL.mdの全内容がtool_resultに注入される。このパターンはCLAUDE.mdプロジェクト指示にも使われている -- システムプロンプトを肥大化させるのではなく、オンデマンドでコンテキストに読み込まれる。2層アプローチにより、モデルの注意力を劣化させることなく数十のスキルにスケールできる。

## 試してみる

```sh
cd learn-claude-code
python agents/s05_knowledge_loading.py
```

試せるプロンプト例:

1. `What skills are available?`
2. `Load the agent-builder skill and follow its instructions`
3. `I need to do a code review -- load the relevant skill first`
4. `Build an MCP server using the mcp-builder skill`
