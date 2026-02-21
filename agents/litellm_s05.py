#!/usr/bin/env python3
"""
litellm_s05.py - Skills (LiteLLM/OpenAI Format)

基于 s05_skill_loading.py，使用 LiteLLM SDK 和 OpenAI 消息格式。
Two-layer skill injection that avoids bloating the system prompt:

    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result

环境变量:
    AZURE_API_KEY      - Azure API 密钥
    AZURE_API_BASE     - Azure 端点 URL
    AZURE_API_VERSION  - API 版本
    AZURE_DEPLOYMENT   - 部署名称 (默认 gpt-5.2)

命令行参数:
    python litellm_s05.py -o session.md   # 输出到日志文件
"""

import json
import os
import re
import subprocess
from pathlib import Path

import litellm
from dotenv import load_dotenv

from logger_openai import create_logger_from_args, parse_logger_args, get_logger_config_string

load_dotenv(override=True)

# ============================================================================
# 配置
# ============================================================================
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5.2")

WORKDIR = Path.cwd()
MODEL = f"azure/{AZURE_DEPLOYMENT}"
SKILLS_DIR = WORKDIR / ".skills"

# 解析命令行参数并初始化日志器
_args = parse_logger_args()
logger = create_logger_from_args(_args)


# -- SkillLoader: parse .skills/*.md files with YAML frontmatter --
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.glob("*.md")):
            name = f.stem
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for the system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"

    def print_loaded_skills(self):
        """打印已加载的技能列表（启动时显示）"""
        if not self.skills:
            print(logger._color("  (no skills found)", "dim"))
            return

        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            path = skill["path"]
            print(f"  {logger._color('📚', 'yellow')} {logger._color(name, 'cyan')}: {desc}")
            if tags:
                print(f"      Tags: {logger._color(tags, 'dim')}")
            print(f"      Path: {logger._color(path, 'dim')}")

    def print_skill_loaded(self, name: str, content: str):
        """打印技能加载详情（调用 load_skill 时显示）"""
        skill = self.skills.get(name)
        if skill:
            print(logger._color(f"\n{'┌' + '─' * 78 + '┐'}", "yellow"))
            print(logger._color(f"│  📚 SKILL LOADED: {name}" + " " * (60 - len(name)) + "│", "yellow"))
            desc = skill["meta"].get("description", "")
            if desc:
                print(logger._color(f"│  Description: {desc[:60]}" + " " * (67 - min(len(desc), 60)) + "│", "yellow"))
            body_lines = skill["body"].split("\n")
            preview = "\n".join(body_lines[:5])
            if len(body_lines) > 5:
                preview += f"\n... ({len(body_lines) - 5} more lines)"
            print(logger._color(f"│" + " " * 78 + "│", "yellow"))
            for line in preview.split("\n")[:7]:
                truncated = line[:74] if len(line) > 74 else line
                print(logger._color(f"│  {truncated}" + " " * (76 - len(truncated)) + "│", "dim"))
            print(logger._color(f"└" + "─" * 78 + "┘", "yellow"))


SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1: skill metadata injected into system prompt
SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

# -- OpenAI 格式的工具定义 --
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}}},
]


def agent_loop(messages: list):
    """Agent 循环"""
    iteration = 0

    while True:
        iteration += 1
        logger.loop_iteration(iteration)
        logger.messages_snapshot(messages, "BEFORE LLM CALL")

        logger.request_raw(model=MODEL, messages=messages, tools=TOOLS, max_tokens=8000)

        response = litellm.completion(
            model=MODEL, messages=messages, tools=TOOLS,
            api_key=AZURE_API_KEY, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION,
        )

        response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
        logger.response_raw(response_dict)

        choice = (response_dict.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        tool_calls = message.get("tool_calls") or []
        usage = response_dict.get("usage") or {}

        logger.llm_response_summary(
            finish_reason,
            {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
            len(tool_calls)
        )

        assistant_msg = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        logger.messages_snapshot(messages, "AFTER APPEND ASSISTANT")

        if finish_reason != "tool_calls":
            logger.loop_end(f"finish_reason = '{finish_reason}'")
            return

        logger.section("Executing Tool Calls", "🔧")
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args_str = fn.get("arguments", "{}")

            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except json.JSONDecodeError:
                fn_args = {}

            logger.tool_call(fn_name, fn_args, tc_id)

            handler = TOOL_HANDLERS.get(fn_name)
            try:
                output = handler(**fn_args) if handler else f"Unknown tool: {fn_name}"
            except Exception as e:
                output = f"Error: {e}"

            # 特殊处理 load_skill
            if fn_name == "load_skill":
                skill_name = fn_args.get("name", "")
                SKILL_LOADER.print_skill_loaded(skill_name, output)

            is_error = str(output).startswith("Error:")
            logger.tool_result(tc_id, str(output), is_error=is_error)
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)})

        logger.messages_snapshot(messages, "AFTER APPEND TOOL RESULTS")
        logger.separator(f"END OF ITERATION {iteration}")


if __name__ == "__main__":
    logger.header("LiteLLM Skill Loading - Azure GPT-5.2", "litellm-s05")
    logger.config(model=MODEL, api_base=AZURE_API_BASE, api_version=AZURE_API_VERSION)

    print(logger._color(f"\n  ⚙️ Logger Config: {get_logger_config_string(_args)}", "dim"))
    if _args.log_file:
        print(logger._color(f"  📁 Log file: {_args.log_file}", "dim"))
    print()

    logger.section("Available Skills", "📚")
    SKILL_LOADER.print_loaded_skills()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            query = input("\033[36mlitellm-s05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        logger.user_input(query)
        history.append({"role": "user", "content": query})
        agent_loop(history)

        logger.separator("FINAL RESPONSE")
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    print(content)
                break
        print()

    logger.session_end("用户退出")
