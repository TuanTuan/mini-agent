#!/usr/bin/env python3
"""
logger_openai.py - OpenAI 格式的 Agent 日志系统

专门为 OpenAI 消息格式设计的日志输出器，直接处理 OpenAI 的数据结构。
支持结构化日志、命令行参数、Markdown 文件输出。

OpenAI 消息格式:
- System: {"role": "system", "content": "..."}
- User: {"role": "user", "content": "..."}
- Assistant: {"role": "assistant", "content": "...", "tool_calls": [...]}
- Tool: {"role": "tool", "tool_call_id": "...", "content": "..."}

OpenAI Tool 格式:
{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "...",
        "parameters": {...}
    }
}

使用方法:
    # 基础用法
    from logger_openai import OpenAILogger
    logger = OpenAILogger(verbose=True, show_raw=True)
    logger.request_raw(model, messages, tools)
    logger.response_raw(response_dict)

    # 输出到 Markdown 文件
    logger = OpenAILogger(
        verbose=True,
        show_raw=True,           # 终端是否显示 raw 数据
        log_file="session.md",   # 日志文件路径
        file_show_raw=True       # 文件中是否显示 raw 数据
    )

    # 使用命令行参数
    from logger_openai import parse_logger_args, create_logger_from_args
    args = parse_logger_args()
    logger = create_logger_from_args(args)

命令行参数:
    python litellm_s01.py --log-file session.md --no-show-raw --file-show-raw
    python litellm_s01.py -q  # 安静模式，只写文件
    python litellm_s01.py --log-file logs/session.md --append
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class OpenAILogger:
    """OpenAI 格式的 Agent 日志输出器，支持结构化日志、命令行参数和 Markdown 文件输出"""

    # ANSI 颜色代码
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    def __init__(
        self,
        verbose: bool = True,
        show_raw: bool = True,
        log_file: Optional[str] = None,
        file_show_raw: bool = True,
        append: bool = False,
    ):
        """
        初始化日志器

        Args:
            verbose: 是否在终端显示详细日志
            show_raw: 是否在终端显示原始 API 数据
            log_file: Markdown 日志文件路径 (None 表示不写入文件)
            file_show_raw: 是否在文件中显示原始 API 数据 (可折叠)
            append: 是否追加到现有日志文件 (False 则覆盖)
        """
        self.verbose = verbose
        self.show_raw = show_raw
        self.log_file = Path(log_file) if log_file else None
        self.file_show_raw = file_show_raw
        self.append = append
        self._iteration = 0
        self._session_start = datetime.now()

        # 初始化日志文件
        if self.log_file:
            self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件"""
        # 确保目录存在
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果不是追加模式，创建新文件
        if not self.append:
            self.log_file.write_text("")
            self._file_write(f"# Agent Session Log (OpenAI Format)\n\n")
            self._file_write(f"**Started:** {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self._file_write("---\n\n")

    def _file_write(self, content: str):
        """写入内容到日志文件"""
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(content)

    def _strip_ansi(self, text: str) -> str:
        """去除 ANSI 颜色码"""
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)

    def _timestamp_plain(self) -> str:
        """获取纯文本时间戳"""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _md_details_start(self, summary: str, open_by_default: bool = False) -> str:
        """生成 Markdown 可折叠区域开始标签"""
        open_attr = " open" if open_by_default else ""
        return f"<details{open_attr}>\n<summary>{summary}</summary>\n\n"

    def _md_details_end(self) -> str:
        """生成 Markdown 可折叠区域结束标签"""
        return "\n</details>\n\n"

    def _md_code_block(self, content: str, language: str = "json") -> str:
        """生成 Markdown 代码块"""
        return f"```{language}\n{content}\n```\n"

    def _color(self, text: str, color: str) -> str:
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _timestamp(self) -> str:
        return self._color(datetime.now().strftime("%H:%M:%S.%f")[:-3], "dim")

    # =========================================================================
    # 基础输出
    # =========================================================================

    def separator(self, title: str = "", char: str = "─", width: int = 80):
        if not self.verbose:
            return
        if title:
            line = char * 10 + f" {title} " + char * (width - 12 - len(title))
        else:
            line = char * width
        print(self._color(f"\n{line}", "dim"))

        # 写入文件
        if self.log_file:
            self._file_write(f"---\n\n**{title}** *({self._timestamp_plain()})*\n\n")

    def header(self, text: str, session_name: str = ""):
        print(self._color(f"\n{'═' * 80}", "cyan"))
        if session_name:
            print(self._color(f"  [{session_name}]", "dim"))
        print(self._color(f"  {text}", "bold"))
        print(self._color(f"{'═' * 80}", "cyan"))

        # 写入文件
        if self.log_file:
            self._file_write(f"## {text}\n\n")
            if session_name:
                self._file_write(f"`[{session_name}]`\n\n")

    def section(self, text: str, icon: str = "▶"):
        if not self.verbose:
            return
        print(self._color(f"\n{icon} {text}", "cyan"))

        # 写入文件
        if self.log_file:
            self._file_write(f"### {icon} {text}\n\n")

    def key_value(self, key: str, value: Any, indent: int = 2, color: str = "yellow", file_value: Any = None):
        """打印键值对

        Args:
            key: 键名
            value: 终端显示的值（可能包含 ANSI 颜色码）
            indent: 缩进空格数
            color: 键名颜色
            file_value: 写入文件的值（如果不同，用于避免 ANSI 码写入文件）
        """
        spaces = " " * indent
        key_str = self._color(f"{key}:", color)
        print(f"{spaces}{key_str} {value}")

        # 写入文件（使用 file_value 或去除 ANSI 码的 value）
        if self.log_file:
            clean_value = file_value if file_value is not None else self._strip_ansi(str(value))
            self._file_write(f"- **{key}:** {clean_value}\n")

    def _print_code_block(self, data: dict, indent: int = 2):
        """打印代码块格式的 JSON"""
        try:
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            spaces = " " * indent
            for line in formatted.split("\n"):
                # 截断过长的行
                if len(line) > 100:
                    line = line[:97] + "..."
                print(self._color(f"{spaces}{line}", "dim"))
        except Exception as e:
            print(self._color(f"  Error: {e}", "red"))

    # =========================================================================
    # 原始 API 数据显示
    # =========================================================================

    def request_raw(self, model: str, messages: list, tools: list, **kwargs):
        """
        显示原始 OpenAI API 请求数据
        """
        # 终端输出
        if self.show_raw:
            print(self._color("\n" + "┌" + "─" * 78 + "┐", "magenta"))
            print(self._color("│  📤 RAW API REQUEST (OpenAI Format)" + " " * 41 + "│", "magenta"))
            print(self._color("└" + "─" * 78 + "┘", "magenta"))

            # 显示请求结构摘要
            print(self._color(f"\n  📊 Request Summary:", "cyan"))
            self.key_value("model", model, indent=4, color="magenta")

            # 统计消息
            msg_summary = {}
            for msg in messages:
                role = msg.get("role", "unknown")
                msg_summary[role] = msg_summary.get(role, 0) + 1
            self.key_value("messages", str(msg_summary), indent=4, color="magenta")
            self.key_value("tools_count", str(len(tools)), indent=4, color="magenta")

            # 显示工具列表
            if tools:
                print(self._color(f"\n  🔧 Tools:", "cyan"))
                for tool in tools:
                    if tool.get("type") == "function":
                        fn = tool.get("function", {})
                        print(self._color(f"    - {fn.get('name', 'unknown')}: {fn.get('description', '')[:50]}...", "dim"))

            # 显示消息摘要
            print(self._color(f"\n  📋 Messages:", "cyan"))
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                role_color = {"system": "magenta", "user": "green", "assistant": "yellow", "tool": "blue"}.get(role, "white")
                content = msg.get("content", "")

                if role == "tool":
                    tool_id = msg.get("tool_call_id", "")[:16]
                    content_preview = str(content)[:40] if content else ""
                    print(f"    [{i}] {self._color(role, role_color)}: id={tool_id}..., content=\"{content_preview}...\"")
                elif isinstance(content, str):
                    preview = content[:50] + ("..." if len(content) > 50 else "")
                    print(f"    [{i}] {self._color(role, role_color)}: \"{preview}\"")
                else:
                    print(f"    [{i}] {self._color(role, role_color)}: {type(content).__name__}")

                # 显示 tool_calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        print(self._color(f"        └─ tool_call: {fn.get('name', '?')}({fn.get('arguments', '{}')[:30]}...)", "dim"))

            # 显示完整请求 JSON
            print(self._color(f"\n  📄 Full Request JSON:", "cyan"))
            request_data = {
                "model": model,
                "messages": messages,
                "tools": tools,
                **kwargs
            }
            self._print_code_block(request_data)

        # 文件输出
        if self.log_file:
            self._file_write(f"#### 📤 API Request\n\n")

            # 请求摘要 (可折叠)
            msg_summary = {}
            for msg in messages:
                role = msg.get("role", "unknown")
                msg_summary[role] = msg_summary.get(role, 0) + 1

            summary_data = {
                "model": model,
                "messages_summary": msg_summary,
                "tools_count": len(tools),
            }
            self._file_write(self._md_details_start("📊 Request Summary (click to expand)"))
            self._file_write(self._md_code_block(json.dumps(summary_data, ensure_ascii=False, indent=2)))
            self._file_write(self._md_details_end())

            # 完整请求 (可折叠)
            if self.file_show_raw:
                request_data = {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    **kwargs
                }
                self._file_write(self._md_details_start("📄 Full Request JSON (click to expand)"))
                self._file_write(self._md_code_block(json.dumps(request_data, ensure_ascii=False, indent=2)))
                self._file_write(self._md_details_end())

    def response_raw(self, response: dict):
        """
        显示原始 OpenAI API 响应数据

        OpenAI 响应格式:
        {
            "id": "chatcmpl-xxx",
            "model": "gpt-5.2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "tool_calls": [...]
                },
                "finish_reason": "stop" | "tool_calls"
            }],
            "usage": {
                "prompt_tokens": N,
                "completion_tokens": N,
                "total_tokens": N
            }
        }
        """
        # 提取关键信息
        response_id = response.get("id", "")
        model = response.get("model", "")
        choices = response.get("choices", [])
        usage = response.get("usage", {})

        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "unknown"
        tool_calls = message.get("tool_calls") or []

        # 终端输出
        if self.show_raw:
            print(self._color("\n" + "┌" + "─" * 78 + "┐", "blue"))
            print(self._color("│  📥 RAW API RESPONSE (OpenAI Format)" + " " * 40 + "│", "blue"))
            print(self._color("└" + "─" * 78 + "┘", "blue"))

            # 显示响应结构摘要
            print(self._color(f"\n  📊 Response Summary:", "cyan"))
            self.key_value("id", response_id, indent=4, color="blue")
            self.key_value("model", model, indent=4, color="blue")
            finish_color = "yellow" if finish_reason == "tool_calls" else "green"
            self.key_value("finish_reason", self._color(finish_reason, finish_color), indent=4, color="blue", file_value=finish_reason)
            self.key_value("usage", f"prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}", indent=4, color="blue")

            # 显示消息内容
            print(self._color(f"\n  📨 Message:", "cyan"))
            self.key_value("role", message.get("role", "assistant"), indent=4, color="blue")

            content = message.get("content")
            if content:
                preview = content[:100] + ("..." if len(content) > 100 else "")
                self.key_value("content", f'"{preview}"', indent=4, color="blue")

            # 显示 tool_calls
            if tool_calls:
                print(self._color(f"\n  ⚡ Tool Calls:", "green"))
                for i, tc in enumerate(tool_calls):
                    tc_id = tc.get("id", "")[:20]
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "unknown")
                    fn_args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        args_preview = json.dumps(args_dict, ensure_ascii=False)[:60]
                    except:
                        args_preview = str(fn_args)[:60]

                    print(self._color(f"    [{i}] {fn_name}()", "green"))
                    print(self._color(f"        id: {tc_id}...", "dim"))
                    print(self._color(f"        args: {args_preview}...", "dim"))

            # 显示完整响应 JSON
            print(self._color(f"\n  📄 Full Response JSON:", "cyan"))
            self._print_code_block(response)

        # 文件输出
        if self.log_file:
            self._file_write(f"#### 📥 API Response\n\n")

            # 响应摘要 (可折叠)
            summary_data = {
                "id": response_id,
                "model": model,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                },
            }
            if tool_calls:
                summary_data["tool_calls_count"] = len(tool_calls)

            self._file_write(self._md_details_start("📊 Response Summary (click to expand)"))
            self._file_write(self._md_code_block(json.dumps(summary_data, ensure_ascii=False, indent=2)))
            self._file_write(self._md_details_end())

            # 完整响应 (可折叠)
            if self.file_show_raw:
                self._file_write(self._md_details_start("📄 Full Response JSON (click to expand)"))
                self._file_write(self._md_code_block(json.dumps(response, ensure_ascii=False, indent=2)))
                self._file_write(self._md_details_end())

    # =========================================================================
    # 循环和消息追踪
    # =========================================================================

    def loop_iteration(self, iteration: int):
        if not self.verbose:
            return
        self._iteration = iteration
        print(self._color(f"\n{'┌' + '─' * 78 + '┐'}", "cyan"))
        print(self._color(f"│  🔄 LOOP ITERATION #{iteration:<62}│", "cyan"))
        print(self._color(f"{'└' + '─' * 78 + '┘'}", "cyan"))

        # 写入文件
        if self.log_file:
            self._file_write(f"\n---\n\n## 🔄 Loop Iteration #{iteration}\n\n")
            self._file_write(f"*Time: {self._timestamp_plain()}*\n\n")

    def messages_snapshot(self, messages: list, title: str = "MESSAGES"):
        if not self.verbose:
            return
        print(self._color(f"\n  📋 {title}", "blue"))
        print(self._color(f"  Total: {len(messages)} messages", "dim"))
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            role_color = {"system": "magenta", "user": "green", "assistant": "yellow", "tool": "blue"}.get(role, "white")
            content = msg.get("content", "")

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")[:16]
                preview = str(content)[:40] if content else ""
                print(f"    [{i}] {self._color(role, role_color)}: id={tool_id}..., \"{preview}...\"")
            elif isinstance(content, str):
                preview = content[:50] + ("..." if len(content) > 50 else "")
                print(f"    [{i}] {self._color(role, role_color)}: \"{preview}\"")
            else:
                print(f"    [{i}] {self._color(role, role_color)}: {type(content).__name__}")

            # 显示 tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    print(self._color(f"        └─ {fn.get('name', '?')}()", "dim"))

        # 写入文件
        if self.log_file:
            self._file_write(f"### 📋 {title}\n\n")
            self._file_write(f"**Total messages:** {len(messages)}\n\n")

            # 可折叠的消息详情
            self._file_write(self._md_details_start("Message Details (click to expand)"))
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "tool":
                    tool_id = msg.get("tool_call_id", "")[:16]
                    preview = str(content)[:100] if content else ""
                    self._file_write(f"- **[{i}] `{role}`:** id={tool_id}..., \"{preview}...\"\n")
                elif isinstance(content, str):
                    preview = content[:150] + ("..." if len(content) > 150 else "")
                    self._file_write(f"- **[{i}] `{role}`:** {preview}\n")
                else:
                    self._file_write(f"- **[{i}] `{role}`:** {type(content).__name__}\n")

            self._file_write(self._md_details_end())

    # =========================================================================
    # 工具调用显示
    # =========================================================================

    def tool_call(self, name: str, args: dict, call_id: str = ""):
        print(self._color(f"\n  ⚡ TOOL CALL", "green"))
        if call_id:
            self.key_value("call_id", self._color(call_id[:24] + "...", "dim"), indent=4, color="green")
        self.key_value("name", self._color(name, "green"), indent=4, color="green")
        self.key_value("arguments", "", indent=4, color="green")
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:60] + "..."
            print(self._color(f"      {k}: {v_str}", "dim"))

        # 写入文件
        if self.log_file:
            self._file_write(f"#### ⚡ Tool Call: `{name}`\n\n")
            if call_id:
                self._file_write(f"- **ID:** `{call_id}`\n")
            self._file_write(f"- **Arguments:**\n\n{self._md_code_block(json.dumps(args, ensure_ascii=False, indent=2))}")

    def tool_result(self, call_id: str, content: str, is_error: bool = False):
        color = "red" if is_error else "blue"
        icon = "❌" if is_error else "✓"
        print(self._color(f"\n  {icon} TOOL RESULT", color))
        self.key_value("call_id", call_id[:24] + "...", indent=4, color=color)
        content_preview = content[:200] + ("..." if len(content) > 200 else "")
        # 多行内容缩进显示
        for line in content_preview.split("\n")[:5]:
            print(self._color(f"      {line}", "dim"))

        # 写入文件
        if self.log_file:
            status = "❌ Error" if is_error else "✓ Success"
            self._file_write(f"#### {status} Tool Result\n\n")
            self._file_write(f"- **Call ID:** `{call_id}`\n\n")

            # 可折叠的完整内容
            self._file_write(self._md_details_start("Full Content (click to expand)"))
            self._file_write(f"```\n{content}\n```\n")
            self._file_write(self._md_details_end())

    # =========================================================================
    # LLM 交互摘要
    # =========================================================================

    def llm_response_summary(self, finish_reason: str, usage: dict, tool_calls_count: int = 0):
        """打印 LLM 响应摘要"""
        if not self.verbose:
            return
        print(self._color(f"\n  📥 LLM RESPONSE SUMMARY", "magenta"))
        finish_color = "yellow" if finish_reason == "tool_calls" else "green"
        self.key_value("finish_reason", self._color(finish_reason, finish_color), indent=4, color="magenta", file_value=finish_reason)
        if tool_calls_count > 0:
            self.key_value("tool_calls_count", str(tool_calls_count), indent=4, color="magenta")
        self.key_value("usage", f"prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}", indent=4, color="magenta")

        # 写入文件
        if self.log_file:
            self._file_write(f"**📥 LLM Response Summary:**\n\n")
            self._file_write(f"- Finish Reason: `{finish_reason}`\n")
            if tool_calls_count > 0:
                self._file_write(f"- Tool Calls: {tool_calls_count}\n")
            self._file_write(f"- Tokens: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}\n\n")

    # =========================================================================
    # 其他
    # =========================================================================

    def loop_end(self, reason: str):
        self.section(f"🏁 LOOP END: {reason}", "🛑")

        # 写入文件
        if self.log_file:
            self._file_write(f"### 🏁 Loop End\n\n")
            self._file_write(f"**Reason:** `{reason}`\n\n")

    def user_input(self, query: str):
        self.separator("USER INPUT")
        print(f"  {query}")

        # 写入文件
        if self.log_file:
            self._file_write(f"### 👤 User Input\n\n")
            self._file_write(f"> {query}\n\n")

    def config(self, **kwargs):
        """显示配置信息"""
        print(self._color(f"\n  ⚙️ Configuration:", "cyan"))
        for k, v in kwargs.items():
            if v:
                self.key_value(k, v, indent=4, color="cyan")

        # 写入文件
        if self.log_file:
            self._file_write(f"### ⚙️ Configuration\n\n")
            for k, v in kwargs.items():
                if v:
                    self._file_write(f"- **{k}:** `{v}`\n")
            self._file_write("\n")

    # =========================================================================
    # 会话结束
    # =========================================================================

    def session_end(self, summary: str = ""):
        """结束会话，写入总结"""
        if self.log_file:
            end_time = datetime.now()
            duration = end_time - self._session_start

            self._file_write(f"\n---\n\n")
            self._file_write(f"## 🏁 Session End\n\n")
            self._file_write(f"**Ended:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self._file_write(f"**Duration:** {str(duration).split('.')[0]}\n\n")
            self._file_write(f"**Total Iterations:** {self._iteration}\n\n")
            if summary:
                self._file_write(f"**Summary:**\n\n{summary}\n")


# =============================================================================
# 命令行参数解析
# =============================================================================

# 兼容 Python 3.8 及以下版本的 BooleanOptionalAction
class BooleanOptionalAction(argparse.Action):
    """兼容旧版 Python 的布尔参数 Action"""
    def __init__(self, option_strings, dest, default=None, type=None,
                 choices=None, required=False, help=None, metavar=None):
        _option_strings = []
        for option_string in option_strings:
            if option_string.startswith("--no-"):
                _option_strings.append(option_string)
            else:
                _option_strings.append(option_string)
                _option_strings.append(f"--no-{option_string[2:]}")
        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string and option_string.startswith("--no-"):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


def add_logger_args(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    为 ArgumentParser 添加日志相关参数

    可以传入现有的 parser，或创建新的 parser。

    Args:
        parser: 现有的 ArgumentParser 实例，如果为 None 则创建新的

    Returns:
        添加了日志参数的 ArgumentParser
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Agent with structured logging (OpenAI format)")

    # 日志输出控制
    log_group = parser.add_argument_group("Logging Options")

    log_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安静模式：不在终端显示详细日志 (等价于 --no-verbose)"
    )

    log_group.add_argument(
        "--verbose", "--no-verbose",
        dest="verbose",
        action=BooleanOptionalAction,
        default=True,
        help="在终端显示详细日志 (默认: True，使用 --no-verbose 或 -q 关闭)"
    )

    log_group.add_argument(
        "--show-raw", "--no-show-raw",
        dest="show_raw",
        action=BooleanOptionalAction,
        default=True,
        help="在终端显示原始 API 请求数据 (默认: True)"
    )

    # 文件输出
    file_group = parser.add_argument_group("File Output Options")

    file_group.add_argument(
        "-o", "--log-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Markdown 日志文件路径 (例如: logs/session.md)"
    )

    file_group.add_argument(
        "--file-show-raw", "--no-file-show-raw",
        dest="file_show_raw",
        action=BooleanOptionalAction,
        default=True,
        help="在文件中显示原始 API 数据 (默认: True，使用可折叠区域)"
    )

    file_group.add_argument(
        "-a", "--append",
        action="store_true",
        help="追加到现有日志文件 (默认: 覆盖)"
    )

    return parser


def parse_logger_args(args: list = None) -> argparse.Namespace:
    """
    解析日志相关的命令行参数

    Args:
        args: 命令行参数列表，None 则使用 sys.argv

    Returns:
        解析后的参数命名空间
    """
    parser = add_logger_args()

    # 处理 -q 等价于 --no-verbose
    parsed = parser.parse_args(args)
    if hasattr(parsed, 'quiet') and parsed.quiet:
        parsed.verbose = False

    return parsed


def create_logger_from_args(args: argparse.Namespace = None) -> OpenAILogger:
    """
    根据命令行参数创建日志器实例

    Args:
        args: parse_logger_args() 返回的参数，如果为 None 则解析命令行

    Returns:
        配置好的 OpenAILogger 实例
    """
    if args is None:
        args = parse_logger_args()

    return OpenAILogger(
        verbose=getattr(args, 'verbose', True),
        show_raw=getattr(args, 'show_raw', True),
        log_file=getattr(args, 'log_file', None),
        file_show_raw=getattr(args, 'file_show_raw', True),
        append=getattr(args, 'append', False),
    )


def get_logger_config_string(args: argparse.Namespace = None) -> str:
    """
    获取当前日志配置的字符串描述（用于启动时显示）

    Args:
        args: 解析后的参数

    Returns:
        配置描述字符串
    """
    if args is None:
        args = parse_logger_args()

    config_parts = []

    if args.verbose:
        config_parts.append("终端: 详细日志")
    else:
        config_parts.append("终端: 简洁模式")

    if args.show_raw:
        config_parts.append("显示 RAW")
    else:
        config_parts.append("隐藏 RAW")

    if args.log_file:
        config_parts.append(f"日志文件: {args.log_file}")
        if args.file_show_raw:
            config_parts.append("文件 RAW: 可折叠")
        else:
            config_parts.append("文件 RAW: 隐藏")
        if args.append:
            config_parts.append("追加模式")
    else:
        config_parts.append("无日志文件")

    return " | ".join(config_parts)
