#!/usr/bin/env python3
"""
logger_openai.py - OpenAI 格式的 Agent 日志系统

专门为 OpenAI 消息格式设计的日志输出器，直接处理 OpenAI 的数据结构。

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
    from logger_openai import OpenAILogger

    logger = OpenAILogger(verbose=True, show_raw=True)
    logger.request_raw(model, messages, tools)
    logger.response_raw(response_dict)
"""

import json
from datetime import datetime
from typing import Any, Optional


class OpenAILogger:
    """OpenAI 格式的 Agent 日志输出器"""

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

    def __init__(self, verbose: bool = True, show_raw: bool = True):
        self.verbose = verbose
        self.show_raw = show_raw
        self._iteration = 0

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

    def header(self, text: str, session_name: str = ""):
        print(self._color(f"\n{'═' * 80}", "cyan"))
        if session_name:
            print(self._color(f"  [{session_name}]", "dim"))
        print(self._color(f"  {text}", "bold"))
        print(self._color(f"{'═' * 80}", "cyan"))

    def section(self, text: str, icon: str = "▶"):
        if not self.verbose:
            return
        print(self._color(f"\n{icon} {text}", "cyan"))

    def key_value(self, key: str, value: Any, indent: int = 2, color: str = "yellow"):
        spaces = " " * indent
        key_str = self._color(f"{key}:", color)
        print(f"{spaces}{key_str} {value}")

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
        if not self.show_raw:
            return

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
        if not self.show_raw:
            return

        print(self._color("\n" + "┌" + "─" * 78 + "┐", "blue"))
        print(self._color("│  📥 RAW API RESPONSE (OpenAI Format)" + " " * 40 + "│", "blue"))
        print(self._color("└" + "─" * 78 + "┘", "blue"))

        # 提取关键信息
        response_id = response.get("id", "")
        model = response.get("model", "")
        choices = response.get("choices", [])
        usage = response.get("usage", {})

        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "unknown"
        tool_calls = message.get("tool_calls") or []

        # 显示响应结构摘要
        print(self._color(f"\n  📊 Response Summary:", "cyan"))
        self.key_value("id", response_id, indent=4, color="blue")
        self.key_value("model", model, indent=4, color="blue")
        finish_color = "yellow" if finish_reason == "tool_calls" else "green"
        self.key_value("finish_reason", self._color(finish_reason, finish_color), indent=4, color="blue")
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

    def tool_result(self, call_id: str, content: str, is_error: bool = False):
        color = "red" if is_error else "blue"
        icon = "❌" if is_error else "✓"
        print(self._color(f"\n  {icon} TOOL RESULT", color))
        self.key_value("call_id", call_id[:24] + "...", indent=4, color=color)
        content_preview = content[:200] + ("..." if len(content) > 200 else "")
        # 多行内容缩进显示
        for line in content_preview.split("\n")[:5]:
            print(self._color(f"      {line}", "dim"))

    # =========================================================================
    # 其他
    # =========================================================================

    def loop_end(self, reason: str):
        self.section(f"🏁 LOOP END: {reason}", "🛑")

    def user_input(self, query: str):
        self.separator("USER INPUT")
        print(f"  {query}")

    def config(self, **kwargs):
        """显示配置信息"""
        print(self._color(f"\n  ⚙️ Configuration:", "cyan"))
        for k, v in kwargs.items():
            if v:
                self.key_value(k, v, indent=4, color="cyan")
