#!/usr/bin/env python3
"""
logger.py - 模块化的 Agent 日志系统

提供结构化的日志输出，用于追踪 Agent Loop 的每一步。
支持多种日志级别和格式化输出，支持 Markdown 文件输出。

使用方法:
    from logger import AgentLogger

    # 基础用法
    logger = AgentLogger(verbose=True, show_raw=True)

    # 输出到 Markdown 文件
    logger = AgentLogger(
        verbose=True,
        show_raw=True,           # 终端是否显示 raw 数据
        log_file="session.md",   # 日志文件路径
        file_show_raw=True       # 文件中是否显示 raw 数据
    )

    # 使用命令行参数
    from logger import parse_logger_args, create_logger_from_args
    args = parse_logger_args()
    logger = create_logger_from_args(args)

    logger.request_raw(request_data)
    logger.response_raw(response_data)

命令行参数:
    python s01_basic_loop.py --log-file session.md --no-show-raw --file-show-raw
    python s01_basic_loop.py -q  # 安静模式，只写文件
    python s01_basic_loop.py --log-file logs/session.md --append
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AgentLogger:
    """Agent 日志输出器，支持结构化日志、原始数据显示和 Markdown 文件输出"""

    # ANSI 颜色代码
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "underline": "\033[4m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bg_black": "\033[40m",
        "bg_red": "\033[41m",
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_magenta": "\033[45m",
        "bg_cyan": "\033[46m",
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
            self._file_write(f"# Agent Session Log\n\n")
            self._file_write(f"**Started:** {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self._file_write("---\n\n")

    def _file_write(self, content: str):
        """写入内容到日志文件"""
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(content)

    def _color(self, text: str, color: str) -> str:
        """添加颜色"""
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def _timestamp(self) -> str:
        """获取时间戳"""
        return self._color(datetime.now().strftime("%H:%M:%S.%f")[:-3], "dim")

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

    # =========================================================================
    # 基础输出方法
    # =========================================================================

    def separator(self, title: str = "", char: str = "─", width: int = 80):
        """打印分隔线"""
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
        """打印标题头"""
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
        """打印章节标题"""
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

    def _strip_ansi(self, text: str) -> str:
        """去除 ANSI 颜色码"""
        import re
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        return ansi_escape.sub('', text)

    def json_block(self, title: str, data: Any, indent: int = 2, color: str = "magenta"):
        """打印 JSON 格式的内容"""
        if not self.verbose:
            return
        spaces = " " * indent
        title_str = self._color(f"{title}:", color)
        print(f"{spaces}{title_str}")
        try:
            formatted = json.dumps(data, ensure_ascii=False, indent=indent + 2)
            for line in formatted.split("\n"):
                print(self._color(f"{spaces}  {line}", "dim"))
        except Exception:
            print(self._color(f"{spaces}  {data}", "dim"))

        # 写入文件
        if self.log_file:
            try:
                formatted = json.dumps(data, ensure_ascii=False, indent=2)
                self._file_write(f"**{title}:**\n\n{self._md_code_block(formatted)}")
            except Exception:
                self._file_write(f"**{title}:** `{data}`\n\n")

    # =========================================================================
    # 原始 API 数据显示 (核心功能)
    # =========================================================================

    def request_raw(self, model: str, system: str, messages: list, tools: list, max_tokens: int = 8000):
        """
        结构化显示原始 API 请求数据

        展示发送给 LLM API 的完整请求结构，帮助理解底层数据格式。
        """
        # 终端输出
        if self.show_raw:
            print(self._color("\n" + "┌" + "─" * 78 + "┐", "magenta"))
            print(self._color("│  📤 RAW API REQUEST" + " " * 57 + "│", "magenta"))
            print(self._color("└" + "─" * 78 + "┘", "magenta"))

            # 构建请求数据结构
            request_data = self._build_request_summary(model, system, messages, tools, max_tokens)
            self._print_structured_json(request_data, "Request Structure")

            # 显示完整请求 JSON (可选)
            print(self._color("\n  📄 Full Request JSON (copy-paste ready):", "cyan"))
            full_request = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "tools": tools,
                "messages": self._serialize_messages(messages)
            }
            self._print_code_block(full_request)

        # 文件输出
        if self.log_file:
            self._file_write_request_raw(model, system, messages, tools, max_tokens)

    def _file_write_request_raw(self, model: str, system: str, messages: list, tools: list, max_tokens: int):
        """将原始请求写入 Markdown 文件"""
        self._file_write(f"#### 📤 API Request\n\n")

        # 请求摘要 (可折叠)
        summary_data = self._build_request_summary(model, system, messages, tools, max_tokens)
        self._file_write(self._md_details_start("📊 Request Summary (click to expand)"))
        self._file_write(self._md_code_block(json.dumps(summary_data, ensure_ascii=False, indent=2)))
        self._file_write(self._md_details_end())

        # 完整请求 (可折叠)
        if self.file_show_raw:
            full_request = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "tools": tools,
                "messages": self._serialize_messages(messages)
            }

            self._file_write(self._md_details_start("📄 Full Request JSON (click to expand)"))
            self._file_write(self._md_code_block(json.dumps(full_request, ensure_ascii=False, indent=2)))
            self._file_write(self._md_details_end())

    def response_raw(self, response):
        """
        结构化显示原始 API 响应数据

        展示从 LLM API 返回的完整响应结构，帮助理解底层数据格式。
        """
        # 终端输出
        if self.show_raw:
            print(self._color("\n" + "┌" + "─" * 78 + "┐", "blue"))
            print(self._color("│  📥 RAW API RESPONSE" + " " * 56 + "│", "blue"))
            print(self._color("└" + "─" * 78 + "┘", "blue"))

            # 构建响应数据结构
            response_data = self._build_response_summary(response)
            self._print_structured_json(response_data, "Response Structure")

            # 显示完整响应 JSON
            print(self._color("\n  📄 Full Response JSON (copy-paste ready):", "cyan"))
            full_response = {
                "id": response.id,
                "model": response.model,
                "role": response.role,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "content": self._serialize_content(response.content)
            }
            self._print_code_block(full_response)

        # 文件输出
        if self.log_file:
            self._file_write_response_raw(response)

    def _file_write_response_raw(self, response):
        """将原始响应写入 Markdown 文件"""
        self._file_write(f"#### 📥 API Response\n\n")

        # 响应摘要 (可折叠)
        summary_data = self._build_response_summary(response)
        self._file_write(self._md_details_start("📊 Response Summary (click to expand)"))
        self._file_write(self._md_code_block(json.dumps(summary_data, ensure_ascii=False, indent=2)))
        self._file_write(self._md_details_end())

        # 完整响应 (可折叠)
        if self.file_show_raw:
            full_response = {
                "id": response.id,
                "model": response.model,
                "role": response.role,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "content": self._serialize_content(response.content)
            }

            self._file_write(self._md_details_start("📄 Full Response JSON (click to expand)"))
            self._file_write(self._md_code_block(json.dumps(full_response, ensure_ascii=False, indent=2)))
            self._file_write(self._md_details_end())

    def _build_request_summary(self, model: str, system: str, messages: list, tools: list, max_tokens: int) -> dict:
        """构建请求摘要"""
        request_data = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system[:100] + "..." if len(system) > 100 else system,
            "tools": [{"name": t["name"], "description": t["description"][:50] + "..."} for t in tools],
            "messages": []
        }

        # 简化消息显示
        for i, msg in enumerate(messages):
            msg_entry = {"role": msg["role"]}
            content = msg.get("content", "")

            if isinstance(content, str):
                msg_entry["content"] = f"<text: {len(content)} chars>"
            elif isinstance(content, list):
                blocks_summary = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "unknown")
                    else:
                        block_type = getattr(block, "type", "unknown")

                    if block_type == "tool_result":
                        tool_id = block.get("tool_use_id", "") if isinstance(block, dict) else getattr(block, "tool_use_id", "")
                        blocks_summary.append(f"tool_result(id={tool_id[:16]}...)")
                    elif block_type == "tool_use":
                        name = block.get("name", "") if isinstance(block, dict) else getattr(block, "name", "")
                        blocks_summary.append(f"tool_use(name={name})")
                    else:
                        blocks_summary.append(block_type)
                msg_entry["content"] = blocks_summary

            request_data["messages"].append(msg_entry)

        return request_data

    def _build_response_summary(self, response) -> dict:
        """构建响应摘要"""
        response_data = {
            "id": response.id,
            "model": response.model,
            "role": response.role,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "content": []
        }

        # 解析 content blocks
        for block in response.content:
            block_type = getattr(block, "type", "unknown")
            block_entry = {"type": block_type}

            if block_type == "text":
                text = getattr(block, "text", "")
                block_entry["text"] = f"<{len(text)} chars>"
            elif block_type == "tool_use":
                block_entry["id"] = getattr(block, "id", "")
                block_entry["name"] = getattr(block, "name", "")
                block_entry["input"] = getattr(block, "input", {})

            response_data["content"].append(block_entry)

        return response_data

    def _serialize_messages(self, messages: list) -> list:
        """序列化消息列表为可 JSON 化的格式"""
        result = []
        for msg in messages:
            msg_dict = {"role": msg["role"]}
            content = msg.get("content", "")

            if isinstance(content, str):
                msg_dict["content"] = content
            elif isinstance(content, list):
                msg_dict["content"] = self._serialize_content(content)
            else:
                msg_dict["content"] = str(content)

            result.append(msg_dict)
        return result

    def _serialize_content(self, content) -> list:
        """序列化 content blocks 为可 JSON 化的格式"""
        result = []
        for block in content:
            if isinstance(block, dict):
                result.append(block)
            else:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    result.append({
                        "type": "text",
                        "text": getattr(block, "text", "")
                    })
                elif block_type == "tool_use":
                    result.append({
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": dict(getattr(block, "input", {}))
                    })
                else:
                    result.append({"type": str(block_type)})
        return result

    def _print_structured_json(self, data: dict, title: str):
        """打印结构化 JSON 数据"""
        print(self._color(f"\n  📊 {title}:", "cyan"))
        try:
            formatted = json.dumps(data, ensure_ascii=False, indent=4)
            for line in formatted.split("\n"):
                if '":' in line:
                    print(self._color(f"    {line}", "dim"))
                else:
                    print(self._color(f"    {line}", "dim"))
        except Exception as e:
            print(self._color(f"    Error formatting: {e}", "red"))

    def _print_code_block(self, data: dict):
        """打印代码块格式的 JSON"""
        try:
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            print(self._color("  " + "┌" + "─" * 76 + "┐", "dim"))
            for line in formatted.split("\n"):
                # 截断过长的行
                if len(line) > 74:
                    line = line[:71] + "..."
                print(self._color(f"  │ {line:<74} │", "dim"))
            print(self._color("  " + "└" + "─" * 76 + "┘", "dim"))
        except Exception as e:
            print(self._color(f"    Error: {e}", "red"))

    # =========================================================================
    # 循环和消息追踪
    # =========================================================================

    def loop_iteration(self, iteration: int):
        """打印循环迭代"""
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

    def messages_snapshot(self, messages: list, title: str = "MESSAGES SNAPSHOT"):
        """打印当前消息列表的快照"""
        if not self.verbose:
            return
        print(self._color(f"\n  📋 {title}", "blue"))
        print(self._color(f"  Total messages: {len(messages)}", "dim"))
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            role_color = "green" if role == "user" else "yellow" if role == "assistant" else "white"
            content = msg.get("content", "")

            # 简化 content 显示
            if isinstance(content, str):
                preview = content[:60] + ("..." if len(content) > 60 else "")
                print(f"    [{i}] {self._color(role, role_color)}: {self._color(preview, 'dim')}")
            elif isinstance(content, list):
                # 工具结果列表
                block_types = []
                for b in content:
                    if isinstance(b, dict):
                        block_types.append(b.get('type', 'unknown'))
                    else:
                        block_types.append(getattr(b, 'type', 'unknown'))
                print(f"    [{i}] {self._color(role, role_color)}: {self._color(str(block_types), 'dim')}")

        # 写入文件
        if self.log_file:
            self._file_write(f"### 📋 {title}\n\n")
            self._file_write(f"**Total messages:** {len(messages)}\n\n")

            # 可折叠的消息详情
            self._file_write(self._md_details_start("Message Details (click to expand)"))
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if isinstance(content, str):
                    preview = content[:200] + ("..." if len(content) > 200 else "")
                    self._file_write(f"- **[{i}] `{role}`:** {preview}\n")
                elif isinstance(content, list):
                    block_types = []
                    for b in content:
                        if isinstance(b, dict):
                            block_types.append(b.get('type', 'unknown'))
                        else:
                            block_types.append(getattr(b, 'type', 'unknown'))
                    self._file_write(f"- **[{i}] `{role}`:** {block_types}\n")

            self._file_write(self._md_details_end())

    # =========================================================================
    # 工具调用显示
    # =========================================================================

    def tool_call(self, name: str, input_data: dict, tool_id: str = ""):
        """打印工具调用"""
        print(self._color(f"\n  ⚡ TOOL CALL", "green"))
        if tool_id:
            self.key_value("id", self._color(tool_id[:24] + "...", "dim"), indent=4, color="green")
        self.key_value("name", self._color(name, "green"), indent=4, color="green")
        self.key_value("input", "", indent=4, color="green")
        for k, v in input_data.items():
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:60] + "..."
            print(self._color(f"      {k}: {v_str}", "dim"))

        # 写入文件
        if self.log_file:
            self._file_write(f"#### ⚡ Tool Call: `{name}`\n\n")
            if tool_id:
                self._file_write(f"- **ID:** `{tool_id}`\n")
            self._file_write(f"- **Input:**\n\n{self._md_code_block(json.dumps(input_data, ensure_ascii=False, indent=2))}")

    def tool_result(self, tool_id: str, content: str, is_error: bool = False):
        """打印工具结果"""
        color = "red" if is_error else "blue"
        icon = "❌" if is_error else "✓"
        print(self._color(f"\n  {icon} TOOL RESULT", color))
        self.key_value("tool_use_id", tool_id[:24] + "...", indent=4, color=color)
        content_preview = content[:300] + ("..." if len(content) > 300 else "")
        self.key_value("content", self._color(f'"{content_preview}"', "dim"), indent=4, color=color)

        # 写入文件
        if self.log_file:
            status = "❌ Error" if is_error else "✓ Success"
            self._file_write(f"#### {status} Tool Result\n\n")
            self._file_write(f"- **Tool ID:** `{tool_id}`\n\n")

            # 可折叠的完整内容
            self._file_write(self._md_details_start("Full Content (click to expand)"))
            self._file_write(f"```\n{content}\n```\n")
            self._file_write(self._md_details_end())

    # =========================================================================
    # LLM 交互摘要
    # =========================================================================

    def llm_request_summary(self, model: str, messages_count: int, tools_count: int):
        """打印 LLM 请求摘要"""
        if not self.verbose:
            return
        print(self._color(f"\n  📤 LLM REQUEST SUMMARY", "magenta"))
        self.key_value("model", model, indent=4, color="magenta")
        self.key_value("messages_count", str(messages_count), indent=4, color="magenta")
        self.key_value("tools_count", str(tools_count), indent=4, color="magenta")
        self.key_value("timestamp", self._timestamp(), indent=4, color="magenta")

        # 写入文件
        if self.log_file:
            self._file_write(f"**📤 LLM Request Summary:**\n\n")
            self._file_write(f"- Model: `{model}`\n")
            self._file_write(f"- Messages: {messages_count}\n")
            self._file_write(f"- Tools: {tools_count}\n\n")

    def llm_response_summary(self, stop_reason: str, usage: dict, content_blocks: int):
        """打印 LLM 响应摘要"""
        if not self.verbose:
            return
        print(self._color(f"\n  📥 LLM RESPONSE SUMMARY", "magenta"))
        stop_color = "yellow" if stop_reason == "tool_use" else "green"
        # 传入原始 stop_reason 作为 file_value，避免 ANSI 码写入文件
        self.key_value("stop_reason", self._color(stop_reason, stop_color), indent=4, color="magenta", file_value=stop_reason)
        self.key_value("content_blocks", str(content_blocks), indent=4, color="magenta")
        self.key_value("usage", f"input={usage.get('input_tokens', 0)}, output={usage.get('output_tokens', 0)}", indent=4, color="magenta")

        # 写入文件
        if self.log_file:
            self._file_write(f"**📥 LLM Response Summary:**\n\n")
            self._file_write(f"- Stop Reason: `{stop_reason}`\n")
            self._file_write(f"- Content Blocks: {content_blocks}\n")
            self._file_write(f"- Tokens: input={usage.get('input_tokens', 0)}, output={usage.get('output_tokens', 0)}\n\n")

    def response_content_blocks(self, content_blocks: list):
        """打印响应内容块详情"""
        if not self.verbose:
            return
        self.section("Response Content Blocks", "📦")
        for i, block in enumerate(content_blocks):
            block_type = getattr(block, "type", "unknown") if not isinstance(block, dict) else block.get("type", "unknown")
            if block_type == "text":
                text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
                text_preview = text[:100] + ("..." if len(text) > 100 else "")
                self.key_value(f"Block [{i}]", f'type={block_type}, text="{text_preview}"', indent=4)
            elif block_type == "tool_use":
                name = getattr(block, "name", "") if not isinstance(block, dict) else block.get("name", "")
                self.key_value(f"Block [{i}]", f"type={block_type}, name={name}", indent=4)

        # 写入文件
        if self.log_file:
            self._file_write(f"### 📦 Response Content Blocks\n\n")
            for i, block in enumerate(content_blocks):
                block_type = getattr(block, "type", "unknown") if not isinstance(block, dict) else block.get("type", "unknown")
                if block_type == "text":
                    text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
                    text_preview = text[:200] + ("..." if len(text) > 200 else "")
                    self._file_write(f"- **Block [{i}]** (text): {text_preview}\n")
                elif block_type == "tool_use":
                    name = getattr(block, "name", "") if not isinstance(block, dict) else block.get("name", "")
                    self._file_write(f"- **Block [{i}]** (tool_use): `{name}`\n")
            self._file_write("\n")

    def loop_end(self, reason: str):
        """打印循环结束"""
        self.section(f"🏁 LOOP END: {reason}", "🛑")

        # 写入文件
        if self.log_file:
            self._file_write(f"### 🏁 Loop End\n\n")
            self._file_write(f"**Reason:** `{reason}`\n\n")

    def user_input(self, query: str):
        """打印用户输入"""
        self.separator("USER INPUT")
        print(f"  {query}")

        # 写入文件
        if self.log_file:
            self._file_write(f"### 👤 User Input\n\n")
            self._file_write(f"> {query}\n\n")

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
# 便捷函数 - 用于向后兼容
# =============================================================================

# 默认全局实例
_default_logger = AgentLogger()


def get_logger(
    verbose: bool = True,
    show_raw: bool = True,
    log_file: Optional[str] = None,
    file_show_raw: bool = True,
    append: bool = False,
) -> AgentLogger:
    """
    获取日志器实例

    Args:
        verbose: 是否在终端显示详细日志
        show_raw: 是否在终端显示原始 API 数据
        log_file: Markdown 日志文件路径
        file_show_raw: 是否在文件中显示原始 API 数据
        append: 是否追加到现有日志文件

    Returns:
        AgentLogger 实例
    """
    return AgentLogger(
        verbose=verbose,
        show_raw=show_raw,
        log_file=log_file,
        file_show_raw=file_show_raw,
        append=append,
    )


# =============================================================================
# 命令行参数解析
# =============================================================================

def add_logger_args(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    为 ArgumentParser 添加日志相关参数

    可以传入现有的 parser，或创建新的 parser。

    Args:
        parser: 现有的 ArgumentParser 实例，如果为 None 则创建新的

    Returns:
        添加了日志参数的 ArgumentParser

    使用示例:
        # 方式1: 使用现有的 parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", help="输入文件")
        add_logger_args(parser)
        args = parser.parse_args()

        # 方式2: 只解析日志参数
        args = parse_logger_args()
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Agent with structured logging")

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


def parse_logger_args(args: list = None) -> argparse.Namespace:
    """
    解析日志相关的命令行参数

    Args:
        args: 命令行参数列表，None 则使用 sys.argv

    Returns:
        解析后的参数命名空间

    使用示例:
        args = parse_logger_args()
        logger = create_logger_from_args(args)
    """
    parser = add_logger_args()

    # 处理 -q 等价于 --no-verbose
    parsed = parser.parse_args(args)
    if hasattr(parsed, 'quiet') and parsed.quiet:
        parsed.verbose = False

    return parsed


def create_logger_from_args(args: argparse.Namespace = None) -> AgentLogger:
    """
    根据命令行参数创建日志器实例

    Args:
        args: parse_logger_args() 返回的参数，如果为 None 则解析命令行

    Returns:
        配置好的 AgentLogger 实例

    使用示例:
        # 在 agent 脚本中
        from logger import create_logger_from_args
        logger = create_logger_from_args()
    """
    if args is None:
        args = parse_logger_args()

    return AgentLogger(
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
