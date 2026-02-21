"use client";

import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useTranslations } from "@/lib/i18n";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface SourceViewerProps {
  source: string;
  filename: string;
}

function highlightLine(line: string): React.ReactNode[] {
  const tokens: React.ReactNode[] = [];
  let remaining = line;
  let key = 0;

  const patterns: {
    regex: RegExp;
    className: string;
  }[] = [
    // Comments
    { regex: /^(#.*)$/, className: "text-zinc-400 italic" },
    { regex: /(#[^"']*$)/, className: "text-zinc-400 italic" },
    // Triple-quoted strings
    { regex: /(""".*?"""|'''.*?''')/s, className: "text-emerald-500" },
    // Double-quoted strings
    { regex: /("(?:[^"\\]|\\.)*")/, className: "text-emerald-500" },
    // Single-quoted strings
    { regex: /('(?:[^'\\]|\\.)*')/, className: "text-emerald-500" },
    // f-strings
    { regex: /(f"(?:[^"\\]|\\.)*"|f'(?:[^'\\]|\\.)*')/, className: "text-emerald-500" },
    // Keywords
    {
      regex:
        /\b(def|class|import|from|return|if|elif|else|while|for|in|not|and|or|is|None|True|False|try|except|raise|with|as|yield|break|continue|pass|global|lambda|async|await)\b/,
      className: "text-blue-400 font-medium",
    },
    // Class/function definitions
    {
      regex: /(?<=\b(?:class|def)\s+)(\w+)/,
      className: "text-amber-400 font-medium",
    },
    // Self keyword
    { regex: /\b(self)\b/, className: "text-purple-400" },
    // Decorators
    { regex: /^(\s*@\w+.*)$/, className: "text-amber-400" },
    // Numbers
    { regex: /\b(\d+(?:\.\d+)?)\b/, className: "text-orange-400" },
  ];

  // For whole-line patterns (comments, decorators)
  const trimmed = remaining.trimStart();
  if (trimmed.startsWith("#")) {
    return [
      <span key={0} className="text-zinc-400 italic">
        {line}
      </span>,
    ];
  }
  if (trimmed.startsWith("@")) {
    return [
      <span key={0} className="text-amber-400">
        {line}
      </span>,
    ];
  }
  if (trimmed.startsWith('"""') || trimmed.startsWith("'''")) {
    return [
      <span key={0} className="text-emerald-500">
        {line}
      </span>,
    ];
  }

  // Token-level highlighting
  let pos = 0;
  const chars = line.split("");
  let result = "";

  // Simple approach: highlight keywords, strings, comments inline
  const parts = line.split(
    /(\b(?:def|class|import|from|return|if|elif|else|while|for|in|not|and|or|is|None|True|False|try|except|raise|with|as|yield|break|continue|pass|global|lambda|async|await|self)\b|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|f"(?:[^"\\]|\\.)*"|f'(?:[^'\\]|\\.)*'|#.*$|\b\d+(?:\.\d+)?\b)/
  );

  const keywordSet = new Set([
    "def", "class", "import", "from", "return", "if", "elif", "else",
    "while", "for", "in", "not", "and", "or", "is", "None", "True",
    "False", "try", "except", "raise", "with", "as", "yield", "break",
    "continue", "pass", "global", "lambda", "async", "await",
  ]);

  return parts.map((part, idx) => {
    if (!part) return null;

    if (keywordSet.has(part)) {
      return (
        <span key={idx} className="text-blue-400 font-medium">
          {part}
        </span>
      );
    }
    if (part === "self") {
      return (
        <span key={idx} className="text-purple-400">
          {part}
        </span>
      );
    }
    if (part.startsWith("#")) {
      return (
        <span key={idx} className="text-zinc-400 italic">
          {part}
        </span>
      );
    }
    if (
      (part.startsWith('"') && part.endsWith('"')) ||
      (part.startsWith("'") && part.endsWith("'")) ||
      (part.startsWith('f"') && part.endsWith('"')) ||
      (part.startsWith("f'") && part.endsWith("'"))
    ) {
      return (
        <span key={idx} className="text-emerald-500">
          {part}
        </span>
      );
    }
    if (/^\d+(?:\.\d+)?$/.test(part)) {
      return (
        <span key={idx} className="text-orange-400">
          {part}
        </span>
      );
    }
    return <span key={idx}>{part}</span>;
  });
}

export function SourceViewer({ source, filename }: SourceViewerProps) {
  const [open, setOpen] = useState(false);
  const t = useTranslations("version");

  const lines = useMemo(() => source.split("\n"), [source]);

  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-3"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t("view_source")}
          </span>
          <span className="font-mono text-xs text-zinc-400">{filename}</span>
        </div>
        <ChevronDown
          size={16}
          className={cn(
            "text-zinc-400 transition-transform duration-200",
            open && "rotate-180"
          )}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="overflow-x-auto border-t border-zinc-200 bg-zinc-950 dark:border-zinc-700">
              <pre className="p-2 text-[10px] leading-4 sm:p-4 sm:text-xs sm:leading-5">
                <code>
                  {lines.map((line, i) => (
                    <div key={i} className="flex">
                      <span className="mr-2 inline-block w-6 shrink-0 select-none text-right text-zinc-600 sm:mr-4 sm:w-8">
                        {i + 1}
                      </span>
                      <span className="text-zinc-200">
                        {highlightLine(line)}
                      </span>
                    </div>
                  ))}
                </code>
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
