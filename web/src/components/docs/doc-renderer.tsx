"use client";

import { useState, useMemo } from "react";
import { useLocale, useTranslations } from "@/lib/i18n";
import { ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import docsData from "@/data/generated/docs.json";

interface DocRendererProps {
  version: string;
}

function renderMarkdown(md: string): string {
  let html = md;

  // Code blocks (```...```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
    const escaped = code
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    return `<pre class="overflow-x-auto rounded-lg bg-zinc-950 p-4 text-xs leading-5 text-zinc-200"><code>${escaped}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="rounded bg-zinc-100 px-1.5 py-0.5 text-sm font-mono dark:bg-zinc-800">$1</code>');

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4 class="mt-6 mb-2 text-base font-semibold">$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3 class="mt-8 mb-3 text-lg font-semibold">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 class="mt-10 mb-4 text-xl font-bold">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 class="mt-10 mb-4 text-2xl font-bold">$1</h1>');

  // Bold / Italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Links
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" class="text-blue-600 hover:underline dark:text-blue-400" target="_blank" rel="noopener">$1</a>'
  );

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li class="ml-4 list-disc">$1</li>');
  html = html.replace(/(<li[^>]*>.*<\/li>\n?)+/g, (match) => `<ul class="my-3 space-y-1 text-sm">${match}</ul>`);

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li class="ml-4 list-decimal">$1</li>');

  // Blockquotes
  html = html.replace(
    /^> (.+)$/gm,
    '<blockquote class="border-l-4 border-zinc-300 pl-4 italic text-zinc-500 dark:border-zinc-600 dark:text-zinc-400">$1</blockquote>'
  );

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr class="my-6 border-zinc-200 dark:border-zinc-700" />');

  // Paragraphs: wrap lines that aren't already tagged
  html = html
    .split("\n\n")
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (
        trimmed.startsWith("<h") ||
        trimmed.startsWith("<pre") ||
        trimmed.startsWith("<ul") ||
        trimmed.startsWith("<ol") ||
        trimmed.startsWith("<blockquote") ||
        trimmed.startsWith("<hr") ||
        trimmed.startsWith("<li")
      ) {
        return trimmed;
      }
      return `<p class="my-3 text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">${trimmed.replace(/\n/g, " ")}</p>`;
    })
    .join("\n");

  return html;
}

export function DocRenderer({ version }: DocRendererProps) {
  const [open, setOpen] = useState(true);
  const locale = useLocale();
  const t = useTranslations("version");

  const doc = useMemo(() => {
    // s09/s10/s11 docs may be stored under combined key
    const docVersion = version;
    const match = docsData.find(
      (d: { version: string; locale: string }) =>
        d.version === docVersion && d.locale === locale
    );
    if (match) return match;
    return docsData.find(
      (d: { version: string; locale: string }) =>
        d.version === docVersion && d.locale === "en"
    );
  }, [version, locale]);

  if (!doc) return null;

  const html = useMemo(() => renderMarkdown(doc.content), [doc.content]);

  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-3"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t("tutorial")}
          </span>
          <span className="text-xs text-zinc-400">{doc.title}</span>
        </div>
        <ChevronDown
          size={16}
          className={cn(
            "shrink-0 text-zinc-400 transition-transform duration-200",
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
            <div className="border-t border-zinc-200 px-4 py-4 dark:border-zinc-700 sm:px-6">
              <div
                className="prose-custom"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
