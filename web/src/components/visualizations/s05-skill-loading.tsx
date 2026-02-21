"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const SKILLS = [
  { name: "/commit", summary: "Create git commits", fullTokens: 320 },
  { name: "/review-pr", summary: "Review pull requests", fullTokens: 480 },
  { name: "/create-pr", summary: "Create pull requests", fullTokens: 410 },
  { name: "/init", summary: "Initialize CLAUDE.md", fullTokens: 260 },
];

const FULL_CONTENT: Record<string, string[]> = {
  "/commit": [
    "1. Run git status + git diff",
    "2. Analyze all staged changes",
    "3. Draft concise commit message",
    "4. Create commit with Co-Authored-By",
  ],
  "/review-pr": [
    "1. Fetch PR diff via gh CLI",
    "2. Analyze changes file by file",
    "3. Check for bugs, style, security",
    "4. Post review comments",
  ],
  "/create-pr": [
    "1. Check branch status and diff",
    "2. Draft PR title and summary",
    "3. Push branch if needed",
    "4. Create PR via gh pr create",
  ],
  "/init": [
    "1. Scan project structure",
    "2. Detect languages and frameworks",
    "3. Generate CLAUDE.md with conventions",
    "4. Include build/test commands",
  ],
};

export default function SkillLoading() {
  const [activeSkill, setActiveSkill] = useState<string | null>(null);

  const baseTokens = 120;
  const activeTokens = activeSkill
    ? baseTokens + (SKILLS.find((s) => s.name === activeSkill)?.fullTokens ?? 0)
    : baseTokens;

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        On-Demand Skill Loading
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-3">
          <div className="mb-1 text-xs font-medium text-zinc-500">
            Layer 1: System Prompt (compact summaries)
          </div>
          <div className="flex flex-wrap gap-2">
            {SKILLS.map((skill) => (
              <button
                key={skill.name}
                onClick={() =>
                  setActiveSkill((prev) =>
                    prev === skill.name ? null : skill.name
                  )
                }
                className={`rounded-md border px-3 py-1.5 text-left transition-colors ${
                  activeSkill === skill.name
                    ? "border-blue-400 bg-blue-50 dark:border-blue-600 dark:bg-blue-900/20"
                    : "border-zinc-200 hover:border-zinc-300 dark:border-zinc-700 dark:hover:border-zinc-600"
                }`}
              >
                <code className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {skill.name}
                </code>
                <div className="text-[10px] text-zinc-500">{skill.summary}</div>
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence mode="wait">
          {activeSkill && FULL_CONTENT[activeSkill] && (
            <motion.div
              key={activeSkill}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="mb-1 text-xs font-medium text-blue-600 dark:text-blue-400">
                Layer 2: Full SKILL.md loaded into context
              </div>
              <div className="rounded-md border border-blue-200 bg-blue-50/50 p-3 dark:border-blue-800 dark:bg-blue-900/10">
                {FULL_CONTENT[activeSkill].map((line, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className="py-0.5 font-mono text-xs text-zinc-700 dark:text-zinc-300"
                  >
                    {line}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-3">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs text-zinc-500">Token usage</span>
            <span className="font-mono text-xs text-zinc-500">
              {activeTokens} tokens
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
            <motion.div
              animate={{ width: `${Math.min((activeTokens / 800) * 100, 100)}%` }}
              transition={{ duration: 0.4 }}
              className="h-full rounded-full bg-blue-500"
            />
          </div>
          <div className="mt-1 text-[10px] text-zinc-400">
            Skills load on demand, not all upfront
          </div>
        </div>
      </div>
    </section>
  );
}
