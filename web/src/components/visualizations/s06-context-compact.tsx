"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";

interface Stage {
  name: string;
  label: string;
  description: string;
  tokensBefore: number;
  tokensAfter: number;
  color: string;
}

const STAGES: Stage[] = [
  {
    name: "micro",
    label: "Micro-Compact",
    description: "Old tool_results replaced with short summaries",
    tokensBefore: 95000,
    tokensAfter: 72000,
    color: "bg-amber-500",
  },
  {
    name: "auto",
    label: "Auto-Compact",
    description: "Token threshold crossed, full conversation compressed",
    tokensBefore: 72000,
    tokensAfter: 28000,
    color: "bg-blue-500",
  },
  {
    name: "manual",
    label: "/compact",
    description: "User triggers explicit compression",
    tokensBefore: 28000,
    tokensAfter: 8000,
    color: "bg-emerald-500",
  },
];

const MAX_TOKENS = 100000;

export default function ContextCompact() {
  const [activeStage, setActiveStage] = useState(-1);

  const currentTokens =
    activeStage < 0
      ? STAGES[0].tokensBefore
      : STAGES[activeStage].tokensAfter;

  const advance = useCallback(() => {
    setActiveStage((prev) => (prev < STAGES.length - 1 ? prev + 1 : -1));
  }, []);

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Three-Layer Context Compression
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-4">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs text-zinc-500">Context window</span>
            <span className="font-mono text-xs text-zinc-500">
              {currentTokens.toLocaleString()} / {MAX_TOKENS.toLocaleString()} tokens
            </span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
            <motion.div
              animate={{ width: `${(currentTokens / MAX_TOKENS) * 100}%` }}
              transition={{ duration: 0.6 }}
              className={`h-full rounded-full ${
                currentTokens > 80000
                  ? "bg-red-500"
                  : currentTokens > 40000
                    ? "bg-amber-500"
                    : "bg-emerald-500"
              }`}
            />
          </div>
        </div>

        <div className="space-y-2">
          {STAGES.map((stage, i) => {
            const isActive = i === activeStage;
            const isPast = i < activeStage;
            const isFuture = i > activeStage;
            return (
              <motion.div
                key={stage.name}
                animate={{
                  opacity: isFuture && activeStage >= 0 ? 0.4 : 1,
                }}
                className={`rounded-lg border p-3 ${
                  isActive
                    ? "border-blue-400 dark:border-blue-600"
                    : isPast
                      ? "border-emerald-300 dark:border-emerald-700"
                      : "border-zinc-200 dark:border-zinc-700"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      isPast ? "bg-emerald-500" : isActive ? stage.color : "bg-zinc-300 dark:bg-zinc-600"
                    }`}
                  />
                  <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                    Stage {i + 1}: {stage.label}
                  </span>
                  {isPast && (
                    <span className="ml-auto text-[10px] text-emerald-600 dark:text-emerald-400">
                      done
                    </span>
                  )}
                </div>
                <p className="mt-1 pl-4 text-xs text-zinc-500">
                  {stage.description}
                </p>
                {(isActive || isPast) && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-1 pl-4 font-mono text-[10px] text-zinc-400"
                  >
                    {stage.tokensBefore.toLocaleString()} {"->"} {stage.tokensAfter.toLocaleString()} tokens
                  </motion.div>
                )}
              </motion.div>
            );
          })}
        </div>

        <div className="mt-3 text-center">
          <button
            onClick={advance}
            className="rounded-md bg-zinc-200 px-4 py-1.5 text-xs font-medium hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
          >
            {activeStage < 0
              ? "Start Compression"
              : activeStage < STAGES.length - 1
                ? "Next Stage"
                : "Reset"}
          </button>
        </div>
      </div>
    </section>
  );
}
