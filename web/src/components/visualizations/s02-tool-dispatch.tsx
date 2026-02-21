"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

const TOOLS = [
  { name: "bash", desc: "Execute shell commands" },
  { name: "read_file", desc: "Read file contents" },
  { name: "write_file", desc: "Create or overwrite a file" },
  { name: "edit_file", desc: "Apply targeted edits" },
];

export default function ToolDispatch() {
  const [activeIndex, setActiveIndex] = useState(-1);

  useEffect(() => {
    let idx = 0;
    const timer = setInterval(() => {
      setActiveIndex(idx % TOOLS.length);
      idx++;
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  const activeTool = activeIndex >= 0 ? TOOLS[activeIndex] : null;

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Tool Dispatch Map
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-4 flex items-center gap-2">
          <span className="text-xs text-zinc-500">Incoming tool_use:</span>
          {activeTool && (
            <motion.code
              key={activeTool.name}
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900/40 dark:text-blue-300"
            >
              {`{ name: "${activeTool.name}" }`}
            </motion.code>
          )}
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          {TOOLS.map((tool, i) => {
            const isActive = i === activeIndex;
            return (
              <motion.div
                key={tool.name}
                animate={{
                  borderColor: isActive ? "#3b82f6" : "transparent",
                  backgroundColor: isActive
                    ? "rgba(59,130,246,0.06)"
                    : "transparent",
                }}
                transition={{ duration: 0.3 }}
                className="relative rounded-lg border-2 p-3"
              >
                <div className="flex items-center gap-2">
                  {isActive && (
                    <motion.div
                      layoutId="dispatch-arrow"
                      className="h-2 w-2 rounded-full bg-blue-500"
                    />
                  )}
                  <code className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
                    {tool.name}
                  </code>
                </div>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  {tool.desc}
                </p>
                {isActive && (
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 1.8 }}
                    className="absolute bottom-0 left-0 h-0.5 rounded-b-lg bg-blue-500"
                  />
                )}
              </motion.div>
            );
          })}
        </div>

        <div className="mt-3 text-center">
          <span className="text-xs text-zinc-400">
            Auto-cycling through dispatch map
          </span>
        </div>
      </div>
    </section>
  );
}
