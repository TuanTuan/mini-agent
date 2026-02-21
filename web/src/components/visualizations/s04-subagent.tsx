"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Phase = "idle" | "delegate" | "child_working" | "summary_back" | "done";

interface Msg {
  text: string;
  color: string;
}

export default function Subagent() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [parentMsgs, setParentMsgs] = useState<Msg[]>([
    { text: "user: Build login + tests", color: "bg-blue-500" },
    { text: "assistant: I'll handle login", color: "bg-zinc-600" },
  ]);
  const [childMsgs, setChildMsgs] = useState<Msg[]>([]);
  const [childFaded, setChildFaded] = useState(false);

  const reset = useCallback(() => {
    setPhase("idle");
    setParentMsgs([
      { text: "user: Build login + tests", color: "bg-blue-500" },
      { text: "assistant: I'll handle login", color: "bg-zinc-600" },
    ]);
    setChildMsgs([]);
    setChildFaded(false);
  }, []);

  useEffect(() => {
    if (phase === "idle") {
      const t = setTimeout(() => setPhase("delegate"), 1500);
      return () => clearTimeout(t);
    }
    if (phase === "delegate") {
      setChildMsgs([{ text: "task: Create test file", color: "bg-purple-500" }]);
      const t = setTimeout(() => setPhase("child_working"), 1200);
      return () => clearTimeout(t);
    }
    if (phase === "child_working") {
      const t1 = setTimeout(() => {
        setChildMsgs((p) => [
          ...p,
          { text: "tool_use: write_file", color: "bg-amber-500" },
        ]);
      }, 800);
      const t2 = setTimeout(() => {
        setChildMsgs((p) => [
          ...p,
          { text: "tool_result: test.py created", color: "bg-emerald-500" },
        ]);
      }, 1600);
      const t3 = setTimeout(() => setPhase("summary_back"), 2400);
      return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
    }
    if (phase === "summary_back") {
      setChildFaded(true);
      setParentMsgs((p) => [
        ...p,
        { text: "summary: Created test.py (3 tests)", color: "bg-emerald-500" },
      ]);
      const t = setTimeout(() => setPhase("done"), 1500);
      return () => clearTimeout(t);
    }
    if (phase === "done") {
      const t = setTimeout(reset, 3000);
      return () => clearTimeout(t);
    }
  }, [phase, reset]);

  const renderMsgList = (msgs: Msg[], faded: boolean) => (
    <div className="space-y-1">
      {msgs.map((msg, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: faded ? 0.3 : 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className={`rounded px-2 py-1 text-[10px] font-medium text-white ${msg.color}`}
        >
          {msg.text}
        </motion.div>
      ))}
    </div>
  );

  return (
    <section className="space-y-4">
      <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
        Subagent Context Isolation
      </h2>
      <div className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
        <div className="mb-2 flex justify-end">
          <button
            onClick={reset}
            className="rounded bg-zinc-200 px-2 py-0.5 text-xs hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
          >
            Replay
          </button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-blue-200 p-3 dark:border-blue-800">
            <div className="mb-2 text-xs font-semibold text-blue-600 dark:text-blue-400">
              Parent Agent
            </div>
            <span className="mb-1 block font-mono text-[10px] text-zinc-400">
              messages[]
            </span>
            {renderMsgList(parentMsgs, false)}
          </div>
          <div className="rounded-lg border border-purple-200 p-3 dark:border-purple-800">
            <div className="mb-2 text-xs font-semibold text-purple-600 dark:text-purple-400">
              Child Subagent
            </div>
            <span className="mb-1 block font-mono text-[10px] text-zinc-400">
              messages[] (fresh)
            </span>
            {childMsgs.length > 0
              ? renderMsgList(childMsgs, childFaded)
              : <span className="text-[10px] text-zinc-400">empty</span>}
            <AnimatePresence>
              {childFaded && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-2 text-[10px] text-zinc-400 italic"
                >
                  context discarded
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {phase === "delegate" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 text-center text-xs text-purple-500"
          >
            delegating task...
          </motion.div>
        )}
        {phase === "summary_back" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 text-center text-xs text-emerald-500"
          >
            summary flows back to parent
          </motion.div>
        )}
      </div>
    </section>
  );
}
