"use client";

import { lazy, Suspense } from "react";

const visualizations: Record<
  string,
  React.LazyExoticComponent<React.ComponentType>
> = {
  s01: lazy(() => import("./s01-agent-loop")),
  s02: lazy(() => import("./s02-tool-dispatch")),
  s03: lazy(() => import("./s03-todo-write")),
  s04: lazy(() => import("./s04-subagent")),
  s05: lazy(() => import("./s05-skill-loading")),
  s06: lazy(() => import("./s06-context-compact")),
  s07: lazy(() => import("./s07-task-system")),
  s08: lazy(() => import("./s08-background-tasks")),
  s09: lazy(() => import("./s09-agent-teams")),
  s10: lazy(() => import("./s10-team-protocols")),
  s11: lazy(() => import("./s11-autonomous-agents")),
};

export function SessionVisualization({ version }: { version: string }) {
  const Component = visualizations[version];
  if (!Component) return null;
  return (
    <Suspense
      fallback={
        <div className="h-64 animate-pulse rounded-lg bg-zinc-100 dark:bg-zinc-800" />
      }
    >
      <Component />
    </Suspense>
  );
}
