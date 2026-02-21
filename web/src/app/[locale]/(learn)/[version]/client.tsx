"use client";

import { ArchDiagram } from "@/components/architecture/arch-diagram";
import { WhatsNew } from "@/components/diff/whats-new";
import { DesignDecisions } from "@/components/architecture/design-decisions";
import { DocRenderer } from "@/components/docs/doc-renderer";
import { SourceViewer } from "@/components/code/source-viewer";
import { AgentLoopSimulator } from "@/components/simulator/agent-loop-simulator";
import { ExecutionFlow } from "@/components/architecture/execution-flow";
import { SessionVisualization } from "@/components/visualizations";

interface VersionDetailClientProps {
  version: string;
  diff: {
    from: string;
    to: string;
    newClasses: string[];
    newFunctions: string[];
    newTools: string[];
    locDelta: number;
  } | null;
  source: string;
  filename: string;
}

export function VersionDetailClient({
  version,
  diff,
  source,
  filename,
}: VersionDetailClientProps) {
  return (
    <>
      {/* Agent Loop Simulator */}
      <AgentLoopSimulator version={version} />

      {/* Concept Visualization */}
      <SessionVisualization version={version} />

      {/* Execution Flow Diagram */}
      <ExecutionFlow version={version} />

      {/* Architecture Diagram */}
      <section>
        <h2 className="mb-4 text-xl font-semibold">Architecture</h2>
        <ArchDiagram version={version} />
      </section>

      {/* What's New */}
      {diff && <WhatsNew diff={diff} />}

      {/* Design Decisions */}
      <DesignDecisions version={version} />

      {/* Tutorial Doc */}
      <DocRenderer version={version} />

      {/* Source Code */}
      <SourceViewer source={source} filename={filename} />
    </>
  );
}
