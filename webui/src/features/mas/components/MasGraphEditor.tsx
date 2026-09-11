import { memo, useMemo } from 'react';
import type { Palette, RlConfig, WorkflowSpec } from '../../../shared/api/types';
import type { RlSettingsSlot } from '../types';
import { executableInfo } from '../model/workflowGraph';
import { useGraphEditor } from '../model/useGraphEditor';
import { GraphToolbar } from './GraphToolbar';
import { GraphPalette } from './GraphPalette';
import { GraphCanvas } from './GraphCanvas';
import { NodeInspector } from './NodeInspector';

type Props = {
  workflow: WorkflowSpec;
  palette: Palette;
  onChange: (workflow: WorkflowSpec) => void;
  rl: RlConfig;
  onRlPatch: (patch: Partial<RlConfig>) => void;
  onRlSave: () => void | Promise<void>;
  savingRl?: boolean;
  renderRlSettings?: RlSettingsSlot;
};

export default memo(function MasGraphEditor({ workflow, palette, onChange, rl, onRlPatch, onRlSave, savingRl, renderRlSettings }: Props) {
  const graph = useGraphEditor(workflow, onChange);
  const executable = useMemo(() => executableInfo(workflow), [workflow]);
  const entryId = workflow.entry_agent || 'hub';
  const rlSettings = useMemo(() => renderRlSettings?.({
    rl, onPatch: onRlPatch, onSave: onRlSave, saving: savingRl,
  }), [renderRlSettings, rl, onRlPatch, onRlSave, savingRl]);

  return <div className="mas-workbench">
    <GraphToolbar palette={palette} executable={executable} entryId={entryId}
      edgeKind={graph.edgeKind} onEdgeKindChange={graph.setEdgeKind} onTemplate={graph.applyTemplate} />
    <GraphPalette palette={palette} nodeCount={graph.nodes.length} onAdd={graph.addNode} />
    <div className="mas-graph-layout">
      <GraphCanvas nodes={graph.nodes} edges={graph.edges}
        onNodesChange={graph.onNodesChange} onEdgesChange={graph.onEdgesChange}
        onConnect={graph.onConnect} onNodeClick={graph.onNodeClick}
        onPaneClick={graph.onPaneClick} onNodeDragStop={graph.onNodeDragStop}
        selectedId={graph.selected?.id} onAdd={graph.addNode} onEntry={graph.setEntry} />
      <NodeInspector selected={graph.selected} palette={palette} entryId={entryId}
        onPatch={graph.updateSelected} onEntry={graph.setEntry} rlSettings={rlSettings} />
    </div>
  </div>;
});
