import type { ReactNode } from 'react';
import type { Edge, Node } from '@xyflow/react';
import type { RlConfig } from '../../shared/api/types';

export type RlSettingsSlot = (args: {
  rl: RlConfig;
  onPatch: (patch: Partial<RlConfig>) => void;
  onSave: () => void | Promise<void>;
  saving?: boolean;
}) => ReactNode;

export type GraphNodeData = Record<string, unknown> & {
  label?: string;
  role?: string;
  skills?: string[];
  tools?: string[];
  system_prompt?: string;
  trainable?: boolean;
  entry?: boolean;
  verify?: string | null;
  max_feedback_hops?: number;
  issue?: string;
};

export type GraphNode = Node<GraphNodeData, 'agent' | 'tool'>;
export type GraphEdge = Edge<{ kind: string }>;
export type GraphSelection = { kind: 'node' | 'edge'; id: string } | null;
export type EditorPanel = 'settings' | null;

export type SelectedGraphNode = {
  id: string;
  type?: string;
  data: GraphNodeData;
};
