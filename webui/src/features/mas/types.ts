import type { Edge, Node } from '@xyflow/react';
import type { AgentKind, Config, RouterStrategy, SamplingOpportunity } from '../../shared/api/types';

export type CanvasMode = 'workflow' | 'sampling';
export type SamplingCanvasState = 'available' | 'configured' | 'compatibility' | 'unavailable';

export type GraphNodeData = Record<string, unknown> & {
  label?: string;
  kind?: AgentKind;
  role?: string;
  skills?: string[];
  tools?: string[];
  memory_scope?: string;
  system_prompt?: string;
  model?: string;
  model_name?: string;
  effective_model_name?: string;
  model_source?: 'local' | 'api' | 'unknown';
  model_inherited?: boolean;
  model_available?: boolean;
  model_trainable?: boolean;
  trainable?: boolean;
  profile?: Config;
  meta?: Config;
  entry?: boolean;
  verify?: string | null;
  max_feedback_hops?: number;
  candidates?: string[];
  strategy?: RouterStrategy;
  scorer?: string | null;
  backend?: string;
  llm_required?: boolean;
  description?: string;
  branchCount?: number;
  canvasMode?: CanvasMode;
  samplingState?: SamplingCanvasState;
  issue?: string;
  related?: boolean;
};

export type EdgeKind = 'tool_call' | 'message' | 'feedback' | 'route' | 'sample_barrier';
export type HandleId = 'top' | 'right' | 'bottom' | 'left';
export type GraphNodeKind = 'agent' | 'tool' | 'router';
export interface GraphNodePreset {
  nodeType: GraphNodeKind;
  id: string;
  agentKind?: AgentKind;
  role?: string;
  backend?: string;
  llmRequired?: boolean;
  description?: string;
}
export type GraphNode = Node<GraphNodeData, GraphNodeKind>;
export type GraphEdge = Edge<{
  kind: EdgeKind;
  meta?: Config;
  lane?: number;
  issue?: string;
  canvasMode?: CanvasMode;
  samplingState?: SamplingCanvasState;
  samplingOpportunity?: SamplingOpportunity;
  samplingSelected?: boolean;
  samplingHovered?: boolean;
}, 'workflow'>;
export type EdgePatch = Partial<Pick<GraphEdge, 'source' | 'target' | 'sourceHandle' | 'targetHandle'>> & { kind?: EdgeKind };
export type GraphSelection = { kind: 'node' | 'edge'; id: string } | null;
export interface TraceFocusRequest {
  token: number;
  agentId: string;
  agentExecutionId?: string;
  toolName?: string;
}

export type SelectedGraphNode = {
  id: string;
  type?: string;
  data: GraphNodeData;
};
