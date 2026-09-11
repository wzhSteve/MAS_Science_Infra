import type { ReactNode } from 'react';
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
};

export type SelectedGraphNode = {
  id: string;
  type?: string;
  data: GraphNodeData;
};
