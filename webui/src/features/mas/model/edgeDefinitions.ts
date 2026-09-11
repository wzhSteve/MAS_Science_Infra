import { CornerDownLeft, MessageSquare, Route, Wrench, CircleAlert, type LucideIcon } from 'lucide-react';
import type { EdgeKind } from '../types';

export const EDGE_KINDS: readonly EdgeKind[] = ['message', 'route', 'feedback', 'tool_call'];
export const KNOWN_TOOLS = ['web_search', 'wikipedia_search', 'execute_python'];

interface EdgeDefinition {
  family: 'tool' | 'agent';
  title: string;
  label: string;
  description: string;
  icon: LucideIcon;
  color: string;
  width: number;
  dash?: string;
  directional: boolean;
}

const definitions: Record<EdgeKind, EdgeDefinition> = {
  tool_call: {
    family: 'tool', title: '工具调用', label: '按需调用', icon: Wrench,
    description: '向 Agent 提供工具能力，由 Agent 自行决定是否调用。',
    color: '#8291a5', width: 1.5, dash: '6 5', directional: false,
  },
  message: {
    family: 'agent', title: '传递信息', label: '传递信息', icon: MessageSquare,
    description: '将上游输出交给下游 Agent，当前不是广播。',
    color: '#6986aa', width: 1.5, directional: true,
  },
  route: {
    family: 'agent', title: '任务路由', label: '任务路由', icon: Route,
    description: '指定后续任务处理的 Agent，当前不是条件分支。',
    color: '#506fc2', width: 2, directional: true,
  },
  feedback: {
    family: 'agent', title: '反馈', label: '反馈', icon: CornerDownLeft,
    description: '由验证方根据结果决定是否返回处理。',
    color: '#a57d39', width: 1.5, dash: '3 3', directional: true,
  },
};

export function isEdgeKind(kind: string): kind is EdgeKind {
  return EDGE_KINDS.some((value) => value === kind);
}

export function edgeDefinition(kind: string): EdgeDefinition {
  return isEdgeKind(kind) ? definitions[kind] : {
    family: 'agent', title: '未知关系', label: kind, description: '请选择当前支持的关系类型。',
    icon: CircleAlert, color: '#b63737', width: 1.5, directional: true,
  };
}
