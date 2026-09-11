import { Boxes, FlaskConical, Cpu, Activity, Network, ScanLine } from 'lucide-react';

export const NAVIGATION = [
  { id: 'experiment', label: 'Experiment', description: '实验配置', icon: FlaskConical },
  { id: 'llm', label: 'LLM', description: '模型连接', icon: Cpu },
  { id: 'mas', label: 'MAS', description: '工作流', icon: Network },
  { id: 'rl', label: 'RL', description: '训练配置', icon: Boxes },
  { id: 'harness', label: 'Harness', description: '诊断', icon: ScanLine },
  { id: 'monitor', label: 'Monitor', description: '监控', icon: Activity },
] as const;
export type PanelId = (typeof NAVIGATION)[number]['id'];
