export const SETTINGS_SECTIONS = [
  { id: 'model', label: '模型绑定', resource: 'llm', description: '实验默认推理模型，Agent 继承这份连接。' },
  { id: 'data', label: '数据与采样', resource: 'rl', description: '训练数据和采样参数；单题调试不依赖这些配置。' },
  { id: 'training', label: '训练方案', resource: 'rl', description: '算法、训练参数与显式启动；不会修改调试模型。' },
  { id: 'diagnostics', label: '评估与诊断', resource: 'harness', description: '配置检查规则，诊断当前实验的最近一次数据集采集。' },
  { id: 'environment', label: '执行环境', resource: 'rl', description: '训练设备与本地模型服务；远程 API 调试无需本机 GPU。' },
] as const;

export type SettingsSection = typeof SETTINGS_SECTIONS[number]['id'];
