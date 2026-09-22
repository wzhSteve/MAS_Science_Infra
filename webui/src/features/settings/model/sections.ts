export const TRAINING_SECTIONS = [
  { id: 'model', label: '训练对象' },
  { id: 'data', label: '数据与奖励' },
  { id: 'training', label: '训练策略' },
  { id: 'environment', label: '执行资源' },
] as const;
export type TrainingSection = typeof TRAINING_SECTIONS[number]['id'];
export type SettingsSection = TrainingSection | 'inference' | 'diagnostics';
export const isTrainingSection = (section: SettingsSection | null | undefined): section is TrainingSection =>
  TRAINING_SECTIONS.some(item => item.id === section);
