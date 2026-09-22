import { memo } from 'react';
import { datasetResourcesApi } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import { useTrainingConfig } from '../../../app/providers/TrainingConfigProvider';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Button } from '../../../shared/ui/button';
import { Select } from '../../../shared/ui/select';

export const DatasetSelection = memo(function DatasetSelection({ active, onManage }: { active: boolean; onManage: () => void }) {
  const catalog = usePollingResource('training-datasets', datasetResourcesApi.list, undefined, active);
  const { draft } = useTrainingConfig();
  return <>
    {(['train_files', 'val_files'] as const).map((key, index) => {
      const current = draft.rl.data?.[key];
      const value = typeof current === 'string' ? current : '';
      return <FormField key={key} label={index === 0 ? '训练集' : '验证集'}>
        <Select value={value} aria-label={index === 0 ? '选择训练集' : '选择验证集'} onChange={event => {
          const next = event.target.value;
          if (next === '__manage__') { onManage(); return; }
          draft.patch(rl => ({ ...rl, data: { ...rl.data, [key]: next } }));
        }}>
          <option value="" disabled>{Array.isArray(current) ? '已配置多个文件' : '选择数据集'}</option>
          {value && !catalog.data?.items.some(item => item.path === value) &&
            <option value={value}>{value.split(/[\\/]/).pop()}</option>}
          {catalog.data?.items.map(item => <option key={item.id} value={item.path}>{item.name}</option>)}
          <option value="__manage__">在模型与数据中管理…</option>
        </Select>
      </FormField>;
    })}
    {catalog.error && <InlineNotice tone="danger">{catalog.error}<Button size="sm" onClick={() => void catalog.refresh()}>重试</Button></InlineNotice>}
  </>;
});
