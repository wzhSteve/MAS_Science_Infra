import { useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import { errorMessage } from '../../../shared/api/http';
import type { Bundle } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { LoadingState } from '../../../shared/components/LoadingState';
import { PageHeader } from '../../../shared/components/PageHeader';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';

type Props = {
  expId: string;
  bundle: Bundle | null;
  onReload: () => void;
  setExpId: (id: string) => void;
};

export function ExperimentPanel(props: Props) {
  if (!props.bundle || props.bundle.id !== props.expId) {
    return <LoadingState label="加载实验配置…" />;
  }
  return <ExperimentSettings key={props.expId} {...props} bundle={props.bundle} />;
}

function ExperimentSettings({ expId, bundle, onReload, setExpId }: Props & { bundle: Bundle }) {
  const [list, setList] = useState<string[]>([]);
  const [listError, setListError] = useState('');
  const [newId, setNewId] = useState('');
  const [seed, setSeed] = useState(Number(bundle.meta.seed || 42));
  const [name, setName] = useState(String(bundle.meta.name || ''));
  const [savedSettings, setSavedSettings] = useState(() => JSON.stringify({ seed, name }));
  const editRevision = useRef(0);
  const { pending, notice, run } = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api.listExperiments(controller.signal).then((result) => {
      if (!controller.signal.aborted) {
        setList(result.experiments);
        setListError('');
      }
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setListError(errorMessage(error));
    });
    return () => controller.abort();
  }, [bundle]);

  const save = async () => {
    const submittedRevision = editRevision.current;
    const success = await run('save', async () => {
      await api.putSection(expId, 'meta', { ...bundle.meta, seed, name: name || bundle.meta.name });
      if (mounted.current) {
        setSavedSettings(JSON.stringify({ seed, name }));
        onReload();
      }
      return 'saved experiment.yaml';
    });
    return success && mounted.current && editRevision.current === submittedRevision;
  };

  useUnsavedChanges('experiment-settings', {
    label: '实验配置', resource: 'meta',
    dirty: JSON.stringify({ seed, name }) !== savedSettings,
    busy: pending === 'save', save,
  });

  return <div className="page-stack settings-page">
    <PageHeader title="Experiment" eyebrow="实验配置" description="管理当前实验的名称与随机种子；配置草稿只在显式保存时写入。" />
    <Section title="当前实验" actions={<StatusBadge tone={bundle.executable.ok ? 'success' : 'warning'}>
      {bundle.executable.ok ? 'executable' : 'blocked'}
    </StatusBadge>}>
      <FormField label="当前实验">
        <Select value={expId} onChange={(event) => setExpId(event.target.value)}>
          {!list.includes(expId) && <option value={expId}>{expId}</option>}
          {list.map((id) => <option key={id} value={id}>{id}</option>)}
        </Select>
      </FormField>
      {listError && <InlineNotice tone="danger">实验列表刷新失败：{listError}</InlineNotice>}
      <div className="form-grid">
        <FormField label="name"><Input value={name} onChange={(event) => { editRevision.current += 1; setName(event.target.value); }} /></FormField>
        <FormField label="seed"><Input type="number" value={seed} onChange={(event) => { editRevision.current += 1; setSeed(Number(event.target.value)); }} /></FormField>
      </div>
      <ActionBar>
        <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={() => { void save(); }}>保存 experiment.yaml</Button>
        <Button onClick={onReload}>重新加载</Button>
      </ActionBar>
      <p className="field-hint">重新加载服务器配置，不覆盖当前实验的未保存草稿。</p>
      <dl className="grid gap-2 text-sm">
        <div className="grid gap-1"><dt className="field-hint">path</dt><dd className="mono m-0 break-all">{bundle.path}</dd></div>
        <div className="grid gap-1"><dt className="field-hint">topology</dt><dd className="mono m-0">{bundle.workflow.topology}</dd></div>
      </dl>
    </Section>
    <Section title="新建实验" description="以随机种子 42 创建独立实验，名称与实验 ID 相同。">
      <div className="max-w-sm">
        <FormField label="实验 ID"><Input placeholder="exp id" value={newId} onChange={(event) => setNewId(event.target.value)} /></FormField>
      </div>
      <ActionBar>
        <Button loading={pending === 'create'} disabled={pending !== null} onClick={() => {
          void run('create', async () => {
            await api.createExperiment(newId, 42, newId);
            if (!mounted.current) return;
            setExpId(newId);
            setNewId('');
            onReload();
          });
        }}>创建</Button>
      </ActionBar>
    </Section>
    {notice && <InlineNotice tone={notice.tone}>{notice.message}</InlineNotice>}
  </div>;
}
