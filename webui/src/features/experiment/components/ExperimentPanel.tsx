import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../../../api/client';
import { experimentApi } from '../api';
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
  const incoming = useMemo(() => ({
    seed: bundle.meta.seed,
    name: bundle.meta.name,
  }), [bundle.meta.seed, bundle.meta.name]);
  const incomingKey = JSON.stringify(incoming);
  const [{ draft, saved }, setDraft] = useState(() => ({
    draft: incoming,
    saved: incomingKey,
  }));
  const lastIncoming = useRef(incomingKey);
  const editRevision = useRef(0);
  const { pending, run } = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    if (pending !== null || lastIncoming.current === incomingKey) return;
    lastIncoming.current = incomingKey;
    setDraft(current => ({
      draft: JSON.stringify(current.draft) === current.saved ? incoming : current.draft,
      saved: incomingKey,
    }));
  }, [incoming, incomingKey, pending]);

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
    const submitted = { ...draft };
    const success = await run('save', async () => {
      await experimentApi.updateExperiment(expId, submitted);
      if (mounted.current) {
        setDraft(current => ({
          draft: editRevision.current === submittedRevision ? submitted : current.draft,
          saved: JSON.stringify(submitted),
        }));
        onReload();
      }
      return 'saved experiment.yaml';
    });
    return success && mounted.current && editRevision.current === submittedRevision;
  };

  useUnsavedChanges('experiment-settings', {
    label: '实验配置', resource: 'meta',
    dirty: JSON.stringify(draft) !== saved,
    busy: pending === 'save', save,
  });

  return <div className="page-stack settings-page">
    <PageHeader title="实验信息" />
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
        <FormField label="name"><Input value={draft.name} onChange={(event) => {
          editRevision.current += 1;
          setDraft(current => ({ ...current, draft: { ...current.draft, name: event.target.value } }));
        }} /></FormField>
        <FormField label="seed"><Input type="number" value={draft.seed} onChange={(event) => {
          editRevision.current += 1;
          setDraft(current => ({ ...current, draft: { ...current.draft, seed: Number(event.target.value) } }));
        }} /></FormField>
      </div>
      <ActionBar>
        <Button onClick={onReload}>重新加载</Button>
      </ActionBar>
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
            return `已创建实验 ${newId}`;
          });
        }}>创建</Button>
      </ActionBar>
    </Section>
  </div>;
}
