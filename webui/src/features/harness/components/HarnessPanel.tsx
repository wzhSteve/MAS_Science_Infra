import { useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { Bundle, Hypothesis, MetaResponse } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { DataTable } from '../../../shared/components/DataTable';
import { EmptyState } from '../../../shared/components/EmptyState';
import { LoadingState } from '../../../shared/components/LoadingState';
import { PageHeader } from '../../../shared/components/PageHeader';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { Button } from '../../../shared/ui/button';
import { Checkbox } from '../../../shared/ui/checkbox';

type Props = { expId: string; bundle: Bundle | null; meta: MetaResponse | null; onReload: () => void; embedded?: boolean };

export function HarnessPanel(props: Props) {
  if (!props.bundle || props.bundle.id !== props.expId) return <LoadingState label="加载 harness.yaml…" />;
  return <HarnessSettings key={props.expId} {...props} bundle={props.bundle} />;
}

function HarnessSettings({ expId, bundle, meta, onReload, embedded = false }: Props & { bundle: Bundle }) {
  const [selected, setSelected] = useState<string[]>(() => [...(bundle.harness.plugins || [])]);
  const [savedPlugins, setSavedPlugins] = useState(() => JSON.stringify(selected));
  const editRevision = useRef(0);
  const dirty = JSON.stringify(selected) !== savedPlugins;
  const [hypotheses, setHypotheses] = useState<Hypothesis[] | null>(null);
  const { pending, run } = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const persist = async () => {
    await api.putSection(expId, 'harness', { plugins: selected });
    if (mounted.current) setSavedPlugins(JSON.stringify(selected));
  };

  const save = async () => {
    const submittedRevision = editRevision.current;
    const success = await run('save', async () => {
      await persist();
      if (mounted.current) onReload();
      return 'saved harness.yaml';
    });
    return success && mounted.current && editRevision.current === submittedRevision;
  };

  useUnsavedChanges('harness-settings', {
    label: 'Harness 配置', resource: 'harness',
    dirty, busy: pending !== null, save,
  });

  return <div className={`page-stack ${embedded ? 'settings-embedded' : 'settings-page'}`}>
    {!embedded && <PageHeader title="Harness" eyebrow="诊断插件" description="选择诊断插件，检查当前实验已有的轨迹产物。" />}
    <Section title="检查规则" description="读取当前实验已有的 collect.json（可由命令行生成），不是单题调试或训练的自动评估。"
      actions={<StatusBadge tone={dirty ? 'warning' : 'neutral'}>{pending === 'save' ? '保存中' : dirty ? '未保存' : '已保存配置'}</StatusBadge>}>
      {!meta && <p className="field-hint">尚未取得插件列表，请等待加载或通过页面上方的提示重试。</p>}
      <div className="grid gap-3 sm:grid-cols-2">
        {(meta?.harness_plugins || []).map((plugin) => {
          const stub = meta?.stub_harness.includes(plugin) || false;
          return <label className="checkbox-label min-w-0" key={plugin}>
            <Checkbox disabled={stub} checked={selected.includes(plugin)} onChange={() => {
              if (!stub) {
                editRevision.current += 1;
                setSelected((current) => current.includes(plugin) ? current.filter((item) => item !== plugin) : [...current, plugin]);
              }
            }} />
            <span className="mono break-all">{plugin}</span>
            {stub && <StatusBadge tone="warning">stub · 不可用</StatusBadge>}
          </label>;
        })}
      </div>
      {meta && meta.harness_plugins.length === 0 && <EmptyState title="暂无可选插件" />}
      <ActionBar>
        {!embedded && <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={() => { void save(); }}>保存</Button>}
        <Button loading={pending === 'diagnose'} disabled={pending !== null} onClick={() => {
          void run('diagnose', async () => {
            await persist();
            if (!mounted.current) return;
            const result = await api.diagnose(expId);
            if (mounted.current) setHypotheses(result.hypotheses || []);
            return `diagnose n=${result.n}`;
          });
        }} title="保存检查规则后，诊断当前实验已有的 collect.json">保存并诊断已有产物</Button>
      </ActionBar>
    </Section>
    <Section title="诊断结果">
      {hypotheses === null ? <EmptyState title="尚未执行诊断">已有 collect.json 时，可选择插件并执行诊断。</EmptyState> :
        hypotheses.length === 0 ? <EmptyState title="没有 hypotheses">本次诊断未产生假设。</EmptyState> :
          <DataTable aria-label="Harness hypotheses">
            <thead><tr><th scope="col">plugin</th><th scope="col">event_id</th><th scope="col">message</th></tr></thead>
            <tbody>{hypotheses.map((hypothesis, index) => <tr key={`${hypothesis.plugin}-${hypothesis.event_id || ''}-${index}`}>
              <td className="mono">{hypothesis.plugin}</td><td className="mono">{hypothesis.event_id || ''}</td><td className="whitespace-pre-wrap break-words">{hypothesis.message}</td>
            </tr>)}</tbody>
          </DataTable>}
    </Section>
  </div>;
}
