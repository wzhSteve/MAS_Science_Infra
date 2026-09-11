import { useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { Bundle, Hypothesis, MetaResponse } from '../../../shared/api/types';
import { ActionBar } from '../../../shared/components/ActionBar';
import { DataTable } from '../../../shared/components/DataTable';
import { EmptyState } from '../../../shared/components/EmptyState';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { LoadingState } from '../../../shared/components/LoadingState';
import { PageHeader } from '../../../shared/components/PageHeader';
import { Section } from '../../../shared/components/Section';
import { StatusBadge } from '../../../shared/components/StatusBadge';
import { useAction } from '../../../shared/hooks/useAction';
import { Button } from '../../../shared/ui/button';
import { Checkbox } from '../../../shared/ui/checkbox';

type Props = { expId: string; bundle: Bundle | null; meta: MetaResponse | null; onReload: () => void };

export function HarnessPanel(props: Props) {
  if (!props.bundle || props.bundle.id !== props.expId) return <LoadingState label="加载 harness.yaml…" />;
  return <HarnessSettings key={props.expId} {...props} bundle={props.bundle} />;
}

function HarnessSettings({ expId, bundle, meta, onReload }: Props & { bundle: Bundle }) {
  const [selected, setSelected] = useState<string[]>(() => [...(bundle.harness.plugins || [])]);
  const [hypotheses, setHypotheses] = useState<Hypothesis[] | null>(null);
  const { pending, notice, run } = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  return <div className="page-stack settings-page">
    <PageHeader title="Harness" eyebrow="诊断插件" description="选择诊断插件，并对最近一次 Collect 的结果运行诊断。" />
    <Section title="插件配置" description="stub 插件尚未实现，不可启用。">
      <div className="grid gap-3 sm:grid-cols-2">
        {(meta?.harness_plugins || []).map((plugin) => {
          const stub = meta?.stub_harness.includes(plugin) || false;
          return <label className="checkbox-label min-w-0" key={plugin}>
            <Checkbox disabled={stub} checked={selected.includes(plugin)} onChange={() => {
              if (!stub) setSelected((current) => current.includes(plugin) ? current.filter((item) => item !== plugin) : [...current, plugin]);
            }} />
            <span className="mono break-all">{plugin}</span>
            {stub && <StatusBadge tone="warning">stub · 不可用</StatusBadge>}
          </label>;
        })}
      </div>
      {meta && meta.harness_plugins.length === 0 && <EmptyState title="暂无可选插件" />}
      <ActionBar>
        <Button variant="primary" loading={pending === 'save'} disabled={pending !== null} onClick={() => {
          void run('save', async () => {
            await api.putSection(expId, 'harness', { plugins: selected });
            if (mounted.current) onReload();
            return 'saved harness.yaml';
          });
        }}>保存</Button>
        <Button loading={pending === 'diagnose'} disabled={pending !== null} onClick={() => {
          void run('diagnose', async () => {
            await api.putSection(expId, 'harness', { plugins: selected });
            if (!mounted.current) return;
            const result = await api.diagnose(expId);
            if (mounted.current) setHypotheses(result.hypotheses || []);
            return `diagnose n=${result.n}`;
          });
        }}>保存并诊断最近 Collect</Button>
      </ActionBar>
      {notice && <InlineNotice tone={notice.tone}>{notice.message}</InlineNotice>}
    </Section>
    <Section title="诊断结果" description="最近一次成功诊断返回的 hypotheses。">
      {hypotheses === null ? <EmptyState title="尚未执行诊断">选择插件后，保存并诊断最近 Collect。</EmptyState> :
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
