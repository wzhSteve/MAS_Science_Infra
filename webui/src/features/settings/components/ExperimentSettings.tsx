import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Settings2, X } from 'lucide-react';
import type { Bundle, MetaResponse, ModelReadinessResponse } from '../../../shared/api/types';
import { Button } from '../../../shared/ui/button';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { Section } from '../../../shared/components/Section';
import { LogPanel } from '../../../shared/components/LogPanel';
import { useAction } from '../../../shared/hooks/useAction';
import { LLMPanel } from '../../llm/components/LLMPanel';
import { HarnessPanel } from '../../harness/components/HarnessPanel';
import { RlTrainingActions } from '../../rl/components/RlTrainingActions';
import { useAgl, useRuntimeCommands, useTraining } from '../../../app/providers/RuntimeProvider';
import { SETTINGS_SECTIONS, type SettingsSection } from '../model/sections';
import { SettingsStatus, useSettingsStatus } from './SettingsStatus';
import { TrainingSettings } from './TrainingSettings';

function RetainedSettings({ visible, children }: { visible: boolean; children: ReactNode }) {
  const [visited, setVisited] = useState(visible);
  useEffect(() => { if (visible) setVisited(true); }, [visible]);
  return <div hidden={!visible}>{(visible || visited) && children}</div>;
}

const TrainingRuntime = memo(function TrainingRuntime({ onStart, pending }: {
  onStart: () => void; pending: string | null;
}) {
  const training = useTraining();
  const agl = useAgl();
  const { stopTrain } = useRuntimeCommands();
  const [logsOpen, setLogsOpen] = useState(false);
  const refresh = useAction();
  const workflowDirty = useSettingsStatus().some(item => item.resource === 'workflow' && item.dirty);
  return <>
    {workflowDirty && <InlineNotice tone="warning">画布设计尚未保存。训练使用服务端已保存的 Workflow；如需使用新设计，请先点击顶部“保存设计”。</InlineNotice>}
    <p className="field-hint">这里沿用现有训练启动流程，不提供多配置原子快照。调试模型不等于可训练策略。</p>
    {training.error && <InlineNotice tone="warning">训练状态读取失败：{training.error}
      <Button size="sm" onClick={() => void training.refresh()}>重试状态</Button>
    </InlineNotice>}
    <RlTrainingActions trainRunId={training.data?.runId ?? null} trainRunning={training.data?.running ?? false}
      aglOnline={Boolean(agl.data?.ok && !agl.error)} pending={pending} onStart={onStart} onStop={stopTrain} />
    <details className="experiment-settings-logs" open={logsOpen} onToggle={event => setLogsOpen(event.currentTarget.open)}>
      <summary>查看训练日志</summary>
      {logsOpen && <LogPanel title="当前 / 最近训练日志" log={training.data?.log || ''} actions={
        <Button size="sm" loading={refresh.pending !== null} onClick={() => {
          void refresh.run('log', training.refresh);
        }}>刷新日志</Button>
      } />}
    </details>
    {refresh.notice && <InlineNotice tone={refresh.notice.tone}>{refresh.notice.message}</InlineNotice>}
  </>;
});

const renderTrainingRuntime = ({ active, ...props }: { onStart: () => void; pending: string | null; active: boolean }) =>
  active ? <TrainingRuntime {...props} /> : null;

const TrainingConnection = memo(function TrainingConnection(props: {
  expId: string; bundle: Bundle; meta: MetaResponse | null; onReload: () => void;
  active: boolean; section: 'training' | 'data' | 'environment';
}) {
  const { startTrain } = useRuntimeCommands();
  return <TrainingSettings {...props} onStartTrain={startTrain} renderRuntime={renderTrainingRuntime} />;
});

function EnvironmentStatus({ readiness, error }: {
  readiness: ModelReadinessResponse | null; error: string | null;
}) {
  const agl = useAgl();
  return <Section title="执行条件">
    <p className="field-hint">调试依赖与训练服务分开检查，以下状态不会自动启动服务。</p>
    {error && <InlineNotice tone="warning">调试条件暂时无法读取：{error}</InlineNotice>}
    {readiness ? <>
      <dl className="experiment-settings-facts">
        <dt>真实推理依赖</dt><dd>{readiness.dependencies.live.available ? '可用' : '需要准备'}</dd>
        <dt>parquet 读取</dt><dd>{readiness.dependencies.parquet.available ? '可用' : '未就绪，不影响单题调试'}</dd>
        <dt>AGL 服务</dt><dd>{agl.loading ? '检查中' : agl.error ? '状态未知' : agl.data?.ok ? '在线' : '离线，不影响远程 API 调试'}</dd>
      </dl>
      {(['live', 'parquet'] as const).map(group => {
        const dependency = readiness.dependencies[group];
        return !dependency.available && <details key={group}>
          <summary>{group === 'live' ? '准备推理依赖' : '准备数据读取依赖'}</summary>
          <p className="field-hint">{dependency.error || dependency.missing.join('、')}</p>
          <p className="field-hint">{dependency.hint}</p>
        </details>;
      })}
    </> : !error && <p className="field-hint">正在读取本地执行条件…</p>}
  </Section>;
}

export const ExperimentSettings = memo(function ExperimentSettings({
  bundle, meta, onReload, open, active, section, onSectionChange, onClose, onCollect, onDemo,
  readiness, readinessError,
}: {
  bundle: Bundle; meta: MetaResponse | null; onReload: () => void;
  open: boolean; active: boolean; section: SettingsSection; onSectionChange: (section: SettingsSection) => void;
  onClose: () => void; onCollect: () => void; onDemo: () => void; readiness: ModelReadinessResponse | null; readinessError: string | null;
}) {
  const visible = open && active;
  const tabs = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (visible) tabs.current?.querySelector<HTMLButtonElement>('[aria-selected="true"]')?.focus({ preventScroll: true });
  }, [visible]);
  const configureModel = useCallback(() => onSectionChange('model'), [onSectionChange]);
  const current = SETTINGS_SECTIONS.find(item => item.id === section) || SETTINGS_SECTIONS[0];
  const modelVisible = visible && (section === 'model' || section === 'environment');
  const trainingVisible = visible && (section === 'training' || section === 'data' || section === 'environment');
  const trainingSection = section === 'data' || section === 'environment' ? section : 'training';

  return <aside className="mas-panel experiment-settings-panel" hidden={!open} aria-label="实验设置" id="experiment-settings"
    onKeyDown={event => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        event.preventDefault();
        event.stopPropagation();
        onClose();
      }
    }}>
    <div className="mas-panel-heading">
      <div><span className="mas-panel-caption">当前实验 · {bundle.meta.name || bundle.id}</span><h2><Settings2 size={16} />实验设置</h2></div>
      <Button size="sm" variant="ghost" onClick={onClose} aria-label="关闭实验设置"><X size={16} /></Button>
    </div>
    <div ref={tabs} className="experiment-settings-tabs" role="tablist" aria-label="设置分组">
      {SETTINGS_SECTIONS.map((item, index) => <button key={item.id} type="button" role="tab"
        id={`settings-tab-${item.id}`} aria-controls="settings-section-content" aria-selected={section === item.id}
        tabIndex={section === item.id ? 0 : -1} onClick={() => onSectionChange(item.id)} onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? SETTINGS_SECTIONS.length - 1
            : (index + (event.key === 'ArrowRight' ? 1 : -1) + SETTINGS_SECTIONS.length) % SETTINGS_SECTIONS.length;
          onSectionChange(SETTINGS_SECTIONS[next].id);
          document.getElementById(`settings-tab-${SETTINGS_SECTIONS[next].id}`)?.focus();
        }}>{item.label}</button>)}
    </div>
    <div className="mas-panel-body experiment-settings-body" role="tabpanel" id="settings-section-content"
      aria-labelledby={`settings-tab-${section}`}>
      <div className="experiment-settings-intro"><p>{current.description}</p><SettingsStatus resource={current.resource} /></div>
      {section === 'data' && <Section title="独立数据集采集" description="采集参数与训练参数分开，沿用原示例 / parquet 预览和采集能力。">
        <Button size="sm" onClick={onCollect}>打开数据集采集</Button>
      </Section>}
      <RetainedSettings visible={modelVisible}>
        <LLMPanel expId={bundle.id} bundle={bundle} onReload={onReload} embedded
          view={section === 'environment' ? 'environment' : 'connection'} onConfigureModel={configureModel} />
      </RetainedSettings>
      <RetainedSettings visible={trainingVisible}>
        <TrainingConnection expId={bundle.id} bundle={bundle} meta={meta} onReload={onReload}
          section={trainingSection} active={trainingVisible} />
      </RetainedSettings>
      <RetainedSettings visible={visible && section === 'diagnostics'}>
        <HarnessPanel expId={bundle.id} bundle={bundle} meta={meta} onReload={onReload} embedded />
      </RetainedSettings>
      {visible && section === 'environment' && <EnvironmentStatus readiness={readiness} error={readinessError} />}
      {section === 'environment' && <Section title="开发与演示" description="模拟执行只演示 Workflow 和记录结构，不调用真实模型，不验证推理能力。">
        <Button size="sm" onClick={onDemo}>打开模拟演示 · Mock</Button>
      </Section>}
    </div>
    <div className="mas-panel-footnote">分组显式保存，不自动运行。数据、训练和训练环境共用一份训练配置；顶部“保存设计”仅保存 Workflow。</div>
  </aside>;
});
