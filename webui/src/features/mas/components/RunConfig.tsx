import { useState, type FormEvent } from 'react';
import { Eye, Play } from 'lucide-react';
import type { useMasDraft } from '../model/useMasDraft';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { Select } from '../../../shared/ui/select';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import type { ModelReadinessState } from '../model/useModelReadiness';

export function RunConfig({ draft, executable, readiness }: {
  draft: ReturnType<typeof useMasDraft>;
  executable: { ok: boolean; reason: string };
  readiness: ModelReadinessState;
}) {
  const [input, setInput] = useState<'demo' | 'parquet'>('demo');
  const [mock, setMock] = useState(true);
  const [count, setCount] = useState('1');
  const [dataCount, setDataCount] = useState('5');
  const [parquet, setParquet] = useState('data/val.parquet');
  const [source, setSource] = useState('gsm8k');
  const [algo, setAlgo] = useState('grpo');
  const amount = Number(input === 'demo' ? count : dataCount);
  const countError = Number.isInteger(amount) && amount > 0 ? undefined : '请输入大于零的整数。';
  const pathError = input === 'parquet' && !parquet.trim() ? '请输入服务端文件路径。' : undefined;
  const valid = !countError && !pathError && Boolean(algo.trim());
  const previewInput = { parquet: parquet.trim(), data_n: Number(dataCount), source: source.trim() };
  const preview = draft.previewData;
  const previewCurrent = preview && preview.input.parquet === previewInput.parquet
    && preview.input.data_n === previewInput.data_n && preview.input.source === previewInput.source;
  const pending = Boolean(draft.pending);
  const readinessError = readiness.loading ? '正在检查本地配置与依赖，请稍候。'
    : readiness.error ? `就绪检查失败：${readiness.error} 请重试检查。`
      : !readiness.data ? '本地就绪状态尚未加载，请刷新检查。' : null;
  const liveError = mock ? null : readinessError || (readiness.data?.ready ? null
    : readiness.data?.blocking_issues.map(issue => issue.message).join('；') || '尚未满足真实运行的前置条件。');
  const parquetError = input !== 'parquet' ? null : readinessError || (readiness.data?.dependencies.parquet.available ? null
    : `parquet 数据读取依赖不可用（模拟执行也需要）：${readiness.data?.dependencies.parquet.hint || '请检查 Control 服务环境。'}`);
  const runBlocked = Boolean(liveError || parquetError);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!valid || pending || !executable.ok || runBlocked) return;
    void draft.collect('collect', {
      mock, algo: algo.trim(),
      ...(input === 'parquet' ? { ...previewInput, n: 1, sequential: true } : { n: amount }),
    });
  };

  return <div className="mas-run-config">
    <form onSubmit={submit}>
      <fieldset disabled={pending} className="mas-run-fields">
        <FormField label="输入数据">
          <Select value={input} onChange={(e) => setInput(e.target.value === 'parquet' ? 'parquet' : 'demo')}>
            <option value="demo">示例任务</option><option value="parquet">parquet 数据集</option>
          </Select>
        </FormField>
        <FormField label="执行方式">
          <Select value={mock ? 'mock' : 'live'} onChange={(e) => setMock(e.target.value === 'mock')}>
            <option value="mock">模拟执行</option><option value="live">真实模型</option>
          </Select>
        </FormField>
        <FormField label={input === 'demo' ? '题目数' : '读取题目数'} error={countError}
          hint={input === 'parquet' ? '筛选后的前 N 条，不是随机抽样。' : '每题采集一条轨迹，与训练的每题采样数独立。'}>
          <Input type="number" min={1} step={1} required value={input === 'demo' ? count : dataCount}
            onChange={(e) => input === 'demo' ? setCount(e.target.value) : setDataCount(e.target.value)} />
        </FormField>
        {input === 'parquet' && <>
          <FormField label="数据文件路径" className="mas-run-path" error={pathError} hint="服务端可访问的路径，不是本地文件上传。">
            <Input value={parquet} required onChange={(e) => setParquet(e.target.value)} className="mono" />
          </FormField>
          <FormField label="数据来源筛选" hint="按文件中的 source 字段筛选；留空读取全部来源。">
            <Input value={source} onChange={(e) => setSource(e.target.value)} placeholder="全部来源" />
          </FormField>
        </>}
      </fieldset>
      <details className="mas-run-advanced">
        <summary>高级配置</summary>
        <FormField label="采集算法" hint="用于生成采集训练信号，不修改 Agent 拓扑或训练配置。" error={!algo.trim() ? '请输入算法。' : undefined}>
          <Input value={algo} disabled={pending} onChange={(e) => setAlgo(e.target.value)} />
        </FormField>
      </details>
      {!mock && <InlineNotice tone="warning">将调用已配置的模型及工具，可能产生费用。</InlineNotice>}
      {liveError && <InlineNotice tone="danger">{liveError}</InlineNotice>}
      {parquetError && parquetError !== liveError && <InlineNotice tone="danger">{parquetError}</InlineNotice>}
      {!executable.ok && <InlineNotice tone="danger">{executable.reason}</InlineNotice>}
      {draft.notice && <InlineNotice tone={draft.notice.tone}>{draft.notice.message}</InlineNotice>}
      <div className="mas-run-submit">
        <Button type="submit" size="sm" variant="primary" disabled={pending || !valid || !executable.ok || runBlocked} loading={draft.running}>
          <Play size={13} aria-hidden="true" />{draft.running ? '正在运行' : mock ? '开始模拟运行' : '开始真实运行'}
        </Button>
        {input === 'parquet' && <Button size="sm" disabled={pending || Boolean(countError || pathError || parquetError)} loading={draft.pending === 'preview'}
          onClick={() => { if (!parquetError) void draft.preview(previewInput); }}><Eye size={14} aria-hidden="true" />预览前 N 条</Button>}
        <span>运行前自动保存 Workflow；保存失败则不执行。</span>
      </div>
    </form>
    {input === 'parquet' && <section className="mas-data-preview" aria-label="数据预览">
      <h3>数据预览 <span>{previewCurrent ? `${preview.data.n} 道题目` : '不调用模型'}</span></h3>
      {!preview ? <p className="mas-console-empty">预览筛选后的题目，再开始运行。</p>
        : !previewCurrent ? <p className="mas-console-empty">输入条件已改变，请重新预览。</p>
        : <div className="mas-preview-list">{(preview.data.tasks || []).map((task, index) =>
          <details key={`${task.id}-${index}`}>
            <summary><code>{task.id}</code><span>{task.source}</span><p>{task.question}</p></summary>
            <p className="mas-preview-question">{task.question}</p>
          </details>)}</div>}
    </section>}
  </div>;
}
