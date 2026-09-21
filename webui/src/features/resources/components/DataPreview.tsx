import { memo, useEffect, useRef, useState, type FormEvent } from 'react';
import { Eye } from 'lucide-react';
import { masApi, type SampleDataInput, type SampleDataResponse } from '../../mas/api';
import { errorMessage } from '../../../shared/api/http';
import { Button } from '../../../shared/ui/button';
import { Input } from '../../../shared/ui/input';
import { FormField } from '../../../shared/components/FormField';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { DownloadText } from '../../../shared/components/DownloadText';

export const DataPreview = memo(function DataPreview({ path, active }: { path: string; active: boolean }) {
  const [file, setFile] = useState(path);
  const [source, setSource] = useState('');
  const [count, setCount] = useState('5');
  const [result, setResult] = useState<{ input: SampleDataInput; data: SampleDataResponse } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!active) { pending.current?.abort(); pending.current = null; setLoading(false); }
    return () => { pending.current?.abort(); pending.current = null; };
  }, [active]);
  const stale = result && (result.input.parquet !== file.trim() || result.input.source !== source.trim() || result.input.data_n !== Number(count));
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (pending.current || !active) return;
    const n = Number(count);
    if (!file.trim() || !Number.isInteger(n) || n < 1 || n > 20) {
      setError('请填写服务端路径，预览条数需为 1–20 的整数。');
      return;
    }
    const input = { parquet: file.trim(), source: source.trim(), data_n: n };
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true); setError(null);
    try {
      const data = await masApi.sampleData(input, controller.signal);
      if (!controller.signal.aborted) setResult({ input, data });
    } catch (reason) {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    } finally {
      if (pending.current === controller) { pending.current = null; setLoading(false); }
    }
  };
  return <div className="resource-preview">
    <form onSubmit={event => void submit(event)}>
      <FormField label="服务端 parquet 路径" hint="仅使用现有任务格式读取接口，不是上传或注册数据集。修改此处不会修改训练配置。">
        <Input required value={file} onChange={event => setFile(event.target.value)} placeholder="填写 Control 服务可访问的路径" />
      </FormField>
      <div className="form-grid">
        <FormField label="来源筛选" hint="按 source 字段过滤，留空不过滤。"><Input value={source} onChange={event => setSource(event.target.value)} /></FormField>
        <FormField label="预览条数" hint="筛选后的前 N 条，不是随机采样或数据总量。">
          <Input type="number" min={1} max={20} required value={count} onChange={event => setCount(event.target.value)} />
        </FormField>
      </div>
      <Button type="submit" size="sm" loading={loading}><Eye size={14} />读取预览</Button>
    </form>
    <p className="field-hint">只读取文件，不调用模型、不启动训练。需要服务端已安装 parquet 读取依赖；文件实际规模和训练格式兼容性不由本预览保证。</p>
    {error && <InlineNotice tone="warning"><div>预览失败，请检查路径、任务格式和服务端数据依赖。
      <DownloadText text={error} filename="data-preview-error.txt" label="下载错误详情" /></div></InlineNotice>}
    {stale && <p className="field-hint">输入条件已改变，下方仍是上次预览，请重新读取。</p>}
    {result && <section aria-label="任务预览结果"><h3>预览任务 · {result.data.n} 条</h3>
      <p className="field-hint">来源路径：{result.input.parquet}</p>
      <ol>{(result.data.tasks || []).map((task, index) => <li key={`${task.id}-${index}`}>
        <div><code>{task.id || `任务 ${index + 1}`}</code><span>{task.source}</span></div>
        <p>{task.question.length > 800 ? `${task.question.slice(0, 800)}…（预览摘要）` : task.question}</p>
      </li>)}</ol>
    </section>}
  </div>;
});
