import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Dialog } from 'radix-ui';
import { Network, X } from 'lucide-react';
import { experimentApi } from '../../api';
import type { Bundle } from '../../../../shared/api/types';
import { errorMessage } from '../../../../shared/api/http';
import { FormField } from '../../../../shared/components/FormField';
import { InlineNotice } from '../../../../shared/components/InlineNotice';
import { Input } from '../../../../shared/ui/input';
import { Button } from '../../../../shared/ui/button';

export function CreateExperimentDialog({ open, onOpenChange, onCreated }: {
  open: boolean; onOpenChange: (open: boolean) => void; onCreated: (bundle: Bundle) => void;
}) {
  const [name, setName] = useState('');
  const [id, setId] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const locked = useRef(false);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    if (!pending && (!open || (!name && !id))) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [pending, open, Boolean(name || id)]);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (locked.current) return;
    const experimentId = id.trim();
    const title = name.trim();
    if (!title || !/^[A-Za-z0-9_-]{1,64}$/.test(experimentId)) {
      setError('请填写实验名称；ID 需为 1–64 位英文字母、数字、下划线或短横线。');
      return;
    }
    locked.current = true;
    setPending(true);
    setError(null);
    try {
      const listing = await experimentApi.listExperiments();
      if (listing.experiments.includes(experimentId)) {
        throw new Error('这个实验 ID 已存在，请更换 ID，或返回首页打开已有实验。');
      }
      const bundle = await experimentApi.createExperiment(experimentId, 42, title);
      if (bundle.id !== experimentId) throw new Error('创建结果的实验身份不匹配，请刷新首页核实，勿重复提交。');
      if (!alive.current) return;
      setName('');
      setId('');
      onOpenChange(false);
      onCreated(bundle);
    } catch (reason) {
      if (alive.current) setError(`未能确认创建完成：${errorMessage(reason)}。若请求曾中断，请先刷新首页核实。`);
    } finally {
      locked.current = false;
      if (alive.current) setPending(false);
    }
  };
  return <Dialog.Root open={open} onOpenChange={value => { if (!locked.current) onOpenChange(value); }}>
    <Dialog.Portal>
      <Dialog.Overlay className="dialog-overlay" />
      <Dialog.Content className="dialog-content experiment-create-dialog">
        <div className="experiment-create-heading">
          <span className="eyebrow">NEW EXPERIMENT</span>
          <Dialog.Close asChild><Button size="sm" variant="ghost" disabled={pending} aria-label="关闭新建实验"><X size={16} /></Button></Dialog.Close>
        </div>
        <Dialog.Title className="dialog-title">新建实验</Dialog.Title>
        <Dialog.Description className="dialog-description">
          从默认的单 Agent 工作流开始。进入画布后再配置模型、添加协作角色，无需先准备训练环境。
        </Dialog.Description>
        <form onSubmit={event => void submit(event)} className="experiment-create-form">
          <FormField label="实验名称"><Input autoFocus required maxLength={160} value={name} disabled={pending}
            placeholder="例如：计划与验证研究" onChange={event => setName(event.target.value)} /></FormField>
          <FormField label="实验 ID" hint="用于实验地址与服务端目录，创建后不在此修改。1–64 位字母、数字、下划线或短横线。">
            <Input required maxLength={64} value={id} disabled={pending} placeholder="例如：planning-study"
              pattern="[A-Za-z0-9_-]{1,64}" autoComplete="off" className="mono" onChange={event => setId(event.target.value)} />
          </FormField>
          <div className="experiment-create-preset"><Network size={18} /><div><strong>默认 Workflow</strong>
            <p>hub Agent · 内置工具 · 随机种子 42</p></div></div>
          <p className="field-hint">不会复制其他实验的配置、密钥或运行记录，也不会自动调用模型。服务端已有的默认模型配置仍按原规则生效。</p>
          {error && <InlineNotice tone="danger">{error}</InlineNotice>}
          <div className="dialog-actions">
            <Button disabled={pending} onClick={() => onOpenChange(false)}>取消</Button>
            <Button type="submit" variant="primary" loading={pending}>创建并打开</Button>
          </div>
        </form>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>;
}
