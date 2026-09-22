import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertDialog } from 'radix-ui';
import { Button } from './button';
import type { TrainingPreflightResponse } from '../api/types';
import type { SettingsSection } from '../../features/settings/model/sections';

export function useTrainConfirmation(onConfigure?: (section: SettingsSection) => void) {
  const [open, setOpen] = useState(false);
  const [review, setReview] = useState<TrainingPreflightResponse | null>(null);
  const resolveRef = useRef<((confirmed: boolean) => void) | null>(null);
  const triggerRef = useRef<HTMLElement | null>(null);
  const confirm = useCallback((preflight: TrainingPreflightResponse) => new Promise<boolean>(resolve => {
    if (resolveRef.current) { resolve(false); return; }
    triggerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    resolveRef.current = resolve;
    setReview(preflight);
    setOpen(true);
  }), []);
  const close = useCallback((accepted: boolean) => {
    resolveRef.current?.(accepted);
    resolveRef.current = null;
    setOpen(false);
  }, []);
  useEffect(() => () => { resolveRef.current?.(false); resolveRef.current = null; }, []);

  const dialog = <AlertDialog.Root open={open} onOpenChange={value => { if (!value) close(false); }}>
    <AlertDialog.Portal>
      <AlertDialog.Overlay className="dialog-overlay" />
      <AlertDialog.Content className="dialog-content" onCloseAutoFocus={event => {
        event.preventDefault();
        triggerRef.current?.focus();
      }}>
        <div className="eyebrow">TRAINING</div>
        <AlertDialog.Title className="dialog-title">启动训练</AlertDialog.Title>
        <AlertDialog.Description className="dialog-description">确认本次训练使用的服务端配置与执行条件。</AlertDialog.Description>
        {review && <div className="training-preflight">
          <dl>
            <div><dt>算法</dt><dd>{review.effective.algorithm.toUpperCase()} · {review.effective.algorithm_source === 'workflow.sampling' ? 'Sampling Policy' : 'RL 配置'}</dd></div>
            <div><dt>模型</dt><dd>{review.effective.model.resource_name || review.effective.model.model_path || '未配置'}</dd></div>
            <div><dt>GPU</dt><dd>{review.effective.gpu_ids.join(', ') || '未选择'}</dd></div>
            <div><dt>采样</dt><dd>group_n={review.effective.group_n} · Branch Sites={review.effective.branch_site_count}</dd></div>
            <div><dt>训练数据</dt><dd>{String(review.effective.data?.train_files || '未配置')}</dd></div>
            <div><dt>验证数据</dt><dd>{String(review.effective.data?.val_files || '未配置')}</dd></div>
          </dl>
          <div className="training-preflight-checks">
            {review.checks.map(check => <div className={`training-preflight-check is-${check.status}`} key={check.id}>
              <strong>{check.label}</strong><span>{check.message}</span>
            </div>)}
          </div>
          {review.warnings.map(issue => <p className="field-hint" key={issue.code}>{issue.message}</p>)}
          {onConfigure && <div className="action-bar">
            {(['model', 'data', 'environment'] as const).map((section, index) =>
              <Button key={section} size="sm" variant="ghost" onClick={() => { close(false); onConfigure(section); }}>
                {['修改模型', '修改数据与采样', '修改执行环境'][index]}
              </Button>)}
          </div>}
        </div>}
        <div className="dialog-actions">
          <AlertDialog.Cancel asChild><Button onClick={() => close(false)}>取消</Button></AlertDialog.Cancel>
          <AlertDialog.Action asChild><Button variant="primary" disabled={!review?.ready} onClick={() => close(true)}>
            {review?.ready ? '确认启动' : '检查未通过'}
          </Button></AlertDialog.Action>
        </div>
      </AlertDialog.Content>
    </AlertDialog.Portal>
  </AlertDialog.Root>;
  return { confirm, dialog };
}
