import { memo } from 'react';
import { InlineNotice } from '../../../shared/components/InlineNotice';
import { DownloadText } from '../../../shared/components/DownloadText';

const stages: Record<string, string> = {
  model: '模型请求失败', model_setup: '模型初始化失败', configuration: '运行配置未就绪',
  runtime: '工作流执行失败', tool: '工具执行失败', execution: 'Agent 执行失败',
  orchestration: 'Agent 编排失败', storage: '运行记录保存失败', skill: '技能执行失败',
};

export const RunErrorDetails = memo(function RunErrorDetails({ message, stage, code, unknown = false, label }: {
  message: string; stage?: string; code?: string; unknown?: boolean; label?: string;
}) {
  return <InlineNotice tone={unknown ? 'warning' : 'danger'}>
    <div className="debug-error-content">
      <strong>{label || (unknown ? '未取得运行结果，请先核实记录，不要重复发送。'
        : stage ? stages[stage] || `运行失败（${stage}）` : '本次操作未完成，输入和设计草稿仍保留。')}</strong>
      {code && <span className="field-hint">错误类型：{code}</span>}
      <DownloadText text={`${stage || ''} ${code || ''}\n${message}`} filename="run-error.txt" label="下载错误详情" />
    </div>
  </InlineNotice>;
});
