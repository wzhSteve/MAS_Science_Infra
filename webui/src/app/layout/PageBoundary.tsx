import { Component, type ErrorInfo, type ReactNode } from 'react';
import { InlineNotice } from '../../shared/components/InlineNotice';
import { Button } from '../../shared/ui/button';

export class PageBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state: { error: string | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error: error.message }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Page rendering failed', error, info); }
  render() {
    if (!this.state.error) return this.props.children;
    return <div className="workspace-content">
      <InlineNotice tone="danger">页面暂时无法显示：{this.state.error}</InlineNotice>
      <p className="field-hint">已保存的实验仍在服务端。重试会重新载入此页面，未保存的本页输入可能无法恢复。</p>
      <Button onClick={() => this.setState({ error: null })}>重试页面</Button>
    </div>;
  }
}
