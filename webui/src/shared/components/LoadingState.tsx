export function LoadingState({ label = '正在加载…' }: { label?: string }) {
  return <div className="loading-state" role="status"><span className="loading-dot" />{label}</div>;
}
