import { memo } from 'react';
import { Network } from 'lucide-react';
import { NAVIGATION, type PanelId } from '../navigation';

export const Sidebar = memo(function Sidebar({ active, onChange }: { active: PanelId; onChange: (id: PanelId) => void }) {
  return <aside className="sidebar">
    <div className="brand"><span className="brand-mark"><Network size={19} strokeWidth={1.6} /></span><div>Science Studio<span>MAS / RL RESEARCH</span></div></div>
    <div className="sidebar-label">WORKSPACE</div>
    <nav aria-label="工作区面板" className="sidebar-nav">
      {NAVIGATION.map(({ id, label, icon: Icon }) =>
        <button type="button" className={`nav-link${id === active ? ' is-active' : ''}`}
          key={id} onClick={() => onChange(id)} aria-current={id === active ? 'page' : undefined} aria-controls={`panel-${id}`}>
          <Icon size={17} strokeWidth={1.7} aria-hidden="true" />{label}
        </button>)}
    </nav>
    <div className="sidebar-footer"><span className="sidebar-footer-dot" />Research workspace<span className="mono">v0.1</span></div>
  </aside>;
});
