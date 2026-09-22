import { memo } from 'react';
import { DropdownMenu } from 'radix-ui';
import { ArrowLeft, Check, ChevronDown } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { NAVIGATION, type PanelId } from '../navigation';

const TOOLS = NAVIGATION.filter(item => item.id === 'records' || item.id === 'harness' || item.id === 'monitor' || item.id === 'experiment');
const LABELS = { records: '训练记录', harness: '诊断工具', monitor: '实验监控', experiment: '实验信息' } as const;

export const WorkspaceMenu = memo(function WorkspaceMenu({ active, onChange }: {
  active: PanelId; onChange: (id: PanelId) => void;
}) {
  return <div className="workspace-navigation">
    {TOOLS.some(item => item.id === active) && <Button size="sm" variant="ghost" onClick={() => onChange('mas')}><ArrowLeft size={14} />返回实验</Button>}
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button size="sm" variant="ghost">更多功能<ChevronDown size={13} /></Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="workspace-navigation-menu" align="end" sideOffset={8} collisionPadding={12}>
          <DropdownMenu.Label className="workspace-navigation-label">实验工具</DropdownMenu.Label>
          {TOOLS.map(({ id, icon: Icon }) => <DropdownMenu.Item
            key={id} className="workspace-navigation-item" onSelect={() => onChange(id)} aria-current={active === id ? 'page' : undefined}>
            <Icon size={15} /><span>{LABELS[id]}</span>
            {active === id && <Check size={14} />}
          </DropdownMenu.Item>)}
          <DropdownMenu.Arrow className="workspace-navigation-arrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  </div>;
});
