import { memo } from 'react';
import { DropdownMenu } from 'radix-ui';
import { Check, ChevronDown, PanelsTopLeft } from 'lucide-react';
import { Button } from '../../shared/ui/button';
import { NAVIGATION, type PanelId } from '../navigation';

export const WorkspaceMenu = memo(function WorkspaceMenu({ active, onChange }: {
  active: PanelId; onChange: (id: PanelId) => void;
}) {
  return <div className="workspace-navigation">
    {active !== 'mas' && <Button size="sm" variant="ghost" onClick={() => onChange('mas')}>
      <PanelsTopLeft size={14} />返回画布
    </Button>}
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button size="sm" variant="ghost">更多功能<ChevronDown size={13} /></Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="workspace-navigation-menu" align="end" sideOffset={8} collisionPadding={12}>
          <DropdownMenu.Label className="workspace-navigation-label">当前实验</DropdownMenu.Label>
          {NAVIGATION.map(({ id, description, icon: Icon }) => <DropdownMenu.Item
            key={id} className="workspace-navigation-item" onSelect={() => onChange(id)} aria-current={active === id ? 'page' : undefined}>
            <Icon size={15} /><span>{id === 'mas' ? 'MAS 画布' : description}</span>
            {active === id && <Check size={14} />}
          </DropdownMenu.Item>)}
          <DropdownMenu.Arrow className="workspace-navigation-arrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  </div>;
});
