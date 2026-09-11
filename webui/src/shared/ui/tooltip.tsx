import { Tooltip as Primitive } from 'radix-ui';
import type { ReactElement, ReactNode } from 'react';

export function Tooltip({ children, content }: { children: ReactElement; content: ReactNode }) {
  return (
    <Primitive.Root>
      <Primitive.Trigger asChild>{children}</Primitive.Trigger>
      <Primitive.Portal>
        <Primitive.Content className="ui-tooltip" sideOffset={6}>
          {content}<Primitive.Arrow className="ui-tooltip-arrow" />
        </Primitive.Content>
      </Primitive.Portal>
    </Primitive.Root>
  );
}
