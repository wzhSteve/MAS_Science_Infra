import { useContext } from 'react';
import { ConfirmContext } from './ConfirmProvider';

export function useConfirm() {
  const confirm = useContext(ConfirmContext);
  if (!confirm) throw new Error('ConfirmProvider is required');
  return confirm;
}
