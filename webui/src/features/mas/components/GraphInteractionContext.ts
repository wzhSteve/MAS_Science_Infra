import { createContext, useContext } from 'react';
import type { GraphEdge } from '../types';
import type { EdgeRules } from '../model/edgeRules';

export const GraphInteractionContext = createContext<{
  rules: EdgeRules;
  reconnecting: GraphEdge | null;
  onSelectEdge: (id: string) => void;
} | null>(null);

export function useGraphInteraction() {
  const context = useContext(GraphInteractionContext);
  if (!context) throw new Error('GraphInteractionContext is required');
  return context;
}
