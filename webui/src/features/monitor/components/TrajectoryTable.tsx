import { memo, useId, useMemo, useRef, useState } from 'react';
import type { Trajectory } from '../../../shared/api/types';
import { Button } from '../../../shared/ui/button';
import { DataTable } from '../../../shared/components/DataTable';
import { EmptyState } from '../../../shared/components/EmptyState';
import { JsonDetails } from '../../../shared/components/JsonDetails';
import { Section } from '../../../shared/components/Section';

export const TrajectoryTable = memo(function TrajectoryTable({ trajectories }: { trajectories: Trajectory[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selectedTrigger = useRef<HTMLButtonElement>(null);
  const detailsId = useId();
  const selected = useMemo(
    () => trajectories.find((trajectory) => trajectory.trajectory_id === selectedId),
    [trajectories, selectedId],
  );

  return (
    <Section title="Trajectories">
      <DataTable aria-label="Trajectories">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">id</th>
            <th scope="col">reward</th>
            <th scope="col">format</th>
            <th scope="col">answer</th>
            <th scope="col">Events</th>
          </tr>
        </thead>
        <tbody>
          {trajectories.map((trajectory, index) => (
            <tr
              key={trajectory.trajectory_id}
              className="cursor-pointer"
              onClick={() => setSelectedId(trajectory.trajectory_id)}
            >
              <td>{index + 1}</td>
              <td className="mono max-w-60 break-all">{trajectory.trajectory_id}</td>
              <td className="mono">{trajectory.reward ?? ''}</td>
              <td>{trajectory.format_ok ? 'yes' : 'no'}</td>
              <td className="min-w-48 max-w-xl whitespace-pre-wrap break-words">{trajectory.answer ?? ''}</td>
              <td>
                <Button
                  size="sm"
                  ref={selectedId === trajectory.trajectory_id ? selectedTrigger : undefined}
                  aria-label={`查看 Events: ${trajectory.trajectory_id}`}
                  aria-expanded={selectedId === trajectory.trajectory_id}
                  aria-controls={detailsId}
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelectedId((current) => current === trajectory.trajectory_id ? null : trajectory.trajectory_id);
                  }}
                >
                  查看 Events
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      {!trajectories.length ? <EmptyState title="暂无 trajectories" /> : null}
      {selected ? (
        <div id={detailsId} className="mt-4" role="region" aria-label={`Events: ${selected.trajectory_id}`}>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h3 className="mono break-all">Events · {selected.trajectory_id}</h3>
            <Button size="sm" onClick={() => {
              selectedTrigger.current?.focus();
              setSelectedId(null);
            }}>关闭 Events</Button>
          </div>
          <JsonDetails key={selected.trajectory_id} value={selected.events} label="完整 Events" open />
        </div>
      ) : null}
    </Section>
  );
});
