// Copyright (c) Microsoft. All rights reserved.

import { useCallback, useEffect, useMemo, useState } from 'react';
import { IconFlask, IconUpload } from '@tabler/icons-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Alert,
  Badge,
  Button,
  Card,
  FileButton,
  Group,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import { SAMPLE_SCIENCE_PAYLOAD, buildScienceView, type ScienceView } from '@/features/science';

const STORAGE_KEY = 'agl-science-payload';

function formatScalar(value: number | null | undefined, digits = 4): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '—';
  }
  return value.toFixed(digits);
}

function kindColor(kind: string): string {
  if (kind === 'error') {
    return 'red';
  }
  if (kind === 'feedback') {
    return 'violet';
  }
  if (kind === 'final_answer') {
    return 'teal';
  }
  if (kind === 'tool_call' || kind === 'tool_result') {
    return 'blue';
  }
  return 'gray';
}

function persistPayload(raw: unknown): { view: ScienceView | null; error: string | null } {
  const { view, error } = buildScienceView(raw);
  if (error || !view) {
    return { view: null, error: error ?? 'Failed to parse Science MAS JSON' };
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(raw));
  } catch {
    // Ignore quota / private-mode failures.
  }
  return { view, error: null };
}

export function SciencePage() {
  const [view, setView] = useState<ScienceView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [paste, setPaste] = useState('');
  const [tensorboard, setTensorboard] = useState('');
  const [pluginFilter, setPluginFilter] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (!saved) {
        return;
      }
      const { view: restored, error: restoreError } = buildScienceView(JSON.parse(saved));
      if (restored && !restoreError) {
        setView(restored);
      }
    } catch {
      // Ignore corrupt localStorage.
    }
  }, []);

  const loadRaw = useCallback((raw: unknown) => {
    const result = persistPayload(raw);
    setError(result.error);
    setView(result.view);
    setSelectedIdx(null);
    setPluginFilter(null);
  }, []);

  const onFile = useCallback(
    (file: File | null) => {
      if (!file) {
        return;
      }
      void file
        .text()
        .then((text) => {
          try {
            loadRaw(JSON.parse(text));
          } catch {
            setError('Invalid JSON file');
          }
        })
        .catch(() => setError('Failed to read file'));
    },
    [loadRaw],
  );

  const hypotheses = useMemo(() => {
    const rows = view?.hypotheses ?? [];
    if (!pluginFilter) {
      return rows;
    }
    return rows.filter((row) => row.plugin === pluginFilter);
  }, [pluginFilter, view?.hypotheses]);

  const pluginOptions = useMemo(() => {
    const names = [...new Set((view?.hypotheses ?? []).map((row) => row.plugin))].filter(Boolean);
    return [{ value: '', label: 'All plugins' }, ...names.map((name) => ({ value: name, label: name }))];
  }, [view?.hypotheses]);

  const selected = selectedIdx != null ? (view?.trajectories[selectedIdx] ?? null) : null;
  const tbFromMeta =
    typeof view?.trainSignal.meta.dashboard === 'string'
      ? view.trainSignal.meta.dashboard
      : typeof view?.trainSignal.meta.tensorboard === 'string'
        ? view.trainSignal.meta.tensorboard
        : '';
  const tbUrl = tensorboard.trim() || tbFromMeta;

  return (
    <Stack gap='md' data-testid='science-page'>
      <Group justify='space-between' align='flex-end'>
        <div>
          <Group gap='xs'>
            <IconFlask size={22} />
            <Title order={2}>Science MAS</Title>
          </Group>
          <Text c='dimmed' size='sm' mt={4}>
            Load a `science-infra collect` JSON to inspect trajectories, harness hypotheses, and TrainSignal. This page
            does not call a new Store API.
          </Text>
        </div>
        <Group gap='xs'>
          <FileButton onChange={onFile} accept='application/json,.json'>
            {(props) => (
              <Button {...props} leftSection={<IconUpload size={16} />} variant='light'>
                Open JSON
              </Button>
            )}
          </FileButton>
          <Button variant='light' onClick={() => loadRaw(SAMPLE_SCIENCE_PAYLOAD)} data-testid='science-load-sample'>
            Load sample
          </Button>
          <Button
            variant='subtle'
            color='gray'
            onClick={() => {
              setView(null);
              setError(null);
              setSelectedIdx(null);
              try {
                window.localStorage.removeItem(STORAGE_KEY);
              } catch {
                // ignore
              }
            }}
          >
            Clear
          </Button>
        </Group>
      </Group>

      {error && (
        <Alert color='red' title='Could not parse JSON'>
          {error}
        </Alert>
      )}

      {!view && (
        <Alert color='blue' title='No collect payload yet'>
          Run `science-infra collect --mock --n 2 --out /tmp/traj.json`, then open that file here. You can also paste JSON
          below or load the built-in sample.
        </Alert>
      )}

      <Card withBorder padding='md' radius='md'>
        <Text size='sm' fw={600} mb={6}>
          Paste collect JSON
        </Text>
        <Textarea
          minRows={4}
          autosize
          maxRows={10}
          value={paste}
          onChange={(event) => setPaste(event.currentTarget.value)}
          placeholder='{"batch": {"trajectories": []}, "train_signal": {}}'
          data-testid='science-paste'
        />
        <Group mt='sm'>
          <Button
            size='xs'
            variant='light'
            onClick={() => {
              try {
                loadRaw(JSON.parse(paste));
              } catch {
                setError('Paste is not valid JSON');
              }
            }}
          >
            Parse paste
          </Button>
        </Group>
      </Card>

      {view && (
        <>
          <SimpleGrid cols={{ base: 2, sm: 3, md: 5 }} spacing='md'>
            <Card withBorder padding='md' radius='md'>
              <Text size='xs' c='dimmed' tt='uppercase'>
                mean reward
              </Text>
              <Text fw={650} size='xl' ff='monospace' data-testid='science-mean-reward'>
                {formatScalar(view.meanReward)}
              </Text>
            </Card>
            <Card withBorder padding='md' radius='md'>
              <Text size='xs' c='dimmed' tt='uppercase'>
                trajectories
              </Text>
              <Text fw={650} size='xl' ff='monospace'>
                {view.n}
              </Text>
            </Card>
            <Card withBorder padding='md' radius='md'>
              <Text size='xs' c='dimmed' tt='uppercase'>
                errors
              </Text>
              <Text fw={650} size='xl' ff='monospace'>
                {view.nError}
              </Text>
            </Card>
            <Card withBorder padding='md' radius='md'>
              <Text size='xs' c='dimmed' tt='uppercase'>
                feedback hops
              </Text>
              <Text fw={650} size='xl' ff='monospace'>
                {view.nFeedback}
              </Text>
            </Card>
            <Card withBorder padding='md' radius='md'>
              <Text size='xs' c='dimmed' tt='uppercase'>
                harness hits
              </Text>
              <Text fw={650} size='xl' ff='monospace'>
                {view.nHypotheses}
              </Text>
            </Card>
          </SimpleGrid>

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing='md'>
            <Card withBorder padding='md' radius='md' id='train-signal'>
              <Title order={4} mb='sm'>
                TrainSignal
              </Title>
              <Text size='sm'>
                advantage=<Text span fw={600}>{view.trainSignal.advantage || '(unset)'}</Text>
                {' · '}
                loss=<Text span fw={600}>{view.trainSignal.loss || '(unset)'}</Text>
              </Text>
              <Group gap={6} mt='sm'>
                {Object.entries(view.eventCounts).map(([kind, count]) => (
                  <Badge key={kind} color={kindColor(kind)} variant='light'>
                    {kind} · {count}
                  </Badge>
                ))}
              </Group>
              <TextInput
                mt='md'
                label='TensorBoard / dashboard URL'
                placeholder='http://127.0.0.1:6006'
                value={tensorboard}
                onChange={(event) => setTensorboard(event.currentTarget.value)}
              />
              {tbUrl ? (
                <Text size='sm' mt='xs'>
                  Open{' '}
                  <Text component='a' href={tbUrl} target='_blank' rel='noreferrer' c='blue' span inherit>
                    {tbUrl}
                  </Text>
                </Text>
              ) : (
                <Text size='xs' c='dimmed' mt='xs'>
                  Optional. Live training curves still sit on the Metrics page after a run.
                </Text>
              )}
            </Card>

            <Card withBorder padding='md' radius='md'>
              <Title order={4} mb='sm'>
                Episode reward
              </Title>
              {view.rewards.length === 0 ? (
                <Text c='dimmed' size='sm'>
                  No rewarded trajectories.
                </Text>
              ) : (
                <ResponsiveContainer width='100%' height={240}>
                  <LineChart data={view.rewards}>
                    <CartesianGrid strokeDasharray='3 3' />
                    <XAxis dataKey='index' label={{ value: 'Episode #', position: 'insideBottom', offset: -2 }} />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type='monotone' dataKey='reward' name='final_reward' stroke='#228be6' dot strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Card>
          </SimpleGrid>

          <Card withBorder padding='md' radius='md'>
            <Title order={4} mb='sm'>
              Trajectories
            </Title>
            <Text c='dimmed' size='xs' mb='sm'>
              Click a row to inspect events.
            </Text>
            <ScrollArea>
              <Table highlightOnHover striped stickyHeader>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>#</Table.Th>
                    <Table.Th>id</Table.Th>
                    <Table.Th>reward</Table.Th>
                    <Table.Th>format</Table.Th>
                    <Table.Th>search</Table.Th>
                    <Table.Th>python</Table.Th>
                    <Table.Th>answer</Table.Th>
                    <Table.Th>events</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {view.trajectories.map((row, index) => (
                    <Table.Tr
                      key={row.trajectoryId || String(index)}
                      onClick={() => setSelectedIdx(index)}
                      style={{ cursor: 'pointer' }}
                      bg={selectedIdx === index ? 'var(--mantine-color-blue-light)' : undefined}
                    >
                      <Table.Td>{index + 1}</Table.Td>
                      <Table.Td>{row.trajectoryId.slice(0, 16)}</Table.Td>
                      <Table.Td>{formatScalar(row.reward, 3)}</Table.Td>
                      <Table.Td>{row.formatOk ? 'yes' : 'no'}</Table.Td>
                      <Table.Td>{row.nSearch}</Table.Td>
                      <Table.Td>{row.nPython}</Table.Td>
                      <Table.Td>{row.answer.slice(0, 80)}</Table.Td>
                      <Table.Td>{row.kinds.slice(0, 8).join(',')}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
            <Title order={5} mt='md' mb='xs'>
              Events
            </Title>
            <ScrollArea h={240} type='auto'>
              <Text size='xs' ff='monospace' style={{ whiteSpace: 'pre-wrap' }}>
                {selected ? JSON.stringify(selected.events, null, 2) : '(select a trajectory)'}
              </Text>
            </ScrollArea>
          </Card>

          <Card withBorder padding='md' radius='md' id='harness'>
            <Group justify='space-between' mb='sm'>
              <Title order={4}>Harness</Title>
              <Select
                size='xs'
                w={220}
                data={pluginOptions}
                value={pluginFilter ?? ''}
                onChange={(value) => setPluginFilter(value || null)}
                allowDeselect={false}
              />
            </Group>
            {hypotheses.length === 0 ? (
              <Text c='dimmed' size='sm'>
                No hypotheses for this filter.
              </Text>
            ) : (
              <Table>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>plugin</Table.Th>
                    <Table.Th>event_id</Table.Th>
                    <Table.Th>message</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {hypotheses.map((hyp, index) => (
                    <Table.Tr key={`${hyp.plugin}-${hyp.eventId}-${index}`}>
                      <Table.Td>
                        <Badge variant='light'>{hyp.plugin}</Badge>
                      </Table.Td>
                      <Table.Td>{hyp.eventId}</Table.Td>
                      <Table.Td>{hyp.message}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            )}
          </Card>
        </>
      )}
    </Stack>
  );
}
