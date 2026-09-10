// Copyright (c) Microsoft. All rights reserved.

import type { Meta, StoryObj } from '@storybook/react';
import { waitFor, within } from '@testing-library/dom';
import userEvent from '@testing-library/user-event';
import { SciencePage } from './Science.page';

const meta: Meta<typeof SciencePage> = {
  title: 'Pages/SciencePage',
  component: SciencePage,
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;

type Story = StoryObj<typeof SciencePage>;

export const Empty: Story = {};

export const WithSample: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.click(canvas.getByTestId('science-load-sample'));
    await waitFor(() => {
      canvas.getByTestId('science-mean-reward');
    });
  },
};
