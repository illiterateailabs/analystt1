import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';

import ChatInterface from '../ChatInterface';
import * as api from '../../../lib/api';
import { useToast } from '../../../hooks/useToast';

// Mock the API functions
jest.mock('../../../lib/api', () => ({
  ...jest.requireActual('../../../lib/api'),
  chatAPI: {
    sendMessage: jest.fn(),
  },
}));

// Mock Next.js useRouter
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock useToast hook
jest.mock('../../../hooks/useToast', () => ({
  useToast: jest.fn(),
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // Disable retries for tests
    },
  },
});

const renderWithClient = (ui: React.ReactElement) => {
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
};

describe('ChatInterface', () => {
  const mockToast = jest.fn();
  const mockPush = jest.fn();

  // Mock scrollIntoView
  const scrollIntoViewMock = jest.fn();
  window.HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient.clear();
    (useRouter as jest.Mock).mockReturnValue({ push: mockPush });
    (useToast as jest.Mock).mockReturnValue({ toast: mockToast });

    // Default successful mock for sendMessage
    (api.chatAPI.sendMessage as jest.Mock).mockResolvedValue({
      data: {
        id: 'msg-123',
        role: 'assistant',
        content: 'Hello, how can I help you?',
        timestamp: new Date().toISOString(),
      },
    });
  });

  // 1. Rendering of chat input and send button.
  test('renders chat input and send button', () => {
    renderWithClient(<ChatInterface />);
    expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  // 2. Displaying messages from user and assistant.
  test('displays initial messages', () => {
    const initialMessages = [
      { id: '1', role: 'user', content: 'Hi there!', timestamp: '2023-01-01T10:00:00Z' },
      { id: '2', role: 'assistant', content: 'Hello!', timestamp: '2023-01-01T10:01:00Z' },
    ];
    renderWithClient(<ChatInterface initialMessages={initialMessages} />);

    expect(screen.getByText('Hi there!')).toBeInTheDocument();
    expect(screen.getByText('Hello!')).toBeInTheDocument();
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('Assistant')).toBeInTheDocument();
  });

  // 3. Sending a message when send button is clicked or Enter is pressed.
  test('sends a message when send button is clicked', async () => {
    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });

    await userEvent.type(input, 'Test message');
    await userEvent.click(sendButton);

    await waitFor(() => {
      expect(api.chatAPI.sendMessage).toHaveBeenCalledWith('Test message', false);
    });
    expect(screen.getByText('Test message')).toBeInTheDocument(); // User message
    expect(screen.getByText('Hello, how can I help you?')).toBeInTheDocument(); // Assistant response
  });

  test('sends a message when Enter is pressed', async () => {
    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');

    await userEvent.type(input, 'Another message{enter}');

    await waitFor(() => {
      expect(api.chatAPI.sendMessage).toHaveBeenCalledWith('Another message', false);
    });
    expect(screen.getByText('Another message')).toBeInTheDocument();
    expect(screen.getByText('Hello, how can I help you?')).toBeInTheDocument();
  });

  // 4. Clearing input after sending a message.
  test('clears input after sending a message', async () => {
    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });

    await userEvent.type(input, 'Message to clear');
    await userEvent.click(sendButton);

    await waitFor(() => {
      expect(input).toHaveValue('');
    });
  });

  // 5. Handling API call for sending message (mocking `chatAPI.sendMessage`).
  test('calls chatAPI.sendMessage with correct parameters', async () => {
    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });

    await userEvent.type(input, 'API test message');
    await userEvent.click(sendButton);

    await waitFor(() => {
      expect(api.chatAPI.sendMessage).toHaveBeenCalledWith('API test message', false);
    });
  });

  // 6. Displaying loading state while waiting for assistant response.
  test('displays loading state while waiting for assistant response', async () => {
    (api.chatAPI.sendMessage as jest.Mock).mockReturnValue(new Promise(() => {})); // Never resolves

    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });

    await userEvent.type(input, 'Loading test');
    await userEvent.click(sendButton);

    expect(screen.getByText('Assistant is typing...')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  // 7. Displaying error state if API call fails.
  test('displays error state if API call fails', async () => {
    (api.chatAPI.sendMessage as jest.Mock).mockRejectedValue(new Error('API Error'));

    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });

    await userEvent.type(input, 'Error test');
    await userEvent.click(sendButton);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
        description: 'Failed to send message. Please try again.',
        variant: 'destructive',
      }));
    });
    expect(screen.queryByText('Error test')).not.toBeInTheDocument(); // User message should not be added
    expect(screen.queryByText('Assistant is typing...')).not.toBeInTheDocument();
  });

  // 8. Scrolling to bottom when new messages are added.
  test('scrolls to bottom when new messages are added', async () => {
    renderWithClient(<ChatInterface />);
    const input = screen.getByPlaceholderText('Type your message...');
    const sendButton = screen.getByRole('button', { name: /send/i });

    // Simulate adding a message
    await userEvent.type(input, 'New message');
    await userEvent.click(sendButton);

    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled();
    });
  });
});
