import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Network } from 'vis-network';
import GraphVisualization from '../GraphVisualization';

// Mock the vis-network library
jest.mock('vis-network', () => ({
  Network: jest.fn(() => ({
    setData: jest.fn(),
    on: jest.fn(),
    off: jest.fn(),
    destroy: jest.fn(),
    fit: jest.fn(),
    focus: jest.fn(),
    setOptions: jest.fn(),
  })),
  DataSet: jest.fn((data) => ({
    add: jest.fn(),
    update: jest.fn(),
    remove: jest.fn(),
    get: jest.fn(() => data), // Mock get to return initial data
  })),
}));

const mockNetwork = Network as jest.MockedClass<typeof Network>;

describe('GraphVisualization', () => {
  const mockGraphData = {
    nodes: [
      { id: 'node1', label: 'Node 1', properties: { type: 'Person' } },
      { id: 'node2', label: 'Node 2', properties: { type: 'Company' } },
    ],
    edges: [
      { from: 'node1', to: 'node2', label: 'WORKS_AT', properties: {} },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the mock implementation for each test
    mockNetwork.mockImplementation(() => ({
      setData: jest.fn(),
      on: jest.fn(),
      off: jest.fn(),
      destroy: jest.fn(),
      fit: jest.fn(),
      focus: jest.fn(),
      setOptions: jest.fn(),
    }));
  });

  // 1. Rendering with empty graph data
  test('renders without crashing with empty graph data', () => {
    render(<GraphVisualization graphData={{ nodes: [], edges: [] }} />);
    expect(screen.getByTestId('graph-container')).toBeInTheDocument();
    expect(mockNetwork).toHaveBeenCalledTimes(1);
  });

  // 2. Rendering with sample graph data (nodes and edges)
  test('renders with sample graph data and initializes vis-network', async () => {
    render(<GraphVisualization graphData={mockGraphData} />);

    await waitFor(() => {
      expect(mockNetwork).toHaveBeenCalledTimes(1);
      const networkInstance = mockNetwork.mock.results[0].value;
      expect(networkInstance.setData).toHaveBeenCalledTimes(1);
      expect(networkInstance.setData).toHaveBeenCalledWith(expect.objectContaining({
        nodes: expect.any(Object), // DataSet instance
        edges: expect.any(Object), // DataSet instance
      }));
    });
  });

  // 3. Node and edge rendering and styling (indirectly via mock)
  test('passes correct options to vis-network for styling', () => {
    render(<GraphVisualization graphData={mockGraphData} />);
    expect(mockNetwork).toHaveBeenCalledWith(
      expect.any(HTMLElement),
      expect.any(Object),
      expect.objectContaining({
        nodes: {
          shape: 'dot',
          font: { multi: 'html', size: 12 },
          color: {
            border: '#2B7CE9',
            background: '#97C2E6',
            highlight: { border: '#2B7CE9', background: '#D2E5FF' },
            hover: { border: '#2B7CE9', background: '#D2E5FF' },
          },
        },
        edges: {
          arrows: 'to',
          color: { inherit: 'from' },
          font: { align: 'middle' },
        },
        physics: {
          enabled: true,
          barnesHut: {
            gravitationalConstant: -2000,
            centralGravity: 0.3,
            springLength: 95,
            springConstant: 0.04,
            damping: 0.09,
            avoidOverlap: 0,
          },
        },
        interaction: {
          hover: true,
          navigationButtons: true,
          keyboard: true,
          zoomView: true,
          dragView: true,
        },
      })
    );
  });

  // 4. User interactions like node selection, dragging, zooming (simulated via mock)
  test('attaches event listeners for user interactions', async () => {
    render(<GraphVisualization graphData={mockGraphData} />);
    await waitFor(() => {
      const networkInstance = mockNetwork.mock.results[0].value;
      expect(networkInstance.on).toHaveBeenCalledWith('selectNode', expect.any(Function));
      expect(networkInstance.on).toHaveBeenCalledWith('click', expect.any(Function));
      expect(networkInstance.on).toHaveBeenCalledWith('hoverNode', expect.any(Function));
      expect(networkInstance.on).toHaveBeenCalledWith('blurNode', expect.any(Function));
      expect(networkInstance.on).toHaveBeenCalledWith('hoverEdge', expect.any(Function));
      expect(networkInstance.on).toHaveBeenCalledWith('blurEdge', expect.any(Function));
    });
  });

  // 5. Handling updates to graph data prop
  test('updates graph data when prop changes', async () => {
    const { rerender } = render(<GraphVisualization graphData={{ nodes: [], edges: [] }} />);
    const networkInstance = mockNetwork.mock.results[0].value;

    const updatedGraphData = {
      nodes: [{ id: 'node3', label: 'Node 3' }],
      edges: [],
    };

    rerender(<GraphVisualization graphData={updatedGraphData} />);

    await waitFor(() => {
      expect(networkInstance.setData).toHaveBeenCalledTimes(2); // Initial + update
      // Verify the second call to setData received the updated data
      const secondCallArgs = networkInstance.setData.mock.calls[1][0];
      expect(secondCallArgs.nodes.get()).toEqual(expect.arrayContaining([
        expect.objectContaining({ id: 'node3' })
      ]));
    });
  });

  // 6. Displaying node/edge details on hover or click (if implemented)
  test('calls onSelectNode when a node is selected', async () => {
    const handleSelectNode = jest.fn();
    render(<GraphVisualization graphData={mockGraphData} onSelectNode={handleSelectNode} />);

    await waitFor(() => {
      const networkInstance = mockNetwork.mock.results[0].value;
      const selectNodeCallback = networkInstance.on.mock.calls.find(call => call[0] === 'selectNode')[1];
      act(() => {
        selectNodeCallback({ nodes: ['node1'], edges: [] });
      });
      expect(handleSelectNode).toHaveBeenCalledWith('node1', 'node');
    });
  });

  test('calls onSelectNode when an edge is selected', async () => {
    const handleSelectNode = jest.fn();
    render(<GraphVisualization graphData={mockGraphData} onSelectNode={handleSelectNode} />);

    await waitFor(() => {
      const networkInstance = mockNetwork.mock.results[0].value;
      const selectNodeCallback = networkInstance.on.mock.calls.find(call => call[0] === 'selectNode')[1];
      act(() => {
        selectNodeCallback({ nodes: [], edges: ['edge1'] });
      });
      expect(handleSelectNode).toHaveBeenCalledWith('edge1', 'edge');
    });
  });

  // 7. Error handling if graph library fails to initialize (simulated by throwing in mock)
  test('handles error if vis-network fails to initialize', () => {
    mockNetwork.mockImplementation(() => {
      throw new Error('Vis-network initialization failed');
    });
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    render(<GraphVisualization graphData={mockGraphData} />);

    expect(screen.getByText('Error loading graph visualization.')).toBeInTheDocument();
    expect(consoleErrorSpy).toHaveBeenCalledWith(expect.stringContaining('Vis-network initialization failed'));
    consoleErrorSpy.mockRestore();
  });

  // 8. Responsiveness or layout adjustments if applicable.
  test('calls network.fit() on initial render and when data changes', async () => {
    const { rerender } = render(<GraphVisualization graphData={mockGraphData} />);
    const networkInstance = mockNetwork.mock.results[0].value;

    await waitFor(() => {
      expect(networkInstance.fit).toHaveBeenCalledTimes(1); // Initial fit
    });

    rerender(<GraphVisualization graphData={{ nodes: [{ id: 'node4', label: 'Node 4' }], edges: [] }} />);

    await waitFor(() => {
      expect(networkInstance.fit).toHaveBeenCalledTimes(2); // Fit after data update
    });
  });

  test('destroys network instance on unmount', () => {
    const { unmount } = render(<GraphVisualization graphData={mockGraphData} />);
    const networkInstance = mockNetwork.mock.results[0].value;
    unmount();
    expect(networkInstance.destroy).toHaveBeenCalledTimes(1);
  });
});
