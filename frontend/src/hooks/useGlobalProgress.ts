import create from 'zustand';
import { devtools } from 'zustand/middleware';
import { useEffect, useCallback } from 'react';
import { createProgressSocket } from '../lib/api';

// ------------------------- Types -------------------------

export interface ProgressState {
  progress: number;
  status: 'idle' | 'running' | 'completed' | 'failed';
  message?: string;
}

interface GlobalProgressStore {
  tasks: Record<string, ProgressState>;
  overallProgress: number;
  activeTaskCount: number;
  webSockets: Record<string, WebSocket>;

  // Actions
  startTracking: (taskId: string) => void;
  updateTaskProgress: (taskId: string, state: Partial<ProgressState>) => void;
  endTracking: (taskId: string) => void;
  _calculateOverallProgress: () => void;
  _setWebSocket: (taskId: string, ws: WebSocket) => void;
}

// ------------------------- Zustand Store -------------------------

const useProgressStore = create<GlobalProgressStore>()(
  devtools(
    (set, get) => ({
      tasks: {},
      overallProgress: 0,
      activeTaskCount: 0,
      webSockets: {},

      _calculateOverallProgress: () => {
        const tasks = Object.values(get().tasks);
        if (tasks.length === 0) {
          set({ overallProgress: 0, activeTaskCount: 0 });
          return;
        }
        const totalProgress = tasks.reduce((sum, task) => sum + (task.progress || 0), 0);
        const runningTasks = tasks.filter(t => t.status === 'running');
        set({
          overallProgress: runningTasks.length > 0 ? totalProgress / runningTasks.length : 100,
          activeTaskCount: runningTasks.length,
        });
      },

      startTracking: (taskId) => {
        if (get().tasks[taskId]) return; // Already tracking

        const ws = createProgressSocket(taskId);
        get()._setWebSocket(taskId, ws);

        ws.onopen = () => {
          console.log(`[Progress] WebSocket connected for task: ${taskId}`);
          get().updateTaskProgress(taskId, { progress: 0, status: 'running', message: 'Task started...' });
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.task_id === taskId) {
              get().updateTaskProgress(taskId, {
                progress: data.progress,
                status: data.status,
                message: data.message,
              });
              if (data.status === 'completed' || data.status === 'failed') {
                get().endTracking(taskId);
              }
            }
          } catch (error) {
            console.error('[Progress] Error parsing message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error(`[Progress] WebSocket error for task ${taskId}:`, error);
          get().updateTaskProgress(taskId, { status: 'failed', message: 'Connection error' });
          get().endTracking(taskId);
        };

        ws.onclose = () => {
          console.log(`[Progress] WebSocket closed for task: ${taskId}`);
          // Ensure task is marked as completed if it reached 100%
          const taskState = get().tasks[taskId];
          if (taskState && taskState.progress === 100 && taskState.status !== 'completed') {
            get().updateTaskProgress(taskId, { status: 'completed' });
          }
          // The endTracking might have already been called, but this is a safeguard
          setTimeout(() => get().endTracking(taskId), 1000); 
        };
      },

      updateTaskProgress: (taskId, state) => {
        set((store) => ({
          tasks: {
            ...store.tasks,
            [taskId]: {
              ...(store.tasks[taskId] || { progress: 0, status: 'idle' }),
              ...state,
            },
          },
        }));
        get()._calculateOverallProgress();
      },

      endTracking: (taskId) => {
        const ws = get().webSockets[taskId];
        if (ws && ws.readyState < 2) { // OPEN or CONNECTING
          ws.close();
        }
        set((store) => {
          const newTasks = { ...store.tasks };
          delete newTasks[taskId];
          const newWebSockets = { ...store.webSockets };
          delete newWebSockets[taskId];
          return { tasks: newTasks, webSockets: newWebSockets };
        });
        get()._calculateOverallProgress();
      },

      _setWebSocket: (taskId, ws) => {
        set(store => ({
          webSockets: {
            ...store.webSockets,
            [taskId]: ws,
          }
        }))
      },
    }),
    { name: 'global-progress-store' }
  )
);

// ------------------------- React Hook -------------------------

/**
 * A global hook to manage and display progress of long-running backend tasks.
 *
 * @returns {object} - The current overall progress, number of active tasks,
 * and functions to start/end task tracking.
 */
export const useGlobalProgress = () => {
  const {
    overallProgress,
    activeTaskCount,
    startTracking,
    endTracking,
    tasks,
    webSockets,
  } = useProgressStore();

  // Cleanup WebSockets on unmount (e.g., page navigation)
  useEffect(() => {
    return () => {
      Object.keys(webSockets).forEach(taskId => {
        const ws = webSockets[taskId];
        if (ws && ws.readyState < 2) {
          ws.close();
        }
      });
    };
  }, [webSockets]);

  const startTask = useCallback((taskId: string) => {
    startTracking(taskId);
  }, [startTracking]);

  const endTask = useCallback((taskId: string) => {
    endTracking(taskId);
  }, [endTracking]);

  return {
    overallProgress,
    activeTaskCount,
    tasks,
    startTask,
    endTask,
  };
};

export default useGlobalProgress;
