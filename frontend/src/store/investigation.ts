import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// Types for the store state
interface ActiveInvestigation {
  id: string;
  title: string;
  entities: string[];
  riskScore: number;
  status: string;
  tags: string[];
}

interface SelectedEntity {
  id: string;
  type: string;
  name: string;
  // Add any other relevant properties for an entity
}

type Theme = 'light' | 'dark' | 'system';
type ChatGraphLinkMode = 'none' | 'highlight' | 'inject';

interface InvestigationState {
  // Global Investigation Context
  activeInvestigation: ActiveInvestigation | null;
  selectedEntities: SelectedEntity[];

  // UI State & User Preferences
  sidebarOpen: boolean;
  contextRibbonExpanded: boolean;
  notifications: number;
  searchOpen: boolean;
  theme: Theme;
  chatGraphLinkMode: ChatGraphLinkMode; // For linking chat and graph interactions

  // Actions
  setActiveInvestigation: (investigation: ActiveInvestigation | null) => void;
  addSelectedEntity: (entity: SelectedEntity) => void;
  removeSelectedEntity: (entityId: string) => void;
  clearSelectedEntities: () => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleContextRibbon: () => void;
  setNotifications: (count: number) => void;
  toggleSearch: () => void;
  setSearchOpen: (open: boolean) => void;
  setTheme: (theme: Theme) => void;
  setChatGraphLinkMode: (mode: ChatGraphLinkMode) => void;
}

// Mock investigation data - consistent with Shell.tsx
const mockActiveInvestigation: ActiveInvestigation = {
  id: 'INV-2025-0042',
  title: 'Cross-Border Transaction Network',
  entities: ['Acme Corp', 'Global Finance Ltd', 'Offshore Holdings'],
  riskScore: 87,
  status: 'In Progress',
  tags: ['Money Laundering', 'Structuring', 'High Risk'],
};

// Create the Zustand store
export const useInvestigationStore = create<InvestigationState>()(
  persist(
    (set, get) => ({
      // Initial State
      activeInvestigation: mockActiveInvestigation,
      selectedEntities: [],
      sidebarOpen: true,
      contextRibbonExpanded: true,
      notifications: 3, // Mock notification count
      searchOpen: false,
      theme: 'system', // Default theme
      chatGraphLinkMode: 'none',

      // Actions
      setActiveInvestigation: (investigation) => set({ activeInvestigation: investigation }),
      addSelectedEntity: (entity) =>
        set((state) => ({
          selectedEntities: [...state.selectedEntities, entity],
        })),
      removeSelectedEntity: (entityId) =>
        set((state) => ({
          selectedEntities: state.selectedEntities.filter((e) => e.id !== entityId),
        })),
      clearSelectedEntities: () => set({ selectedEntities: [] }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleContextRibbon: () => set((state) => ({ contextRibbonExpanded: !state.contextRibbonExpanded })),
      setNotifications: (count) => set({ notifications: count }),
      toggleSearch: () => set((state) => ({ searchOpen: !state.searchOpen })),
      setSearchOpen: (open) => set({ searchOpen: open }),
      setTheme: (theme) => set({ theme: theme }),
      setChatGraphLinkMode: (mode) => set({ chatGraphLinkMode: mode }),
    }),
    {
      name: 'investigation-storage', // name of the item in localStorage
      storage: createJSONStorage(() => localStorage), // Use localStorage
      partialize: (state) => ({
        // Only persist these parts of the state
        sidebarOpen: state.sidebarOpen,
        contextRibbonExpanded: state.contextRibbonExpanded,
        theme: state.theme,
        activeInvestigation: state.activeInvestigation, // Persist active investigation
      }),
    }
  )
);
