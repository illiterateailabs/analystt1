import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTheme } from 'next-themes';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard, 
  FolderOpen, 
  Network, 
  MessageSquare, 
  Lightbulb, 
  Settings, 
  Sun, 
  Moon, 
  ChevronLeft, 
  ChevronRight, 
  User, 
  Search, 
  AlertTriangle, 
  Bell, 
  LogOut,
  FileText,
  ExternalLink,
  Download,
  Share2,
  ShieldAlert
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/useToast';
import ErrorBoundary from './ErrorBoundary';

interface ShellProps {
  children: React.ReactNode;
}

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Investigations', href: '/investigations', icon: FolderOpen },
  { name: 'Graph Explorer', href: '/graph', icon: Network },
  { name: 'Analysis Chat', href: '/analysis', icon: MessageSquare },
  { name: 'Prompts', href: '/prompts', icon: Lightbulb },
  { name: 'Settings', href: '/settings', icon: Settings },
];

// Mock investigation data - would come from a global state store in production
const mockActiveInvestigation = {
  id: 'INV-2025-0042',
  title: 'Cross-Border Transaction Network',
  entities: ['Acme Corp', 'Global Finance Ltd', 'Offshore Holdings'],
  riskScore: 87,
  status: 'In Progress',
  tags: ['Money Laundering', 'Structuring', 'High Risk'],
};

const Shell: React.FC<ShellProps> = ({ children }) => {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const { user, logout } = useAuth();
  const { toast } = useToast();
  
  // State
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [contextRibbonExpanded, setContextRibbonExpanded] = useState(true);
  const [activeInvestigation, setActiveInvestigation] = useState(mockActiveInvestigation);
  const [notifications, setNotifications] = useState(3); // Mock notification count
  const [searchOpen, setSearchOpen] = useState(false);
  
  // Toggle sidebar
  const toggleSidebar = useCallback(() => {
    setSidebarOpen(prev => !prev);
  }, []);
  
  // Toggle theme
  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
    toast({
      description: `Switched to ${theme === 'dark' ? 'light' : 'dark'} mode`,
      variant: 'info',
    });
  }, [theme, setTheme, toast]);
  
  // Toggle context ribbon
  const toggleContextRibbon = useCallback(() => {
    setContextRibbonExpanded(prev => !prev);
  }, []);
  
  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K - Global search
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen(true);
      }
      
      // Cmd/Ctrl + B - Toggle sidebar
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
      
      // Cmd/Ctrl + Shift + D - Toggle theme
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'd') {
        e.preventDefault();
        toggleTheme();
      }
      
      // Cmd/Ctrl + Shift + G - Focus graph (would be implemented in the workspace component)
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'g') {
        e.preventDefault();
        toast({
          description: 'Graph focus shortcut triggered',
          variant: 'info',
        });
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleSidebar, toggleTheme, toast]);
  
  // Risk score color based on value
  const getRiskScoreColor = (score: number) => {
    if (score >= 80) return 'bg-red-600 text-white';
    if (score >= 60) return 'bg-orange-500 text-white';
    if (score >= 40) return 'bg-yellow-400 text-gray-900';
    return 'bg-green-500 text-white';
  };
  
  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{ width: sidebarOpen ? '240px' : '64px' }}
        transition={{ duration: 0.2 }}
        className="flex-shrink-0 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex flex-col z-20 shadow-sm"
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 dark:border-gray-800">
          {sidebarOpen ? (
            <Link href="/" className="flex items-center gap-2">
              <ShieldAlert className="h-8 w-8 text-blue-600 dark:text-blue-400" />
              <span className="text-xl font-bold text-blue-600 dark:text-blue-400">Analyst Agent</span>
            </Link>
          ) : (
            <Link href="/" className="flex items-center justify-center w-full">
              <ShieldAlert className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            </Link>
          )}
          <button
            onClick={toggleSidebar}
            className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? <ChevronLeft className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
          </button>
        </div>
        
        {/* Navigation Links */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = pathname?.startsWith(item.href);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800',
                  sidebarOpen ? 'justify-start' : 'justify-center'
                )}
                aria-current={isActive ? 'page' : undefined}
                title={!sidebarOpen ? item.name : undefined}
              >
                <item.icon className={cn('flex-shrink-0', sidebarOpen ? 'mr-3 h-5 w-5' : 'h-6 w-6')} />
                {sidebarOpen && <span>{item.name}</span>}
              </Link>
            );
          })}
        </nav>
        
        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800 space-y-2">
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className={cn(
              'flex items-center w-full px-3 py-2 rounded-md text-sm font-medium transition-colors',
              'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
            )}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <>
                <Sun className={cn('flex-shrink-0', sidebarOpen ? 'mr-3 h-5 w-5' : 'h-6 w-6')} />
                {sidebarOpen && <span>Light Mode</span>}
              </>
            ) : (
              <>
                <Moon className={cn('flex-shrink-0', sidebarOpen ? 'mr-3 h-5 w-5' : 'h-6 w-6')} />
                {sidebarOpen && <span>Dark Mode</span>}
              </>
            )}
          </button>
          
          {/* User Profile - Simplified when collapsed */}
          {user && (
            <div className={cn(
              'flex items-center justify-between px-3 py-2 rounded-md',
              'text-gray-700 dark:text-gray-300',
              !sidebarOpen && 'justify-center'
            )}>
              {sidebarOpen ? (
                <>
                  <div className="flex items-center">
                    <div className="h-8 w-8 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center text-blue-600 dark:text-blue-400 mr-3">
                      {user.username?.charAt(0).toUpperCase() || 'U'}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium">{user.username}</span>
                      <span className="text-xs text-gray-500 dark:text-gray-400 truncate">
                        {user.email}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={logout}
                    className="p-1 rounded-md text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
                    aria-label="Logout"
                  >
                    <LogOut className="h-4 w-4" />
                  </button>
                </>
              ) : (
                <button
                  onClick={logout}
                  className="p-1 rounded-md text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
                  aria-label="Logout"
                >
                  <LogOut className="h-5 w-5" />
                </button>
              )}
            </div>
          )}
        </div>
      </motion.div>
      
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Navigation Bar */}
        <header className="h-16 flex items-center justify-between px-4 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 z-10">
          {/* Left: Page Title / Breadcrumbs */}
          <div className="flex items-center">
            <h1 className="text-lg font-medium">
              {navigation.find(item => pathname?.startsWith(item.href))?.name || 'Dashboard'}
            </h1>
          </div>
          
          {/* Center: Mini Case Switcher (Active Investigation Context) */}
          {activeInvestigation && (
            <div className="hidden md:flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                Active Case:
              </span>
              <button className="flex items-center px-3 py-1 rounded-md bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 text-sm font-medium hover:bg-blue-100 dark:hover:bg-blue-800/50 transition-colors">
                <span>{activeInvestigation.id}</span>
                <ChevronRight className="ml-1 h-4 w-4" />
              </button>
            </div>
          )}
          
          {/* Right: Action Icons */}
          <div className="flex items-center space-x-3">
            {/* Search Button */}
            <button
              onClick={() => setSearchOpen(true)}
              className="p-2 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              aria-label="Search"
            >
              <Search className="h-5 w-5" />
            </button>
            
            {/* Notifications */}
            <button className="p-2 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 relative">
              <Bell className="h-5 w-5" />
              {notifications > 0 && (
                <span className="absolute top-1 right-1 h-4 w-4 rounded-full bg-red-500 text-white text-xs flex items-center justify-center">
                  {notifications}
                </span>
              )}
            </button>
            
            {/* User Menu (Mobile Only) */}
            <div className="md:hidden">
              <button className="p-2 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800">
                <User className="h-5 w-5" />
              </button>
            </div>
          </div>
        </header>
        
        {/* Context Ribbon - Shows active investigation details */}
        {activeInvestigation && (
          <div className={cn(
            'bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 transition-all',
            contextRibbonExpanded ? 'py-3' : 'py-1'
          )}>
            <div className="px-4 flex items-center justify-between">
              <div className="flex items-center space-x-4">
                {/* Investigation Title & ID */}
                <div>
                  <h2 className="text-lg font-semibold">{activeInvestigation.title}</h2>
                  {contextRibbonExpanded && (
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      ID: {activeInvestigation.id} | Status: {activeInvestigation.status}
                    </p>
                  )}
                </div>
                
                {/* Risk Score Badge */}
                <div className="flex flex-col items-center">
                  <div className={cn(
                    'px-2 py-1 rounded-md text-sm font-medium',
                    getRiskScoreColor(activeInvestigation.riskScore)
                  )}>
                    Risk: {activeInvestigation.riskScore}
                  </div>
                </div>
              </div>
              
              {/* Right side: Tags & Actions */}
              <div className="flex items-center space-x-3">
                {/* Tags */}
                {contextRibbonExpanded && (
                  <div className="hidden md:flex items-center space-x-2">
                    {activeInvestigation.tags.map((tag, index) => (
                      <span 
                        key={index} 
                        className="px-2 py-1 rounded-md bg-gray-100 dark:bg-gray-800 text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                
                {/* Quick Actions */}
                <div className="flex items-center space-x-1">
                  {contextRibbonExpanded && (
                    <>
                      <button className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800" title="Export Report">
                        <Download className="h-4 w-4" />
                      </button>
                      <button className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800" title="Share">
                        <Share2 className="h-4 w-4" />
                      </button>
                      <button className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800" title="Open External">
                        <ExternalLink className="h-4 w-4" />
                      </button>
                    </>
                  )}
                  <button 
                    onClick={toggleContextRibbon}
                    className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
                    title={contextRibbonExpanded ? "Collapse" : "Expand"}
                  >
                    {contextRibbonExpanded ? (
                      <ChevronLeft className="h-4 w-4 rotate-90" />
                    ) : (
                      <ChevronRight className="h-4 w-4 rotate-90" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Page Content - Wrapped in ErrorBoundary */}
        <main className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-950 relative">
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
        </main>
      </div>
      
      {/* Global Search Modal */}
      <AnimatePresence>
        {searchOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40"
              onClick={() => setSearchOpen(false)}
            />
            
            {/* Search Modal */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
              className="fixed top-20 left-1/2 transform -translate-x-1/2 w-full max-w-2xl bg-white dark:bg-gray-900 rounded-lg shadow-xl z-50 overflow-hidden"
            >
              <div className="p-4">
                <div className="flex items-center border-b border-gray-200 dark:border-gray-800 pb-4">
                  <Search className="h-5 w-5 text-gray-500 dark:text-gray-400 mr-3" />
                  <input
                    type="text"
                    placeholder="Search cases, entities, transactions..."
                    className="flex-1 bg-transparent border-0 focus:ring-0 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 text-lg"
                    autoFocus
                  />
                  <button
                    onClick={() => setSearchOpen(false)}
                    className="p-1 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                  >
                    <span className="text-sm">ESC</span>
                  </button>
                </div>
                
                {/* Search Results Placeholder */}
                <div className="py-4 text-center text-gray-500 dark:text-gray-400">
                  <p>Type to search across cases, entities, and transactions</p>
                  <p className="text-sm mt-2">Press <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">↵</kbd> to select</p>
                </div>
              </div>
              
              {/* Keyboard Shortcuts Help */}
              <div className="bg-gray-50 dark:bg-gray-800 px-4 py-3 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center justify-between">
                  <span>Keyboard Shortcuts:</span>
                  <div className="flex space-x-4">
                    <div className="flex items-center">
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm mr-1">⌘</kbd>
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm">K</kbd>
                      <span className="ml-2">Search</span>
                    </div>
                    <div className="flex items-center">
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm mr-1">⌘</kbd>
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm">B</kbd>
                      <span className="ml-2">Toggle Sidebar</span>
                    </div>
                    <div className="flex items-center">
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm mr-1">⌘</kbd>
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm mr-1">⇧</kbd>
                      <kbd className="px-1 bg-white dark:bg-gray-700 rounded shadow-sm">D</kbd>
                      <span className="ml-2">Toggle Theme</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
      
      {/* Keyboard Shortcut Overlay - Hidden by default */}
      <div className="fixed bottom-4 right-4 z-10 hidden">
        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-3 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center space-x-2">
            <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">⌘K</kbd>
            <span>Search</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Shell;
