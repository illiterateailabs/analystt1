'use client'

import { useState } from 'react'
import { ChatInterface } from '@/components/chat/ChatInterface'
import { GraphVisualization } from '@/components/graph/GraphVisualization'
import { AnalysisPanel } from '@/components/analysis/AnalysisPanel'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { 
  ChatBubbleLeftRightIcon, 
  ChartBarIcon, 
  CircleStackIcon,
  CpuChipIcon 
} from '@heroicons/react/24/outline'

type ActiveView = 'chat' | 'graph' | 'analysis' | 'sandbox'

export default function Home() {
  const [activeView, setActiveView] = useState<ActiveView>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const views = [
    {
      id: 'chat' as const,
      name: 'Chat',
      icon: ChatBubbleLeftRightIcon,
      description: 'Natural language interaction with AI'
    },
    {
      id: 'graph' as const,
      name: 'Graph',
      icon: CircleStackIcon,
      description: 'Graph database visualization and queries'
    },
    {
      id: 'analysis' as const,
      name: 'Analysis',
      icon: ChartBarIcon,
      description: 'Data analysis and fraud detection'
    },
    {
      id: 'sandbox' as const,
      name: 'Sandbox',
      icon: CpuChipIcon,
      description: 'Code execution environment'
    }
  ]

  const renderActiveView = () => {
    switch (activeView) {
      case 'chat':
        return <ChatInterface />
      case 'graph':
        return <GraphVisualization />
      case 'analysis':
        return <AnalysisPanel />
      case 'sandbox':
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <CpuChipIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">Sandbox Environment</h3>
              <p className="mt-1 text-sm text-gray-500">
                Code execution environment coming soon...
              </p>
            </div>
          </div>
        )
      default:
        return <ChatInterface />
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar
        views={views}
        activeView={activeView}
        onViewChange={setActiveView}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header
          activeView={activeView}
          onSidebarToggle={() => setSidebarOpen(!sidebarOpen)}
        />

        {/* Main content area */}
        <main className="flex-1 overflow-hidden">
          {renderActiveView()}
        </main>
      </div>
    </div>
  )
}
