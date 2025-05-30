'use client'

import React from 'react'
import PromptsManager from '../../components/prompts/PromptsManager'

export default function PromptsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Agent Prompt Management</h1>
        <p className="mt-2 text-gray-600">
          View and edit system prompts for all agents in the system. Changes will take effect immediately for new agent instances.
        </p>
      </header>
      
      <main>
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <PromptsManager />
        </div>
      </main>
    </div>
  )
}
