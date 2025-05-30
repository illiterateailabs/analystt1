'use client'

import { Fragment } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import clsx from 'clsx'

interface View {
  id: string
  name: string
  icon: React.ComponentType<{ className?: string }>
  description: string
}

interface SidebarProps {
  views: View[]
  activeView: string
  onViewChange: (view: string) => void
  isOpen: boolean
  onToggle: () => void
}

export function Sidebar({
  views,
  activeView,
  onViewChange,
  isOpen,
  onToggle,
}: SidebarProps) {
  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo/Brand */}
      <div className="flex items-center px-6 py-4 border-b border-gray-200">
        <div className="flex items-center">
          <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">AA</span>
          </div>
          <div className="ml-3">
            <h2 className="text-lg font-semibold text-gray-900">
              Analyst Agent
            </h2>
            <p className="text-xs text-gray-500">v1.0.0</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {views.map((view) => {
          const Icon = view.icon
          const isActive = activeView === view.id

          return (
            <button
              key={view.id}
              onClick={() => onViewChange(view.id)}
              className={clsx(
                'w-full flex items-center px-3 py-3 text-left rounded-lg transition-colors duration-200',
                isActive
                  ? 'bg-primary-50 text-primary-700 border border-primary-200'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              )}
            >
              <Icon
                className={clsx(
                  'h-6 w-6 mr-3',
                  isActive ? 'text-primary-600' : 'text-gray-400'
                )}
              />
              <div className="flex-1">
                <div className="font-medium">{view.name}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {view.description}
                </div>
              </div>
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          <div className="flex items-center justify-between mb-2">
            <span>System Status</span>
            <span className="inline-flex items-center">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-1" />
              Online
            </span>
          </div>
          <div className="text-gray-400">
            Powered by Gemini, Neo4j & e2b.dev
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop sidebar */}
      <div
        className={clsx(
          'hidden lg:flex lg:flex-col lg:w-80 lg:fixed lg:inset-y-0 bg-white border-r border-gray-200 transition-transform duration-300',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <SidebarContent />
      </div>

      {/* Mobile sidebar */}
      <Transition.Root show={isOpen} as={Fragment}>
        <Dialog as="div" className="relative z-50 lg:hidden" onClose={onToggle}>
          <Transition.Child
            as={Fragment}
            enter="transition-opacity ease-linear duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="transition-opacity ease-linear duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-900/80" />
          </Transition.Child>

          <div className="fixed inset-0 flex">
            <Transition.Child
              as={Fragment}
              enter="transition ease-in-out duration-300 transform"
              enterFrom="-translate-x-full"
              enterTo="translate-x-0"
              leave="transition ease-in-out duration-300 transform"
              leaveFrom="translate-x-0"
              leaveTo="-translate-x-full"
            >
              <Dialog.Panel className="relative mr-16 flex w-full max-w-xs flex-1">
                <Transition.Child
                  as={Fragment}
                  enter="ease-in-out duration-300"
                  enterFrom="opacity-0"
                  enterTo="opacity-100"
                  leave="ease-in-out duration-300"
                  leaveFrom="opacity-100"
                  leaveTo="opacity-0"
                >
                  <div className="absolute left-full top-0 flex w-16 justify-center pt-5">
                    <button
                      type="button"
                      className="-m-2.5 p-2.5"
                      onClick={onToggle}
                    >
                      <span className="sr-only">Close sidebar</span>
                      <XMarkIcon
                        className="h-6 w-6 text-white"
                        aria-hidden="true"
                      />
                    </button>
                  </div>
                </Transition.Child>

                <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white px-6 pb-2">
                  <SidebarContent />
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </Dialog>
      </Transition.Root>

      {/* Spacer for desktop when sidebar is open */}
      <div
        className={clsx(
          'hidden lg:block transition-all duration-300',
          isOpen ? 'lg:w-80' : 'lg:w-0'
        )}
      />
    </>
  )
}
