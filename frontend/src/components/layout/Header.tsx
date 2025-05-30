'use client'

import { Fragment } from 'react'
import { Menu, Transition } from '@headlessui/react'
import {
  Bars3Icon,
  BellIcon,
  UserCircleIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
} from '@heroicons/react/24/outline'
import { useQuery } from 'react-query'
import { systemAPI } from '@/lib/api'

interface HeaderProps {
  activeView: string
  onSidebarToggle: () => void
}

export function Header({ activeView, onSidebarToggle }: HeaderProps) {
  const { data: healthData } = useQuery(
    'health',
    systemAPI.healthCheck,
    {
      refetchInterval: 30000, // Check every 30 seconds
      retry: false,
    }
  )

  const getViewTitle = (view: string) => {
    switch (view) {
      case 'chat':
        return 'AI Chat Assistant'
      case 'graph':
        return 'Graph Database'
      case 'analysis':
        return 'Data Analysis'
      case 'sandbox':
        return 'Code Sandbox'
      default:
        return 'Analyst\'s Augmentation Agent'
    }
  }

  const getServiceStatus = (service: string) => {
    const status = healthData?.data?.services?.[service]
    if (status === 'connected' || status === 'initialized') {
      return 'online'
    }
    return 'offline'
  }

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="flex items-center justify-between px-4 py-3">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <button
            onClick={onSidebarToggle}
            className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          
          <div>
            <h1 className="text-xl font-semibold text-gray-900">
              {getViewTitle(activeView)}
            </h1>
            <p className="text-sm text-gray-500">
              AI-powered analyst workflow augmentation
            </p>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Service status indicators */}
          <div className="hidden md:flex items-center space-x-3">
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  getServiceStatus('gemini') === 'online'
                    ? 'bg-green-400'
                    : 'bg-red-400'
                }`}
              />
              <span className="text-xs text-gray-500">Gemini</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  getServiceStatus('neo4j') === 'online'
                    ? 'bg-green-400'
                    : 'bg-red-400'
                }`}
              />
              <span className="text-xs text-gray-500">Neo4j</span>
            </div>
            
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  getServiceStatus('e2b') === 'online'
                    ? 'bg-green-400'
                    : 'bg-red-400'
                }`}
              />
              <span className="text-xs text-gray-500">e2b</span>
            </div>
          </div>

          {/* Notifications */}
          <button className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500">
            <BellIcon className="h-6 w-6" />
          </button>

          {/* User menu */}
          <Menu as="div" className="relative">
            <Menu.Button className="flex items-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500">
              <UserCircleIcon className="h-8 w-8" />
            </Menu.Button>

            <Transition
              as={Fragment}
              enter="transition ease-out duration-100"
              enterFrom="transform opacity-0 scale-95"
              enterTo="transform opacity-100 scale-100"
              leave="transition ease-in duration-75"
              leaveFrom="transform opacity-100 scale-100"
              leaveTo="transform opacity-0 scale-95"
            >
              <Menu.Items className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
                <div className="py-1">
                  <Menu.Item>
                    {({ active }) => (
                      <a
                        href="#"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } flex items-center px-4 py-2 text-sm text-gray-700`}
                      >
                        <UserCircleIcon className="mr-3 h-5 w-5" />
                        Profile
                      </a>
                    )}
                  </Menu.Item>
                  
                  <Menu.Item>
                    {({ active }) => (
                      <a
                        href="#"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } flex items-center px-4 py-2 text-sm text-gray-700`}
                      >
                        <Cog6ToothIcon className="mr-3 h-5 w-5" />
                        Settings
                      </a>
                    )}
                  </Menu.Item>
                  
                  <div className="border-t border-gray-100" />
                  
                  <Menu.Item>
                    {({ active }) => (
                      <a
                        href="#"
                        className={`${
                          active ? 'bg-gray-100' : ''
                        } flex items-center px-4 py-2 text-sm text-gray-700`}
                      >
                        <ArrowRightOnRectangleIcon className="mr-3 h-5 w-5" />
                        Sign out
                      </a>
                    )}
                  </Menu.Item>
                </div>
              </Menu.Items>
            </Transition>
          </Menu>
        </div>
      </div>
    </header>
  )
}
