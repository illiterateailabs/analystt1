'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Search, Home, BarChart2, MessageSquare, Database, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');

  // Main navigation links for the application
  const mainLinks = [
    { href: '/', label: 'Home', icon: <Home className="w-5 h-5 mr-2" /> },
    { href: '/dashboard', label: 'Dashboard', icon: <BarChart2 className="w-5 h-5 mr-2" /> },
    { href: '/analysis', label: 'Analysis', icon: <Database className="w-5 h-5 mr-2" /> },
    { href: '/chat', label: 'Chat', icon: <MessageSquare className="w-5 h-5 mr-2" /> },
  ];

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Navigate to search results page with the query
      router.push(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const goBack = () => {
    router.back();
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 px-4 dark:bg-gray-900">
      <div className="w-full max-w-md p-8 space-y-8 bg-white rounded-lg shadow-lg dark:bg-gray-800">
        <div className="text-center">
          <h1 className="text-6xl font-bold text-gray-300 dark:text-gray-700">404</h1>
          <h2 className="mt-2 text-2xl font-semibold text-gray-900 dark:text-gray-100">Page Not Found</h2>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>

        {/* Search functionality */}
        <form onSubmit={handleSearch} className="mt-8">
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for something else..."
              className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            />
            <button
              type="submit"
              className="absolute inset-y-0 right-0 flex items-center px-4 text-gray-700 bg-gray-100 border border-gray-300 rounded-r-md hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-500"
            >
              <Search className="w-5 h-5" />
            </button>
          </div>
        </form>

        {/* Navigation links */}
        <div className="mt-8">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">
            Navigate to a main section:
          </h3>
          <div className="grid grid-cols-2 gap-3 mt-4">
            {mainLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
              >
                {link.icon}
                {link.label}
              </Link>
            ))}
          </div>
        </div>

        {/* Back button */}
        <button
          onClick={goBack}
          className="flex items-center justify-center w-full px-4 py-2 mt-8 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:bg-blue-700 dark:hover:bg-blue-800"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Go Back
        </button>
      </div>

      <p className="mt-8 text-sm text-gray-500 dark:text-gray-400">
        If you believe this is an error, please contact support.
      </p>
    </div>
  );
}
