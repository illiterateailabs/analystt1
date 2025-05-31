'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import { getUserInfo, getAuthHeaders } from '@/lib/auth';
import { API_BASE_URL, USER_ROLES, TASK_STATUS, ROUTES, REVIEW_STATUS } from '@/lib/constants';

interface Task {
  task_id: string;
  crew_name: string;
  state: string;
  last_updated: string;
  current_agent?: string;
  error?: string;
  review_id?: string;
}

interface Review {
  review_id: string;
  task_id: string;
  findings: string;
  risk_level: string;
  status: string;
  created_at: string;
}

export default function DashboardPage() {
  const userInfo = getUserInfo();
  const [pendingReviews, setPendingReviews] = useState<Review[]>([]);
  const [recentTasks, setRecentTasks] = useState<Task[]>([]);
  const [loadingReviews, setLoadingReviews] = useState(true);
  const [loadingTasks, setLoadingTasks] = useState(true);
  const [errorReviews, setErrorReviews] = useState<string | null>(null);
  const [errorTasks, setErrorTasks] = useState<string | null>(null);

  useEffect(() => {
    const fetchPendingReviews = async () => {
      setLoadingReviews(true);
      setErrorReviews(null);
      try {
        const headers = await getAuthHeaders();
        const response = await fetch(`${API_BASE_URL}/crew/reviews`, { headers }); // Assuming an endpoint to list all reviews
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch pending reviews.');
        }
        const data = await response.json();
        const filteredReviews = data.reviews.filter((review: Review) => review.status === REVIEW_STATUS.PENDING);
        setPendingReviews(filteredReviews);
      } catch (err: any) {
        console.error('Error fetching pending reviews:', err);
        setErrorReviews(err.message || 'An unexpected error occurred while fetching pending reviews.');
      } finally {
        setLoadingReviews(false);
      }
    };

    const fetchRecentTasks = async () => {
      setLoadingTasks(true);
      setErrorTasks(null);
      try {
        const headers = await getAuthHeaders();
        const response = await fetch(`${API_BASE_URL}/crew/tasks`, { headers }); // Assuming an endpoint to list all tasks
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch recent tasks.');
        }
        const data = await response.json();
        // Sort by last_updated and take a few recent ones
        const sortedTasks = data.tasks.sort((a: Task, b: Task) => new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime());
        setRecentTasks(sortedTasks.slice(0, 5)); // Show top 5 recent tasks
      } catch (err: any) {
        console.error('Error fetching recent tasks:', err);
        setErrorTasks(err.message || 'An unexpected error occurred while fetching recent tasks.');
      } finally {
        setLoadingTasks(false);
      }
    };

    if (userInfo) {
      fetchPendingReviews();
      fetchRecentTasks();
    }
  }, [userInfo]);

  if (!userInfo) {
    // ProtectedRoute will handle redirection if not authenticated
    return null;
  }

  const isAdmin = userInfo.role === USER_ROLES.ADMIN;
  const isCompliance = userInfo.role === USER_ROLES.COMPLIANCE;

  return (
    <ProtectedRoute roles={[USER_ROLES.ADMIN, USER_ROLES.ANALYST, USER_ROLES.COMPLIANCE, USER_ROLES.AUDITOR]}>
      <div className="min-h-screen bg-gray-100 p-4 sm:p-6 lg:p-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Dashboard</h1>

          <div className="bg-white shadow-md rounded-lg p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">Welcome, {userInfo.sub}!</h2>
            <p className="text-gray-600">Your role: <span className="font-medium capitalize">{userInfo.role}</span></p>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            <Link href={ROUTES.CHAT} className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-6 rounded-lg shadow-md flex items-center justify-center transition duration-300">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
              Run New Crew
            </Link>
            {isCompliance && (
              <Link href={ROUTES.REVIEWS} className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-4 px-6 rounded-lg shadow-md flex items-center justify-center transition duration-300">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
                View All Reviews
              </Link>
            )}
            <Link href={ROUTES.GRAPH} className="bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-6 rounded-lg shadow-md flex items-center justify-center transition duration-300">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path></svg>
              Graph Visualization
            </Link>
            {isAdmin && (
              <Link href={ROUTES.PROMPTS} className="bg-amber-600 hover:bg-amber-700 text-white font-bold py-4 px-6 rounded-lg shadow-md flex items-center justify-center transition duration-300">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                Manage Prompts
              </Link>
            )}
          </div>

          {/* Pending Reviews Section */}
          <div className="bg-white shadow-md rounded-lg overflow-hidden mb-6">
            <div className="p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Pending Compliance Reviews</h2>
              {loadingReviews ? (
                <div className="flex justify-center items-center py-8">
                  <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </div>
              ) : errorReviews ? (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                  <strong className="font-bold">Error!</strong>
                  <span className="block sm:inline"> {errorReviews}</span>
                </div>
              ) : pendingReviews.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p>No pending reviews at this time.</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Review ID</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Task ID</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk Level</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {pendingReviews.map((review) => (
                        <tr key={review.review_id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{review.review_id}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{review.task_id}</td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              review.risk_level === 'Critical' ? 'bg-red-100 text-red-800' :
                              review.risk_level === 'High' ? 'bg-orange-100 text-orange-800' :
                              review.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-green-100 text-green-800'
                            }`}>
                              {review.risk_level}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {new Date(review.created_at).toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <Link href={ROUTES.REVIEW_DETAIL(review.task_id)} className="text-blue-600 hover:text-blue-900">
                              Review
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          {/* Recent Tasks Section */}
          <div className="bg-white shadow-md rounded-lg overflow-hidden">
            <div className="p-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Recent Crew Tasks</h2>
              {loadingTasks ? (
                <div className="flex justify-center items-center py-8">
                  <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </div>
              ) : errorTasks ? (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                  <strong className="font-bold">Error!</strong>
                  <span className="block sm:inline"> {errorTasks}</span>
                </div>
              ) : recentTasks.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p>No recent tasks found.</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Task ID</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Crew</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Updated</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Current Agent</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {recentTasks.map((task) => (
                        <tr key={task.task_id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{task.task_id}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{task.crew_name}</td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              task.state === TASK_STATUS.RUNNING ? 'bg-blue-100 text-blue-800' :
                              task.state === TASK_STATUS.COMPLETED ? 'bg-green-100 text-green-800' :
                              task.state === TASK_STATUS.PAUSED ? 'bg-yellow-100 text-yellow-800' :
                              task.state === TASK_STATUS.FAILED ? 'bg-red-100 text-red-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {task.state}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {new Date(task.last_updated).toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {task.current_agent || 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          {/* Admin-only Section */}
          {isAdmin && (
            <div className="bg-white shadow-md rounded-lg overflow-hidden mt-6">
              <div className="p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Admin Controls</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <Link href="/admin/users" className="bg-gray-100 hover:bg-gray-200 p-4 rounded-lg flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                    <span className="text-gray-900 font-medium">User Management</span>
                  </Link>
                  <Link href="/admin/metrics" className="bg-gray-100 hover:bg-gray-200 p-4 rounded-lg flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span className="text-gray-900 font-medium">System Metrics</span>
                  </Link>
                  <Link href="/admin/settings" className="bg-gray-100 hover:bg-gray-200 p-4 rounded-lg flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <span className="text-gray-900 font-medium">System Settings</span>
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}
