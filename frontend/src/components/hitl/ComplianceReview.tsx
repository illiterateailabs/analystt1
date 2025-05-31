'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { API_BASE_URL } from '@/lib/constants'; // Assuming this file exists
import { getAuthHeaders } from '@/lib/auth'; // Assuming this file exists

interface ReviewDetails {
  review_id: string;
  task_id: string;
  findings: string;
  risk_level: string;
  regulatory_implications: string[];
  details?: Record<string, any>;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  responses: Array<{
    status: string;
    reviewer: string;
    comments?: string;
    timestamp: string;
  }>;
}

interface TaskMetadata {
  crew_name: string;
  inputs: Record<string, any>;
  created_at: string;
  last_updated: string;
  current_agent: string;
  paused_at?: string;
  result?: any;
  error?: string;
}

interface ComplianceReviewProps {
  taskId: string;
}

export default function ComplianceReview({ taskId }: ComplianceReviewProps) {
  const [reviewDetails, setReviewDetails] = useState<ReviewDetails | null>(null);
  const [taskMetadata, setTaskMetadata] = useState<TaskMetadata | null>(null);
  const [comment, setComment] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchReviewDetails = async () => {
      setLoading(true);
      setError(null);
      try {
        const headers = await getAuthHeaders();
        
        // Fetch task metadata
        const taskResponse = await fetch(`${API_BASE_URL}/crew/status/${taskId}`, { headers });
        if (!taskResponse.ok) {
          const errorData = await taskResponse.json();
          throw new Error(errorData.detail || 'Failed to fetch task status.');
        }
        const taskData: TaskMetadata = await taskResponse.json();
        setTaskMetadata(taskData);

        // Fetch review details
        const reviewResponse = await fetch(`${API_BASE_URL}/crew/review/${taskId}`, { headers });
        if (!reviewResponse.ok) {
          const errorData = await reviewResponse.json();
          throw new Error(errorData.detail || 'Failed to fetch review details.');
        }
        const reviewData: ReviewDetails = await reviewResponse.json();
        setReviewDetails(reviewData);

      } catch (err: any) {
        console.error('Error fetching review details:', err);
        setError(err.message || 'An unexpected error occurred while fetching review details.');
      } finally {
        setLoading(false);
      }
    };

    if (taskId) {
      fetchReviewDetails();
    }
  }, [taskId]);

  const handleSubmitReview = async (status: 'approved' | 'rejected') => {
    setSubmitting(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const headers = await getAuthHeaders();
      const userInfo = await (await getAuthHeaders()).user; // Assuming user info is available in headers or a separate call
      const reviewer = userInfo?.email || 'unknown_reviewer'; // Fallback

      const response = await fetch(`${API_BASE_URL}/crew/resume/${taskId}`, {
        method: 'POST',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          status,
          reviewer,
          comments: comment,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to ${status} review.`);
      }

      const data = await response.json();
      setSuccessMessage(`Review ${status} successfully! Task status: ${data.status}`);
      // Optionally, refresh data or redirect
      router.push('/dashboard'); // Or to a reviews list page
    } catch (err: any) {
      console.error(`Error submitting review (${status}):`, err);
      setError(err.message || `An unexpected error occurred during ${status} review.`);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-center">
          <svg className="animate-spin h-10 w-10 text-blue-500 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="mt-3 text-gray-700">Loading review details...</p>
        </div>
      </div>
    );
  }

  if (error && !reviewDetails) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative w-full max-w-md" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      </div>
    );
  }

  if (!reviewDetails || !taskMetadata) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100 p-4">
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative w-full max-w-md" role="alert">
          <strong className="font-bold">Information:</strong>
          <span className="block sm:inline"> No review details found for this task ID.</span>
        </div>
      </div>
    );
  }

  const isReviewPending = reviewDetails.status === 'pending';

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-4xl mx-auto">
        {successMessage && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Success!</strong>
            <span className="block sm:inline"> {successMessage}</span>
          </div>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}

        <div className="bg-white shadow-md rounded-lg overflow-hidden mb-4">
          <div className="p-6">
            <h1 className="text-2xl font-bold text-gray-800 mb-2">Compliance Review</h1>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-sm text-gray-600">Task ID</p>
                <p className="font-medium">{reviewDetails.task_id}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Review ID</p>
                <p className="font-medium">{reviewDetails.review_id}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Crew</p>
                <p className="font-medium">{taskMetadata.crew_name}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Created</p>
                <p className="font-medium">{new Date(taskMetadata.created_at).toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Paused</p>
                <p className="font-medium">{taskMetadata.paused_at ? new Date(taskMetadata.paused_at).toLocaleString() : 'N/A'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Current Agent</p>
                <p className="font-medium">{taskMetadata.current_agent}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white shadow-md rounded-lg overflow-hidden mb-4">
          <div className="p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Risk Assessment</h2>
            <div className="mb-4">
              <div className="flex items-center mb-2">
                <span className="text-sm font-medium text-gray-700 w-32">Risk Level:</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  reviewDetails.risk_level === 'Critical' ? 'bg-red-100 text-red-800' :
                  reviewDetails.risk_level === 'High' ? 'bg-orange-100 text-orange-800' :
                  reviewDetails.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {reviewDetails.risk_level}
                </span>
              </div>
            </div>
            <div className="mb-4">
              <h3 className="text-lg font-semibold mb-2">Regulatory Implications</h3>
              <ul className="list-disc pl-5 space-y-1">
                {reviewDetails.regulatory_implications.map((implication, index) => (
                  <li key={index} className="text-gray-700">{implication}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white shadow-md rounded-lg overflow-hidden mb-4">
          <div className="p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Compliance Findings</h2>
            <div className="prose max-w-none">
              <div className="whitespace-pre-wrap bg-gray-50 p-4 rounded border border-gray-200">
                {reviewDetails.findings}
              </div>
            </div>
          </div>
        </div>

        {reviewDetails.details && (
          <div className="bg-white shadow-md rounded-lg overflow-hidden mb-4">
            <div className="p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Additional Details</h2>
              <pre className="bg-gray-50 p-4 rounded border border-gray-200 overflow-auto">
                {JSON.stringify(reviewDetails.details, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {isReviewPending ? (
          <div className="bg-white shadow-md rounded-lg overflow-hidden">
            <div className="p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Submit Review</h2>
              <div className="mb-4">
                <label htmlFor="comment" className="block text-sm font-medium text-gray-700 mb-2">
                  Comments
                </label>
                <textarea
                  id="comment"
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Add your comments here..."
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                ></textarea>
              </div>
              <div className="flex space-x-4">
                <button
                  onClick={() => handleSubmitReview('approved')}
                  disabled={submitting}
                  className={`px-4 py-2 rounded-md text-white font-medium ${
                    submitting ? 'bg-green-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  {submitting ? 'Processing...' : 'Approve'}
                </button>
                <button
                  onClick={() => handleSubmitReview('rejected')}
                  disabled={submitting}
                  className={`px-4 py-2 rounded-md text-white font-medium ${
                    submitting ? 'bg-red-400 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700'
                  }`}
                >
                  {submitting ? 'Processing...' : 'Reject'}
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white shadow-md rounded-lg overflow-hidden">
            <div className="p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Review History</h2>
              {reviewDetails.responses.length > 0 ? (
                <div className="space-y-4">
                  {reviewDetails.responses.map((response, index) => (
                    <div key={index} className="border-l-4 border-blue-500 pl-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium">{response.reviewer}</span>
                        <span className="text-sm text-gray-500">{new Date(response.timestamp).toLocaleString()}</span>
                      </div>
                      <div className="flex items-center mb-2">
                        <span className="text-sm mr-2">Status:</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          response.status === 'approved' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {response.status.toUpperCase()}
                        </span>
                      </div>
                      {response.comments && (
                        <div className="text-gray-700 bg-gray-50 p-3 rounded">
                          {response.comments}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-600">No review responses yet.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
