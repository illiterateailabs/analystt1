'use client'

import { useState, useRef, useEffect } from 'react'
import { useMutation } from 'react-query'
import { PaperAirplaneIcon, PhotoIcon } from '@heroicons/react/24/outline'
import { chatAPI, handleAPIError } from '@/lib/api'
import toast from 'react-hot-toast'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  cypher_query?: string
  graph_results?: any[]
  execution_details?: any
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Hello! I\'m your AI analyst assistant. I can help you with data analysis, graph queries, fraud detection, and more. What would you like to explore today?',
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState('')
  const [includeGraphData, setIncludeGraphData] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessageMutation = useMutation(
    (data: { message: string; include_graph_data: boolean }) =>
      chatAPI.sendMessage(data),
    {
      onSuccess: (response) => {
        const assistantMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date(),
          cypher_query: response.data.cypher_query,
          graph_results: response.data.graph_results,
          execution_details: response.data.execution_details,
        }
        setMessages((prev) => [...prev, assistantMessage])
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  const analyzeImageMutation = useMutation(
    ({ file, prompt }: { file: File; prompt?: string }) =>
      chatAPI.analyzeImage(file, prompt, true),
    {
      onSuccess: (response) => {
        const assistantMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: response.data.analysis,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, assistantMessage])
        
        if (response.data.entities?.length > 0) {
          toast.success(`Extracted ${response.data.entities.length} entities from image`)
        }
      },
      onError: (error) => {
        const errorInfo = handleAPIError(error)
        toast.error(errorInfo.message)
      },
    }
  )

  const handleSendMessage = () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    
    sendMessageMutation.mutate({
      message: inputValue,
      include_graph_data: includeGraphData,
    })

    setInputValue('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file')
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: `[Uploaded image: ${file.name}]`,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    
    analyzeImageMutation.mutate({
      file,
      prompt: 'Analyze this image and extract any relevant entities for fraud detection analysis.',
    })

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const renderMessage = (message: Message) => {
    const isUser = message.role === 'user'

    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div
          className={`max-w-3xl px-4 py-3 rounded-lg ${
            isUser
              ? 'bg-primary-600 text-white'
              : 'bg-white border border-gray-200 text-gray-900'
          }`}
        >
          <div className="prose prose-sm max-w-none">
            {isUser ? (
              <p className="text-white">{message.content}</p>
            ) : (
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '')
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>

          {/* Show Cypher query if available */}
          {message.cypher_query && (
            <div className="mt-3 p-3 bg-gray-100 rounded border-l-4 border-blue-500">
              <p className="text-sm font-medium text-gray-700 mb-2">
                Generated Cypher Query:
              </p>
              <SyntaxHighlighter
                language="cypher"
                style={vscDarkPlus}
                customStyle={{ fontSize: '12px', margin: 0 }}
              >
                {message.cypher_query}
              </SyntaxHighlighter>
            </div>
          )}

          {/* Show graph results if available */}
          {message.graph_results && message.graph_results.length > 0 && (
            <div className="mt-3 p-3 bg-green-50 rounded border-l-4 border-green-500">
              <p className="text-sm font-medium text-gray-700 mb-2">
                Query Results ({message.graph_results.length} records):
              </p>
              <div className="text-xs text-gray-600 max-h-32 overflow-y-auto">
                <pre>{JSON.stringify(message.graph_results.slice(0, 3), null, 2)}</pre>
                {message.graph_results.length > 3 && (
                  <p className="mt-2 text-gray-500">
                    ... and {message.graph_results.length - 3} more records
                  </p>
                )}
              </div>
            </div>
          )}

          <div className="mt-2 text-xs opacity-70">
            {message.timestamp.toLocaleTimeString()}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          {messages.map(renderMessage)}
          
          {/* Loading indicator */}
          {(sendMessageMutation.isLoading || analyzeImageMutation.isLoading) && (
            <div className="flex justify-start mb-4">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                <div className="flex items-center space-x-2">
                  <div className="spinner" />
                  <span className="text-gray-500">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 bg-white p-4">
        <div className="max-w-4xl mx-auto">
          {/* Options */}
          <div className="flex items-center mb-3">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeGraphData}
                onChange={(e) => setIncludeGraphData(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="ml-2 text-sm text-gray-600">
                Include graph database context
              </span>
            </label>
          </div>

          {/* Input */}
          <div className="flex items-end space-x-3">
            <div className="flex-1">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about your data, fraud patterns, or analysis..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                rows={3}
                disabled={sendMessageMutation.isLoading}
              />
            </div>
            
            <div className="flex flex-col space-y-2">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-3 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                title="Upload image"
              >
                <PhotoIcon className="h-6 w-6" />
              </button>
              
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || sendMessageMutation.isLoading}
                className="p-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <PaperAirplaneIcon className="h-6 w-6" />
              </button>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
      </div>
    </div>
  )
}
