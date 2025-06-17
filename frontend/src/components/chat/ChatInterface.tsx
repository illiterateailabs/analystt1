import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import { format } from 'date-fns';
import { Send, RefreshCw, ChevronDown, Code, AlertCircle, Image, FileText } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { chatAPI } from '@/lib/api';
import { cn } from '@/lib/utils';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import ErrorBoundary from '@/components/layout/ErrorBoundary';
import { useToast } from '@/hooks/useToast';

// Types
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

interface ChatInterfaceProps {
  initialConversationId?: string;
  initialMessages?: ChatMessage[];
  onConversationCreated?: (conversationId: string) => void;
  onMessageSent?: (message: ChatMessage) => void;
  className?: string;
}

/**
 * ChatInterface component provides a complete chat UI with message history, 
 * input field, and various message types support.
 */
const ChatInterface: React.FC<ChatInterfaceProps> = ({
  initialConversationId,
  initialMessages = [],
  onConversationCreated,
  onMessageSent,
  className,
}) => {
  const router = useRouter();
  const { toast } = useToast();
  
  // State
  const [conversationId, setConversationId] = useState<string | undefined>(initialConversationId);
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  // Auto-resize textarea as content grows
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    setInputValue(textarea.value);
    
    // Reset height to auto to correctly calculate the new height
    textarea.style.height = 'auto';
    
    // Set new height based on scrollHeight, with a max height
    const newHeight = Math.min(textarea.scrollHeight, 200);
    textarea.style.height = `${newHeight}px`;
  };
  
  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback((smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: smooth ? 'smooth' : 'auto',
        block: 'end' 
      });
    }
  }, []);
  
  // Handle scroll events to show/hide scroll-to-bottom button
  const handleScroll = useCallback(() => {
    if (!messageContainerRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = messageContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    setShowScrollButton(!isNearBottom);
  }, []);
  
  // Load conversation messages if conversationId is provided
  useEffect(() => {
    const loadConversation = async () => {
      if (!conversationId) return;
      
      try {
        setIsLoading(true);
        const response = await chatAPI.getChatSession(conversationId);
        setMessages(response.messages || []);
        setIsLoading(false);
      } catch (err) {
        setError('Failed to load conversation');
        setIsLoading(false);
        toast({
          title: 'Error',
          description: 'Failed to load conversation history',
          variant: 'destructive',
        });
      }
    };
    
    loadConversation();
  }, [conversationId, toast]);
  
  // Scroll to bottom on initial load and when messages change
  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    }
  }, [messages, scrollToBottom]);
  
  // Add scroll event listener
  useEffect(() => {
    const messageContainer = messageContainerRef.current;
    if (messageContainer) {
      messageContainer.addEventListener('scroll', handleScroll);
      return () => messageContainer.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll]);
  
  // Focus textarea on mount
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);
  
  // Send message handler
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!inputValue.trim()) return;
    
    const userMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString(),
    };
    
    // Optimistically add user message
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    
    // Show typing indicator
    setIsTyping(true);
    setIsLoading(true);
    setError(null);
    
    try {
      // Send message to API
      const response = await chatAPI.sendMessage(
        userMessage.content,
        true // include graph data
      );
      
      // Hide typing indicator
      setIsTyping(false);
      
      if (response.data) {
        // Update conversation ID if this is a new conversation
        if (!conversationId && response.data.conversation_id) {
          setConversationId(response.data.conversation_id);
          if (onConversationCreated) {
            onConversationCreated(response.data.conversation_id);
          }
        }
        
        // Add assistant response
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date().toISOString(),
          metadata: {
            cypher_query: response.data.cypher_query,
            graph_results: response.data.graph_results,
            execution_details: response.data.execution_details,
          }
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        
        if (onMessageSent) {
          onMessageSent(assistantMessage);
        }
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
      toast({
        title: 'Error',
        description: 'Failed to send message. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
      setIsTyping(false);
      scrollToBottom();
    }
  };
  
  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  // Retry sending message
  const handleRetry = () => {
    setError(null);
    handleSendMessage();
  };
  
  // Render message content with markdown and code highlighting
  const renderMessageContent = (content: string) => {
    return (
      <ReactMarkdown
        className="prose prose-sm max-w-none dark:prose-invert"
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={atomDark}
                language={match[1]}
                PreTag="div"
                className="rounded-md my-2 text-sm"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code className={cn("bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded", className)} {...props}>
                {children}
              </code>
            );
          },
          p({ children }) {
            return <p className="mb-2 last:mb-0">{children}</p>;
          },
          ul({ children }) {
            return <ul className="list-disc pl-6 mb-2">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="list-decimal pl-6 mb-2">{children}</ol>;
          },
          li({ children }) {
            return <li className="mb-1">{children}</li>;
          },
          a({ children, href }) {
            return (
              <a 
                href={href} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-blue-600 dark:text-blue-400 hover:underline"
              >
                {children}
              </a>
            );
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-300 dark:divide-gray-700">
                  {children}
                </table>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };
  
  // Render message bubble
  const MessageBubble = ({ message }: { message: ChatMessage }) => {
    const isUser = message.role === 'user';
    const formattedTime = format(new Date(message.timestamp), 'h:mm a');
    
    return (
      <div 
        className={cn(
          "flex w-full mb-4",
          isUser ? "justify-end" : "justify-start"
        )}
      >
        <div 
          className={cn(
            "flex flex-col max-w-[80%] md:max-w-[70%] rounded-lg px-4 py-2",
            isUser 
              ? "bg-blue-600 text-white rounded-br-none" 
              : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-bl-none"
          )}
        >
          {/* Message content */}
          <div className="break-words">
            {renderMessageContent(message.content)}
          </div>
          
          {/* Metadata sections for assistant messages */}
          {!isUser && message.metadata?.cypher_query && (
            <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
              <details className="text-sm">
                <summary className="font-medium cursor-pointer flex items-center gap-1 text-gray-600 dark:text-gray-300">
                  <Code size={16} />
                  <span>Generated Cypher Query</span>
                </summary>
                <div className="mt-2 bg-gray-800 text-gray-100 p-2 rounded-md overflow-x-auto">
                  <pre className="text-xs">{message.metadata.cypher_query}</pre>
                </div>
              </details>
            </div>
          )}
          
          {!isUser && message.metadata?.graph_results && message.metadata.graph_results.length > 0 && (
            <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
              <details className="text-sm">
                <summary className="font-medium cursor-pointer flex items-center gap-1 text-gray-600 dark:text-gray-300">
                  <FileText size={16} />
                  <span>Graph Results ({message.metadata.graph_results.length} records)</span>
                </summary>
                <div className="mt-2 bg-gray-800 text-gray-100 p-2 rounded-md overflow-x-auto">
                  <pre className="text-xs">{JSON.stringify(message.metadata.graph_results, null, 2)}</pre>
                </div>
              </details>
            </div>
          )}
          
          {/* Message timestamp */}
          <div 
            className={cn(
              "text-xs mt-1 self-end",
              isUser ? "text-gray-100" : "text-gray-500 dark:text-gray-400"
            )}
          >
            {formattedTime}
          </div>
        </div>
      </div>
    );
  };
  
  // Typing indicator component
  const TypingIndicator = () => (
    <div className="flex items-center space-x-2 mb-4">
      <div className="flex space-x-1">
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
      <span className="text-sm text-gray-500">AI is thinking...</span>
    </div>
  );
  
  // Empty state component
  const EmptyState = () => (
    <div className="flex flex-col items-center justify-center h-full text-center p-6">
      <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mb-4">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          className="w-8 h-8 text-blue-600 dark:text-blue-300"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
          />
        </svg>
      </div>
      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
        Start a new conversation
      </h3>
      <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-sm">
        Ask questions about fraud patterns, analyze transactions, or request graph database queries.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-md">
        <button
          className="flex items-center justify-center bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg px-4 py-3 text-sm text-gray-700 dark:text-gray-300 transition-colors"
          onClick={() => setInputValue("Show me recent suspicious transactions")}
        >
          Show me recent suspicious transactions
        </button>
        <button
          className="flex items-center justify-center bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg px-4 py-3 text-sm text-gray-700 dark:text-gray-300 transition-colors"
          onClick={() => setInputValue("Analyze this transaction pattern for fraud")}
        >
          Analyze transaction patterns
        </button>
        <button
          className="flex items-center justify-center bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg px-4 py-3 text-sm text-gray-700 dark:text-gray-300 transition-colors"
          onClick={() => setInputValue("Find connections between these entities")}
        >
          Find entity connections
        </button>
        <button
          className="flex items-center justify-center bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg px-4 py-3 text-sm text-gray-700 dark:text-gray-300 transition-colors"
          onClick={() => setInputValue("Explain structuring patterns in banking")}
        >
          Explain fraud patterns
        </button>
      </div>
    </div>
  );
  
  return (
    <ErrorBoundary>
      <div className={cn("flex flex-col h-full bg-white dark:bg-gray-900", className)}>
        {/* Chat messages area */}
        <div 
          ref={messageContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth"
          onScroll={handleScroll}
        >
          {isLoading && messages.length === 0 ? (
            <div className="flex justify-center items-center h-full">
              <LoadingSpinner size="lg" variant="primary" showLabel label="Loading conversation..." />
            </div>
          ) : messages.length === 0 ? (
            <EmptyState />
          ) : (
            <>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              
              {/* Typing indicator */}
              {isTyping && <TypingIndicator />}
              
              {/* Error message */}
              {error && (
                <div className="flex items-center justify-center">
                  <div className="bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200 px-4 py-2 rounded-lg flex items-center gap-2">
                    <AlertCircle size={16} />
                    <span>{error}</span>
                    <button 
                      onClick={handleRetry}
                      className="ml-2 bg-red-100 dark:bg-red-800 hover:bg-red-200 dark:hover:bg-red-700 text-red-800 dark:text-red-200 px-2 py-1 rounded-md text-xs flex items-center"
                    >
                      <RefreshCw size={12} className="mr-1" />
                      Retry
                    </button>
                  </div>
                </div>
              )}
              
              {/* Invisible element for scrolling to bottom */}
              <div ref={messagesEndRef} />
            </>
          )}
          
          {/* Scroll to bottom button */}
          <AnimatePresence>
            {showScrollButton && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                onClick={() => scrollToBottom()}
                className="absolute bottom-24 right-6 bg-gray-800 dark:bg-gray-700 text-white p-2 rounded-full shadow-lg hover:bg-gray-700 dark:hover:bg-gray-600 transition-colors"
                aria-label="Scroll to bottom"
              >
                <ChevronDown size={20} />
              </motion.button>
            )}
          </AnimatePresence>
        </div>
        
        {/* Message input area */}
        <div className="border-t border-gray-200 dark:border-gray-800 p-4">
          <form onSubmit={handleSendMessage} className="flex flex-col gap-2">
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder="Type your message..."
                className="w-full border border-gray-300 dark:border-gray-700 rounded-lg py-3 px-4 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none min-h-[56px] max-h-[200px] bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                rows={1}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className={cn(
                  "absolute right-3 bottom-3 p-1.5 rounded-md",
                  isLoading || !inputValue.trim() 
                    ? "text-gray-400 cursor-not-allowed" 
                    : "text-blue-600 hover:bg-blue-100 dark:hover:bg-blue-900 transition-colors"
                )}
                aria-label="Send message"
              >
                {isLoading ? (
                  <LoadingSpinner size="sm" variant="primary" />
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
            
            <div className="text-xs text-gray-500 dark:text-gray-400 flex justify-between items-center px-1">
              <span>Press Enter to send, Shift+Enter for new line</span>
              <div className="flex items-center gap-2">
                {/* File upload button (placeholder for future functionality) */}
                <button
                  type="button"
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md transition-colors"
                  aria-label="Upload image"
                  onClick={() => toast({
                    title: "Coming soon",
                    description: "Image upload functionality will be available soon.",
                  })}
                >
                  <Image size={16} />
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default ChatInterface;
