// @ts-check
const { withSentryConfig } = require('@sentry/nextjs');
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

/**
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false, // Remove X-Powered-By header for security
  
  // Improve production build performance
  swcMinify: true, // Use SWC for minification (faster than Terser)
  
  // Code splitting and chunking strategy
  experimental: {
    optimizeCss: true, // Extract and optimize CSS
    optimizePackageImports: ['@heroicons/react', 'lucide-react', 'recharts', 'd3'],
    serverActions: {
      bodySizeLimit: '2mb', // Limit server action payload size
    },
  },
  
  // Configure module/nomodule for better browser compatibility
  productionBrowserSourceMaps: process.env.NODE_ENV === 'production', // Enable source maps in production for Sentry
  
  // Image optimization configuration
  images: {
    domains: [
      'localhost', // Allow local images
      'analyst-agent-storage.s3.amazonaws.com', // Example S3 bucket
    ],
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 60, // Cache optimized images for 60 seconds minimum
  },
  
  // Configure headers for security and caching
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
        ],
      },
      {
        // Cache static assets longer
        source: '/static/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        // Cache images
        source: '/images/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=86400, stale-while-revalidate=31536000',
          },
        ],
      },
    ];
  },
  
  // Configure redirects
  async redirects() {
    return [
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
    ];
  },
  
  // Webpack configuration for optimizations and customizations
  webpack: (config, { dev, isServer }) => {
    // Only run in production client builds
    if (!dev && !isServer) {
      // Enable tree shaking and dead code elimination
      config.optimization.usedExports = true;
      
      // Split chunks more granularly
      config.optimization.splitChunks = {
        chunks: 'all',
        maxInitialRequests: 25,
        minSize: 20000,
        cacheGroups: {
          default: false,
          vendors: false,
          framework: {
            name: 'framework',
            chunks: 'all',
            test: /[\\/]node_modules[\\/](react|react-dom|scheduler|prop-types)[\\/]/,
            priority: 40,
            enforce: true,
          },
          commons: {
            name: 'commons',
            chunks: 'all',
            minChunks: 2,
            priority: 20,
          },
          lib: {
            test: /[\\/]node_modules[\\/]/,
            name(module) {
              const packageName = module.context.match(
                /[\\/]node_modules[\\/](.*?)([\\/]|$)/
              )[1];
              
              // Group larger libraries into their own chunks
              if (
                ['@heroicons', 'lucide-react', 'd3', 'recharts', 'vis-network'].some(
                  lib => packageName.startsWith(lib)
                )
              ) {
                return `npm.${packageName.replace('@', '')}`;
              }
              
              return 'vendors';
            },
            priority: 10,
            chunks: 'all',
          },
        },
      };
    }
    
    return config;
  },
  
  // Environment variables to expose to the browser
  env: {
    APP_VERSION: process.env.npm_package_version || '1.0.0',
  },
  
  // Configure Sentry for error monitoring
  sentry: {
    hideSourceMaps: false, // Keep source maps for Sentry error reporting
    disableServerWebpackPlugin: process.env.NODE_ENV !== 'production',
    disableClientWebpackPlugin: process.env.NODE_ENV !== 'production',
  },
  
  // Output directory for static exports (if needed)
  distDir: process.env.BUILD_DIR || '.next',
};

// Apply bundle analyzer
const configWithBundleAnalyzer = withBundleAnalyzer(nextConfig);

// Apply Sentry config in production only
const config = process.env.NODE_ENV === 'production'
  ? withSentryConfig(configWithBundleAnalyzer, {
      silent: true,
      org: process.env.SENTRY_ORG || 'illiterate-ai-labs',
      project: process.env.SENTRY_PROJECT || 'analyst-agent-frontend',
    })
  : configWithBundleAnalyzer;

module.exports = config;
