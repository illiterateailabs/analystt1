/**
 * Mock Data Generator for Graph Visualization Testing
 * 
 * This file provides mock data generators for testing the graph visualization component
 * without requiring a connection to the backend. It generates realistic fraud investigation
 * results with nodes, edges, fraud patterns, compliance findings, and recommendations.
 */

import { v4 as uuidv4 } from 'uuid';

// Node types with their properties
const NODE_TYPES = {
  Person: ['name', 'age', 'nationality', 'occupation', 'risk_profile'],
  Account: ['account_number', 'bank_name', 'balance', 'opening_date', 'account_type', 'currency'],
  Transaction: ['amount', 'timestamp', 'description', 'reference', 'currency', 'status'],
  Organization: ['name', 'industry', 'registration_number', 'country', 'founding_date'],
  Asset: ['name', 'value', 'acquisition_date', 'type', 'location'],
  Event: ['name', 'date', 'location', 'description', 'participants'],
  Location: ['address', 'city', 'country', 'postal_code', 'coordinates']
};

// Edge types
const EDGE_TYPES = [
  'OWNS', 'TRANSFERS_TO', 'WORKS_FOR', 'CONTROLS', 'PARTICIPATED_IN',
  'RELATED_TO', 'LOCATED_AT', 'MANAGES', 'TRANSACTS_WITH', 'BELONGS_TO'
];

// Fraud pattern templates
const FRAUD_PATTERNS = [
  {
    name: 'Structuring',
    description: 'Multiple small transactions designed to avoid reporting thresholds',
    confidence_range: [0.65, 0.95]
  },
  {
    name: 'Layering',
    description: 'Complex series of transactions to obscure the source of funds',
    confidence_range: [0.70, 0.90]
  },
  {
    name: 'Smurfing',
    description: 'Breaking down large transactions into multiple smaller ones using different accounts',
    confidence_range: [0.60, 0.85]
  },
  {
    name: 'Round Tripping',
    description: 'Funds transferred through multiple accounts and returned to the originator',
    confidence_range: [0.75, 0.95]
  },
  {
    name: 'Shell Company Activity',
    description: 'Transactions through entities with no apparent business purpose',
    confidence_range: [0.80, 0.98]
  },
  {
    name: 'Unusual Transaction Patterns',
    description: 'Transactions that deviate significantly from expected behavior',
    confidence_range: [0.55, 0.80]
  },
  {
    name: 'High-Risk Jurisdiction',
    description: 'Transactions involving entities from high-risk or sanctioned countries',
    confidence_range: [0.70, 0.90]
  }
];

// Compliance regulations
const REGULATIONS = [
  {
    name: 'AML Directive 5',
    description: 'EU Anti-Money Laundering Directive requirements',
    recommendations: [
      'Enhance customer due diligence for high-risk transactions',
      'Implement real-time transaction monitoring',
      'Update risk assessment methodology'
    ]
  },
  {
    name: 'BSA/FINCEN',
    description: 'Bank Secrecy Act compliance requirements',
    recommendations: [
      'File Suspicious Activity Report (SAR)',
      'Implement enhanced transaction monitoring',
      'Conduct additional KYC verification'
    ]
  },
  {
    name: 'FATF Recommendations',
    description: 'Financial Action Task Force guidelines',
    recommendations: [
      'Apply risk-based approach to monitoring',
      'Strengthen beneficial ownership identification',
      'Enhance record-keeping procedures'
    ]
  },
  {
    name: 'OFAC Sanctions',
    description: 'Office of Foreign Assets Control compliance',
    recommendations: [
      'Block suspicious transactions',
      'Report to regulatory authorities',
      'Implement enhanced screening procedures'
    ]
  }
];

/**
 * Generate a random integer between min and max (inclusive)
 */
function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Generate a random float between min and max
 */
function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * Generate a random date within the last year
 */
function randomDate(): string {
  const now = new Date();
  const pastDate = new Date(now.getTime() - randomInt(0, 365) * 24 * 60 * 60 * 1000);
  return pastDate.toISOString();
}

/**
 * Generate random node properties based on type
 */
function generateNodeProperties(type: string): Record<string, any> {
  const properties: Record<string, any> = {};
  const possibleProps = NODE_TYPES[type as keyof typeof NODE_TYPES] || [];
  
  // Add type-specific properties
  switch (type) {
    case 'Person':
      properties.name = `${['John', 'Jane', 'Alex', 'Maria', 'Sam', 'Emma', 'Michael'][randomInt(0, 6)]} ${['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller'][randomInt(0, 6)]}`;
      properties.age = randomInt(18, 75);
      properties.nationality = ['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU'][randomInt(0, 6)];
      properties.occupation = ['Businessman', 'Consultant', 'Investor', 'Manager', 'Director', 'Trader', 'Broker'][randomInt(0, 6)];
      properties.risk_profile = ['Low', 'Medium', 'High'][randomInt(0, 2)];
      break;
      
    case 'Account':
      properties.account_number = `AC${randomInt(100000, 999999)}`;
      properties.bank_name = ['GlobalBank', 'CitiTrust', 'NationalFinance', 'OceanicBank', 'MetroCredit'][randomInt(0, 4)];
      properties.balance = Math.round(randomFloat(1000, 1000000) * 100) / 100;
      properties.opening_date = randomDate();
      properties.account_type = ['Checking', 'Savings', 'Investment', 'Business', 'Offshore'][randomInt(0, 4)];
      properties.currency = ['USD', 'EUR', 'GBP', 'JPY', 'CHF'][randomInt(0, 4)];
      break;
      
    case 'Transaction':
      properties.amount = Math.round(randomFloat(100, 50000) * 100) / 100;
      properties.timestamp = randomDate();
      properties.description = ['Payment', 'Transfer', 'Withdrawal', 'Deposit', 'Investment', 'Fee', 'Interest'][randomInt(0, 6)];
      properties.reference = `TX${randomInt(1000000, 9999999)}`;
      properties.currency = ['USD', 'EUR', 'GBP', 'JPY', 'CHF'][randomInt(0, 4)];
      properties.status = ['Completed', 'Pending', 'Failed', 'Flagged'][randomInt(0, 3)];
      break;
      
    case 'Organization':
      properties.name = `${['Global', 'United', 'Premier', 'Elite', 'Advanced', 'Strategic', 'Superior'][randomInt(0, 6)]} ${['Holdings', 'Enterprises', 'Solutions', 'Partners', 'Investments', 'Capital', 'Group'][randomInt(0, 6)]}`;
      properties.industry = ['Finance', 'Technology', 'Real Estate', 'Energy', 'Manufacturing', 'Trading', 'Consulting'][randomInt(0, 6)];
      properties.registration_number = `ORG${randomInt(10000, 99999)}`;
      properties.country = ['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU'][randomInt(0, 6)];
      properties.founding_date = randomDate();
      break;
      
    case 'Asset':
      properties.name = `${['Commercial', 'Residential', 'Industrial', 'Luxury', 'Investment'][randomInt(0, 4)]} ${['Property', 'Vehicle', 'Artwork', 'Yacht', 'Aircraft', 'Portfolio', 'Commodity'][randomInt(0, 6)]}`;
      properties.value = Math.round(randomFloat(10000, 5000000) * 100) / 100;
      properties.acquisition_date = randomDate();
      properties.type = ['Real Estate', 'Vehicle', 'Art', 'Investment', 'Commodity', 'Luxury Item'][randomInt(0, 5)];
      properties.location = ['New York', 'London', 'Dubai', 'Singapore', 'Hong Kong', 'Zurich', 'Tokyo'][randomInt(0, 6)];
      break;
      
    default:
      // Add some generic properties for other types
      possibleProps.forEach(prop => {
        properties[prop] = `${prop}-${randomInt(1, 100)}`;
      });
  }
  
  return properties;
}

/**
 * Generate a mock node
 */
function generateNode(id: string, type: string): any {
  const properties = generateNodeProperties(type);
  
  return {
    id,
    label: properties.name || properties.account_number || properties.reference || `${type}-${id.substring(0, 5)}`,
    type,
    properties,
    risk_score: randomFloat(0, 1)
  };
}

/**
 * Generate a mock edge between nodes
 */
function generateEdge(id: string, fromNode: any, toNode: any): any {
  // Choose appropriate edge type based on node types
  let edgeType = EDGE_TYPES[randomInt(0, EDGE_TYPES.length - 1)];
  
  // Make edge types more realistic based on connected node types
  if (fromNode.type === 'Person' && toNode.type === 'Account') {
    edgeType = 'OWNS';
  } else if (fromNode.type === 'Account' && toNode.type === 'Transaction') {
    edgeType = 'INITIATES';
  } else if (fromNode.type === 'Transaction' && toNode.type === 'Account') {
    edgeType = 'TRANSFERS_TO';
  } else if (fromNode.type === 'Person' && toNode.type === 'Organization') {
    edgeType = 'WORKS_FOR';
  } else if (fromNode.type === 'Organization' && toNode.type === 'Account') {
    edgeType = 'CONTROLS';
  }
  
  return {
    id,
    from: fromNode.id,
    to: toNode.id,
    label: edgeType,
    properties: {
      timestamp: randomDate(),
      weight: randomFloat(0.1, 1).toFixed(2)
    }
  };
}

/**
 * Generate a fraud pattern
 */
function generateFraudPattern(id: string, affectedNodes: any[]): any {
  const patternTemplate = FRAUD_PATTERNS[randomInt(0, FRAUD_PATTERNS.length - 1)];
  const confidence = randomFloat(
    patternTemplate.confidence_range[0], 
    patternTemplate.confidence_range[1]
  );
  
  return {
    id,
    name: patternTemplate.name,
    description: patternTemplate.description,
    confidence,
    affected_entities: affectedNodes.map(node => node.id)
  };
}

/**
 * Generate a compliance finding
 */
function generateComplianceFinding(fraudPatterns: any[]): any {
  const regulation = REGULATIONS[randomInt(0, REGULATIONS.length - 1)];
  const statuses = ['compliant', 'non-compliant', 'warning'];
  const status = statuses[randomInt(0, 2)];
  
  let description = regulation.description;
  if (status === 'non-compliant') {
    description = `Violation of ${regulation.name}: ${fraudPatterns.map(p => p.name).join(', ')} detected`;
  } else if (status === 'warning') {
    description = `Potential concern with ${regulation.name}: Unusual activity detected`;
  }
  
  return {
    regulation: regulation.name,
    status,
    description,
    recommendation: status !== 'compliant' ? regulation.recommendations[randomInt(0, regulation.recommendations.length - 1)] : undefined
  };
}

/**
 * Generate a complete mock fraud investigation result
 */
export function generateMockFraudInvestigation(nodeCount = 20, complexity = 'medium'): any {
  // Adjust edge count based on complexity
  let edgeMultiplier = 1.5; // medium
  if (complexity === 'low') {
    edgeMultiplier = 1.2;
  } else if (complexity === 'high') {
    edgeMultiplier = 2.0;
  }
  
  const edgeCount = Math.floor(nodeCount * edgeMultiplier);
  const patternCount = Math.max(1, Math.floor(nodeCount / 5));
  
  // Generate nodes
  const nodes = [];
  const nodeTypes = Object.keys(NODE_TYPES);
  
  for (let i = 0; i < nodeCount; i++) {
    const nodeId = `n${i}_${uuidv4().substring(0, 8)}`;
    const nodeType = nodeTypes[randomInt(0, nodeTypes.length - 1)];
    nodes.push(generateNode(nodeId, nodeType));
  }
  
  // Generate edges
  const edges = [];
  for (let i = 0; i < edgeCount; i++) {
    const edgeId = `e${i}_${uuidv4().substring(0, 8)}`;
    const fromNode = nodes[randomInt(0, nodes.length - 1)];
    const toNode = nodes[randomInt(0, nodes.length - 1)];
    
    // Avoid self-loops
    if (fromNode.id !== toNode.id) {
      edges.push(generateEdge(edgeId, fromNode, toNode));
    }
  }
  
  // Generate fraud patterns
  const fraudPatterns = [];
  for (let i = 0; i < patternCount; i++) {
    const patternId = `p${i}_${uuidv4().substring(0, 8)}`;
    // Each pattern affects 2-5 nodes
    const affectedNodeCount = randomInt(2, 5);
    const affectedNodes = [];
    
    for (let j = 0; j < affectedNodeCount; j++) {
      const randomNode = nodes[randomInt(0, nodes.length - 1)];
      if (!affectedNodes.includes(randomNode)) {
        affectedNodes.push(randomNode);
      }
    }
    
    fraudPatterns.push(generateFraudPattern(patternId, affectedNodes));
  }
  
  // Generate compliance findings
  const complianceFindings = [];
  const findingCount = randomInt(1, 4);
  
  for (let i = 0; i < findingCount; i++) {
    complianceFindings.push(generateComplianceFinding(fraudPatterns));
  }
  
  // Generate risk assessment
  const overallRiskScore = randomFloat(0.3, 0.9);
  const riskFactors = {
    transaction_volume: randomFloat(0.2, 0.8),
    geographic_risk: randomFloat(0.2, 0.8),
    customer_profile: randomFloat(0.2, 0.8),
    transaction_patterns: randomFloat(0.2, 0.8)
  };
  
  // Generate executive summary
  const patternNames = fraudPatterns.map(p => p.name).join(', ');
  const executiveSummary = `Analysis identified ${fraudPatterns.length} potential fraud pattern${fraudPatterns.length !== 1 ? 's' : ''}: ${patternNames}. Overall risk assessment is ${overallRiskScore > 0.7 ? 'HIGH' : overallRiskScore > 0.4 ? 'MEDIUM' : 'LOW'} (${(overallRiskScore * 100).toFixed(1)}%). ${complianceFindings.filter(f => f.status === 'non-compliant').length} compliance violations detected requiring immediate attention.`;
  
  // Generate recommendations
  const recommendations = [
    `Enhance monitoring of ${nodes.filter(n => n.risk_score > 0.7).length} high-risk entities identified in this analysis`,
    `Conduct enhanced due diligence on transactions involving ${fraudPatterns.map(p => p.name.toLowerCase()).join(' and ')}`,
    `Update risk assessment methodology to address ${complianceFindings.filter(f => f.status !== 'compliant').length} compliance concerns`,
    `Implement additional controls for transactions with similar patterns`
  ];
  
  // Return the complete mock data
  return {
    graph_data: {
      nodes,
      edges
    },
    fraud_patterns: fraudPatterns,
    risk_assessment: {
      overall_score: overallRiskScore,
      factors: riskFactors,
      summary: `Overall risk level: ${overallRiskScore > 0.7 ? 'HIGH' : overallRiskScore > 0.4 ? 'MEDIUM' : 'LOW'}`
    },
    compliance_findings: complianceFindings,
    executive_summary: executiveSummary,
    recommendations
  };
}

/**
 * Generate a simple mock fraud investigation with pre-defined structure
 * Useful for demos and screenshots
 */
export function generateSimpleMockFraudInvestigation(): any {
  // Create a small network with a clear fraud pattern
  const nodes = [
    {
      id: 'person1',
      label: 'John Smith',
      type: 'Person',
      properties: {
        name: 'John Smith',
        age: 42,
        nationality: 'US',
        occupation: 'Businessman',
        risk_profile: 'High'
      },
      risk_score: 0.85
    },
    {
      id: 'org1',
      label: 'Global Holdings Ltd',
      type: 'Organization',
      properties: {
        name: 'Global Holdings Ltd',
        industry: 'Finance',
        registration_number: 'ORG12345',
        country: 'Cayman Islands',
        founding_date: '2023-01-15T00:00:00.000Z'
      },
      risk_score: 0.78
    },
    {
      id: 'account1',
      label: 'AC123456',
      type: 'Account',
      properties: {
        account_number: 'AC123456',
        bank_name: 'OceanicBank',
        balance: 1250000,
        opening_date: '2023-02-10T00:00:00.000Z',
        account_type: 'Business',
        currency: 'USD'
      },
      risk_score: 0.72
    },
    {
      id: 'account2',
      label: 'AC789012',
      type: 'Account',
      properties: {
        account_number: 'AC789012',
        bank_name: 'MetroCredit',
        balance: 850000,
        opening_date: '2023-02-15T00:00:00.000Z',
        account_type: 'Business',
        currency: 'EUR'
      },
      risk_score: 0.65
    },
    {
      id: 'account3',
      label: 'AC345678',
      type: 'Account',
      properties: {
        account_number: 'AC345678',
        bank_name: 'CitiTrust',
        balance: 950000,
        opening_date: '2023-03-01T00:00:00.000Z',
        account_type: 'Offshore',
        currency: 'USD'
      },
      risk_score: 0.82
    },
    {
      id: 'tx1',
      label: 'TX1000001',
      type: 'Transaction',
      properties: {
        amount: 450000,
        timestamp: '2023-04-05T10:30:00.000Z',
        description: 'Transfer',
        reference: 'TX1000001',
        currency: 'USD',
        status: 'Completed'
      },
      risk_score: 0.75
    },
    {
      id: 'tx2',
      label: 'TX1000002',
      type: 'Transaction',
      properties: {
        amount: 320000,
        timestamp: '2023-04-10T14:15:00.000Z',
        description: 'Transfer',
        reference: 'TX1000002',
        currency: 'EUR',
        status: 'Completed'
      },
      risk_score: 0.68
    },
    {
      id: 'tx3',
      label: 'TX1000003',
      type: 'Transaction',
      properties: {
        amount: 380000,
        timestamp: '2023-04-15T09:45:00.000Z',
        description: 'Transfer',
        reference: 'TX1000003',
        currency: 'USD',
        status: 'Completed'
      },
      risk_score: 0.71
    },
    {
      id: 'org2',
      label: 'Shell Corp Inc',
      type: 'Organization',
      properties: {
        name: 'Shell Corp Inc',
        industry: 'Consulting',
        registration_number: 'ORG67890',
        country: 'Belize',
        founding_date: '2023-01-20T00:00:00.000Z'
      },
      risk_score: 0.88
    },
    {
      id: 'person2',
      label: 'Jane Doe',
      type: 'Person',
      properties: {
        name: 'Jane Doe',
        age: 38,
        nationality: 'UK',
        occupation: 'Director',
        risk_profile: 'High'
      },
      risk_score: 0.76
    }
  ];
  
  const edges = [
    {
      id: 'e1',
      from: 'person1',
      to: 'org1',
      label: 'CONTROLS',
      properties: {
        timestamp: '2023-02-01T00:00:00.000Z',
        weight: '0.9'
      }
    },
    {
      id: 'e2',
      from: 'org1',
      to: 'account1',
      label: 'OWNS',
      properties: {
        timestamp: '2023-02-10T00:00:00.000Z',
        weight: '0.9'
      }
    },
    {
      id: 'e3',
      from: 'account1',
      to: 'tx1',
      label: 'INITIATES',
      properties: {
        timestamp: '2023-04-05T10:30:00.000Z',
        weight: '0.8'
      }
    },
    {
      id: 'e4',
      from: 'tx1',
      to: 'account2',
      label: 'TRANSFERS_TO',
      properties: {
        timestamp: '2023-04-05T10:30:00.000Z',
        weight: '0.8'
      }
    },
    {
      id: 'e5',
      from: 'account2',
      to: 'tx2',
      label: 'INITIATES',
      properties: {
        timestamp: '2023-04-10T14:15:00.000Z',
        weight: '0.7'
      }
    },
    {
      id: 'e6',
      from: 'tx2',
      to: 'account3',
      label: 'TRANSFERS_TO',
      properties: {
        timestamp: '2023-04-10T14:15:00.000Z',
        weight: '0.7'
      }
    },
    {
      id: 'e7',
      from: 'account3',
      to: 'tx3',
      label: 'INITIATES',
      properties: {
        timestamp: '2023-04-15T09:45:00.000Z',
        weight: '0.8'
      }
    },
    {
      id: 'e8',
      from: 'tx3',
      to: 'account1',
      label: 'TRANSFERS_TO',
      properties: {
        timestamp: '2023-04-15T09:45:00.000Z',
        weight: '0.8'
      }
    },
    {
      id: 'e9',
      from: 'org2',
      to: 'account2',
      label: 'OWNS',
      properties: {
        timestamp: '2023-02-15T00:00:00.000Z',
        weight: '0.9'
      }
    },
    {
      id: 'e10',
      from: 'person2',
      to: 'org2',
      label: 'CONTROLS',
      properties: {
        timestamp: '2023-01-25T00:00:00.000Z',
        weight: '0.85'
      }
    },
    {
      id: 'e11',
      from: 'person1',
      to: 'person2',
      label: 'RELATED_TO',
      properties: {
        timestamp: '2022-12-01T00:00:00.000Z',
        weight: '0.95'
      }
    },
    {
      id: 'e12',
      from: 'org2',
      to: 'account3',
      label: 'CONTROLS',
      properties: {
        timestamp: '2023-03-01T00:00:00.000Z',
        weight: '0.8'
      }
    }
  ];
  
  const fraudPatterns = [
    {
      id: 'pattern1',
      name: 'Round Tripping',
      description: 'Funds transferred through multiple accounts and returned to the originator',
      confidence: 0.92,
      affected_entities: ['account1', 'tx1', 'account2', 'tx2', 'account3', 'tx3']
    },
    {
      id: 'pattern2',
      name: 'Shell Company Activity',
      description: 'Transactions through entities with no apparent business purpose',
      confidence: 0.88,
      affected_entities: ['org1', 'org2', 'account2', 'account3']
    }
  ];
  
  const complianceFindings = [
    {
      regulation: 'AML Directive 5',
      status: 'non-compliant',
      description: 'Violation of AML Directive 5: Round Tripping, Shell Company Activity detected',
      recommendation: 'File Suspicious Activity Report (SAR) within 24 hours'
    },
    {
      regulation: 'FATF Recommendations',
      status: 'warning',
      description: 'Potential concern with FATF Recommendations: Unusual activity detected in high-risk jurisdictions',
      recommendation: 'Enhance beneficial ownership identification'
    },
    {
      regulation: 'BSA/FINCEN',
      status: 'non-compliant',
      description: 'Violation of BSA/FINCEN: Suspicious circular transaction patterns identified',
      recommendation: 'Implement enhanced transaction monitoring'
    }
  ];
  
  return {
    graph_data: {
      nodes,
      edges
    },
    fraud_patterns: fraudPatterns,
    risk_assessment: {
      overall_score: 0.85,
      factors: {
        transaction_volume: 0.72,
        geographic_risk: 0.88,
        customer_profile: 0.78,
        transaction_patterns: 0.91
      },
      summary: 'Overall risk level: HIGH'
    },
    compliance_findings: complianceFindings,
    executive_summary: 'Analysis identified 2 potential fraud patterns: Round Tripping, Shell Company Activity. Overall risk assessment is HIGH (85.0%). 2 compliance violations detected requiring immediate attention. The transaction pattern shows clear evidence of round-tripping through offshore entities.',
    recommendations: [
      'File Suspicious Activity Reports (SARs) for all identified entities within 24 hours',
      'Freeze accounts pending further investigation',
      'Conduct enhanced due diligence on all related parties',
      'Update monitoring systems to detect similar round-tripping patterns'
    ]
  };
}
