// Analyst's Augmentation Agent - Neo4j Schema Initialization
// This script sets up the initial graph schema for fraud detection and analysis

// ============================================================================
// CONSTRAINTS - Ensure data integrity
// ============================================================================

// Entity constraints
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT organization_id_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT crypto_wallet_id_unique IF NOT EXISTS FOR (w:CryptoWallet) REQUIRE w.address IS UNIQUE;

// Transaction constraints
CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE;

// Document constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Alert constraints
CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:Alert) REQUIRE a.id IS UNIQUE;

// ============================================================================
// INDEXES - Optimize query performance
// ============================================================================

// Entity indexes
CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX person_name_index IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX organization_name_index IF NOT EXISTS FOR (o:Organization) ON (o.name);

// Transaction indexes
CREATE INDEX transaction_amount_index IF NOT EXISTS FOR (t:Transaction) ON (t.amount);
CREATE INDEX transaction_date_index IF NOT EXISTS FOR (t:Transaction) ON (t.date);
CREATE INDEX transaction_type_index IF NOT EXISTS FOR (t:Transaction) ON (t.type);
CREATE INDEX transaction_status_index IF NOT EXISTS FOR (t:Transaction) ON (t.status);

// Document indexes
CREATE INDEX document_type_index IF NOT EXISTS FOR (d:Document) ON (d.type);
CREATE INDEX document_date_index IF NOT EXISTS FOR (d:Document) ON (d.created_date);

// Alert indexes
CREATE INDEX alert_severity_index IF NOT EXISTS FOR (a:Alert) ON (a.severity);
CREATE INDEX alert_status_index IF NOT EXISTS FOR (a:Alert) ON (a.status);
CREATE INDEX alert_date_index IF NOT EXISTS FOR (a:Alert) ON (a.created_date);

// Risk score indexes
CREATE INDEX entity_risk_score_index IF NOT EXISTS FOR (e:Entity) ON (e.risk_score);
CREATE INDEX transaction_risk_score_index IF NOT EXISTS FOR (t:Transaction) ON (t.risk_score);

// ============================================================================
// VECTOR INDEXES - For similarity search and embeddings
// ============================================================================

// Document content embeddings (for Gemini-generated embeddings)
CREATE VECTOR INDEX document_content_embeddings IF NOT EXISTS
FOR (d:Document) ON (d.content_embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// Entity description embeddings
CREATE VECTOR INDEX entity_description_embeddings IF NOT EXISTS
FOR (e:Entity) ON (e.description_embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// ============================================================================
// SAMPLE DATA - Initial dataset for testing and demonstration
// ============================================================================

// Create sample persons
CREATE (p1:Person:Entity {
  id: 'person_001',
  name: 'John Smith',
  email: 'john.smith@email.com',
  phone: '+1-555-0101',
  date_of_birth: date('1985-03-15'),
  nationality: 'US',
  risk_score: 0.2,
  created_date: datetime(),
  updated_date: datetime()
});

CREATE (p2:Person:Entity {
  id: 'person_002',
  name: 'Maria Garcia',
  email: 'maria.garcia@email.com',
  phone: '+1-555-0102',
  date_of_birth: date('1978-07-22'),
  nationality: 'MX',
  risk_score: 0.1,
  created_date: datetime(),
  updated_date: datetime()
});

CREATE (p3:Person:Entity {
  id: 'person_003',
  name: 'Alex Chen',
  email: 'alex.chen@email.com',
  phone: '+1-555-0103',
  date_of_birth: date('1990-11-08'),
  nationality: 'CN',
  risk_score: 0.7,
  created_date: datetime(),
  updated_date: datetime()
});

// Create sample organizations
CREATE (o1:Organization:Entity {
  id: 'org_001',
  name: 'TechCorp Industries',
  registration_number: 'TC123456789',
  country: 'US',
  industry: 'Technology',
  risk_score: 0.3,
  created_date: datetime(),
  updated_date: datetime()
});

CREATE (o2:Organization:Entity {
  id: 'org_002',
  name: 'Global Finance Ltd',
  registration_number: 'GF987654321',
  country: 'UK',
  industry: 'Financial Services',
  risk_score: 0.8,
  created_date: datetime(),
  updated_date: datetime()
});

// Create sample crypto wallets
CREATE (w1:CryptoWallet:Entity {
  id: 'wallet_001',
  address: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
  currency: 'BTC',
  risk_score: 0.5,
  created_date: datetime(),
  updated_date: datetime()
});

CREATE (w2:CryptoWallet:Entity {
  id: 'wallet_002',
  address: '0x742d35Cc6634C0532925a3b8D4C2C4e4C4C4C4C4',
  currency: 'ETH',
  risk_score: 0.9,
  created_date: datetime(),
  updated_date: datetime()
});

// Create sample transactions
CREATE (t1:Transaction {
  id: 'txn_001',
  amount: 50000.00,
  currency: 'USD',
  type: 'wire_transfer',
  date: datetime('2024-01-15T10:30:00Z'),
  status: 'completed',
  risk_score: 0.6,
  description: 'International wire transfer',
  created_date: datetime(),
  updated_date: datetime()
});

CREATE (t2:Transaction {
  id: 'txn_002',
  amount: 25000.00,
  currency: 'USD',
  type: 'cash_deposit',
  date: datetime('2024-01-16T14:20:00Z'),
  status: 'completed',
  risk_score: 0.8,
  description: 'Large cash deposit',
  created_date: datetime(),
  updated_date: datetime()
});

CREATE (t3:Transaction {
  id: 'txn_003',
  amount: 75000.00,
  currency: 'USD',
  type: 'crypto_exchange',
  date: datetime('2024-01-17T09:15:00Z'),
  status: 'completed',
  risk_score: 0.9,
  description: 'Cryptocurrency exchange transaction',
  created_date: datetime(),
  updated_date: datetime()
});

// Create sample documents
CREATE (d1:Document {
  id: 'doc_001',
  type: 'identity_document',
  title: 'Passport - John Smith',
  content: 'US Passport for John Smith, issued 2020',
  created_date: datetime('2024-01-10T08:00:00Z'),
  updated_date: datetime()
});

CREATE (d2:Document {
  id: 'doc_002',
  type: 'financial_statement',
  title: 'Bank Statement - Global Finance Ltd',
  content: 'Monthly bank statement showing large transactions',
  created_date: datetime('2024-01-12T12:00:00Z'),
  updated_date: datetime()
});

// Create sample alerts
CREATE (a1:Alert {
  id: 'alert_001',
  type: 'suspicious_transaction',
  severity: 'high',
  status: 'open',
  title: 'Large cash deposit pattern detected',
  description: 'Multiple large cash deposits in short timeframe',
  created_date: datetime('2024-01-16T15:00:00Z'),
  updated_date: datetime()
});

// ============================================================================
// RELATIONSHIPS - Connect the entities
// ============================================================================

// Person-Organization relationships
MATCH (p1:Person {id: 'person_001'}), (o1:Organization {id: 'org_001'})
CREATE (p1)-[:WORKS_FOR {position: 'CEO', start_date: date('2020-01-01')}]->(o1);

MATCH (p3:Person {id: 'person_003'}), (o2:Organization {id: 'org_002'})
CREATE (p3)-[:OWNS {ownership_percentage: 75.0, start_date: date('2019-06-01')}]->(o2);

// Person-Wallet relationships
MATCH (p1:Person {id: 'person_001'}), (w1:CryptoWallet {id: 'wallet_001'})
CREATE (p1)-[:OWNS_WALLET {verified: true, created_date: datetime('2023-01-01T00:00:00Z')}]->(w1);

MATCH (p3:Person {id: 'person_003'}), (w2:CryptoWallet {id: 'wallet_002'})
CREATE (p3)-[:OWNS_WALLET {verified: false, created_date: datetime('2023-06-01T00:00:00Z')}]->(w2);

// Transaction relationships
MATCH (p1:Person {id: 'person_001'}), (t1:Transaction {id: 'txn_001'}), (p2:Person {id: 'person_002'})
CREATE (p1)-[:SENT_TRANSACTION]->(t1)-[:RECEIVED_BY]->(p2);

MATCH (p2:Person {id: 'person_002'}), (t2:Transaction {id: 'txn_002'}), (o2:Organization {id: 'org_002'})
CREATE (p2)-[:SENT_TRANSACTION]->(t2)-[:RECEIVED_BY]->(o2);

MATCH (w1:CryptoWallet {id: 'wallet_001'}), (t3:Transaction {id: 'txn_003'}), (w2:CryptoWallet {id: 'wallet_002'})
CREATE (w1)-[:SENT_TRANSACTION]->(t3)-[:RECEIVED_BY]->(w2);

// Document relationships
MATCH (p1:Person {id: 'person_001'}), (d1:Document {id: 'doc_001'})
CREATE (p1)-[:HAS_DOCUMENT {document_type: 'identity', verified: true}]->(d1);

MATCH (o2:Organization {id: 'org_002'}), (d2:Document {id: 'doc_002'})
CREATE (o2)-[:HAS_DOCUMENT {document_type: 'financial', verified: true}]->(d2);

// Alert relationships
MATCH (t2:Transaction {id: 'txn_002'}), (a1:Alert {id: 'alert_001'})
CREATE (t2)-[:TRIGGERED_ALERT {confidence: 0.85}]->(a1);

MATCH (p2:Person {id: 'person_002'}), (a1:Alert {id: 'alert_001'})
CREATE (p2)-[:SUBJECT_OF_ALERT]->(a1);

// ============================================================================
// VERIFICATION
// ============================================================================

// Return summary of created data
MATCH (n) 
RETURN labels(n)[0] as NodeType, count(n) as Count
ORDER BY NodeType;
