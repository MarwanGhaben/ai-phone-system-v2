-- AI Voice Platform v2 - Database Initialization
-- Run automatically when PostgreSQL container starts

\echo 'Creating database schema...'

-- Sequences (PostgreSQL requires this for auto-increment)
CREATE SEQUENCE IF NOT EXISTS calls_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS conversations_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS conversation_turns_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS appointments_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS users_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS tenants_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS knowledge_articles_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS analytics_events_id_seq START 1;

-- =====================================================
-- CALLS TABLE - Phone call records
-- =====================================================
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY DEFAULT nextval('calls_id_seq'),
    call_sid VARCHAR(100) UNIQUE NOT NULL,
    tenant_id INTEGER DEFAULT 1,
    phone_number VARCHAR(50),
    caller_name VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    detected_language VARCHAR(10),
    direction VARCHAR(20) DEFAULT 'inbound',  -- inbound | outbound
    status VARCHAR(50) DEFAULT 'in_progress',  -- in_progress | completed | failed | transferred
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    conversation_turns INTEGER DEFAULT 0,
    transferred BOOLEAN DEFAULT FALSE,
    transferred_to VARCHAR(100),
    recording_url TEXT,
    transcription TEXT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calls_phone ON calls(phone_number);
CREATE INDEX idx_calls_status ON calls(status);
CREATE INDEX idx_calls_started ON calls(started_at);
CREATE INDEX idx_calls_tenant ON calls(tenant_id);

-- =====================================================
-- CONVERSATIONS TABLE - Conversation history
-- =====================================================
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY DEFAULT nextval('conversations_id_seq'),
    call_sid VARCHAR(100) REFERENCES calls(call_sid) ON DELETE CASCADE,
    tenant_id INTEGER DEFAULT 1,
    state JSONB,  -- Current conversation state
    intent_history JSONB,  -- Track intents over time
    summary TEXT,  -- AI-generated summary
    sentiment VARCHAR(50),  -- positive | neutral | negative
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_conversations_call_sid ON conversations(call_sid);

-- =====================================================
-- CONVERSATION TURNS TABLE - Individual exchanges
-- =====================================================
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY DEFAULT nextval('conversation_turns_id_seq'),
    call_sid VARCHAR(100) REFERENCES calls(call_sid) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- user | assistant | system
    content TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    audio_duration_ms INTEGER,
    stt_confidence FLOAT,
    intent VARCHAR(100),
    entities JSONB,  -- Extracted entities (dates, names, etc.)
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_turns_call_sid ON conversation_turns(call_sid);
CREATE INDEX idx_turns_timestamp ON conversation_turns(timestamp);

-- =====================================================
-- APPOINTMENTS TABLE - Booked appointments
-- =====================================================
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY DEFAULT nextval('appointments_id_seq'),
    call_sid VARCHAR(100) REFERENCES calls(call_sid) ON DELETE SET NULL,
    tenant_id INTEGER DEFAULT 1,
    client_name VARCHAR(100) NOT NULL,
    client_phone VARCHAR(50) NOT NULL,
    client_email VARCHAR(100),
    accountant_name VARCHAR(100),
    appointment_time TIMESTAMP WITH TIME ZONE NOT NULL,
    appointment_time_formatted VARCHAR(200),
    client_type VARCHAR(50),  -- individual | corporate
    language VARCHAR(10) DEFAULT 'en',
    status VARCHAR(50) DEFAULT 'confirmed',  -- pending | confirmed | cancelled | completed
    booking_url TEXT,
    event_type_uri VARCHAR(255),
    service_id VARCHAR(100),
    staff_id VARCHAR(100),
    ms_booking_id VARCHAR(100),
    reminder_sent BOOLEAN DEFAULT FALSE,
    reminder_sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_appointments_phone ON appointments(client_phone);
CREATE INDEX idx_appointments_time ON appointments(appointment_time);
CREATE INDEX idx_appointments_status ON appointments(status);

-- =====================================================
-- USERS TABLE - Admin users
-- =====================================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY DEFAULT nextval('users_id_seq'),
    tenant_id INTEGER DEFAULT 1,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    role VARCHAR(50) DEFAULT 'admin',  -- admin | viewer
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(100),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- TENANTS TABLE - Multi-tenant support
-- =====================================================
CREATE TABLE IF NOT EXISTS tenants (
    id INTEGER PRIMARY KEY DEFAULT nextval('tenants_id_seq'),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(50) UNIQUE NOT NULL,
    industry VARCHAR(100),
    phone_number VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    settings JSONB,  -- Tenant-specific settings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- KNOWLEDGE ARTICLES TABLE - FAQ/Knowledge base
-- =====================================================
CREATE TABLE IF NOT EXISTS knowledge_articles (
    id INTEGER PRIMARY KEY DEFAULT nextval('knowledge_articles_id_seq'),
    tenant_id INTEGER DEFAULT 1,
    category VARCHAR(100),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    priority INTEGER DEFAULT 0,
    tags TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_knowledge_tenant ON knowledge_articles(tenant_id);
CREATE INDEX idx_knowledge_category ON knowledge_articles(category);
CREATE INDEX idx_knowledge_language ON knowledge_articles(language);

-- =====================================================
-- CALLERS TABLE - Caller recognition / profiles
-- =====================================================
CREATE SEQUENCE IF NOT EXISTS callers_id_seq START 1;

CREATE TABLE IF NOT EXISTS callers (
    id INTEGER PRIMARY KEY DEFAULT nextval('callers_id_seq'),
    phone_number VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    call_count INTEGER DEFAULT 1,
    first_call TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_call TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_callers_phone ON callers(phone_number);
CREATE INDEX idx_callers_tenant ON callers(tenant_id);
CREATE INDEX idx_callers_last_call ON callers(last_call);

-- =====================================================
-- ANALYTICS EVENTS TABLE - Usage metrics
-- =====================================================
CREATE TABLE IF NOT EXISTS analytics_events (
    id INTEGER PRIMARY KEY DEFAULT nextval('analytics_events_id_seq'),
    tenant_id INTEGER DEFAULT 1,
    call_sid VARCHAR(100),
    event_type VARCHAR(100) NOT NULL,  -- call_started | stt_begin | tts_begin | intent_detected | etc.
    event_data JSONB,
    duration_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_analytics_tenant ON analytics_events(tenant_id);
CREATE INDEX idx_analytics_type ON analytics_events(event_type);
CREATE INDEX idx_analytics_timestamp ON analytics_events(timestamp);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update timestamp trigger to relevant tables
CREATE TRIGGER update_calls_updated_at BEFORE UPDATE ON calls
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_appointments_updated_at BEFORE UPDATE ON appointments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_updated_at BEFORE UPDATE ON knowledge_articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_callers_updated_at BEFORE UPDATE ON callers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- DEFAULT DATA
-- =====================================================

-- Insert default tenant
INSERT INTO tenants (name, slug, industry) VALUES
('iFlex Tax', 'iflextax', 'accounting')
ON CONFLICT (slug) DO NOTHING;

-- Insert default admin user (password: admin123 - CHANGE THIS!)
INSERT INTO users (username, password_hash, email, role) VALUES
('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYkWqC5fXL6', 'admin@example.com', 'admin')
ON CONFLICT (username) DO NOTHING;

\echo 'Database schema created successfully!'
