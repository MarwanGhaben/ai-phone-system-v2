-- T005-B empty-database compatibility schema.
-- The caller owns the transaction. This file creates no application data or
-- migration ledger and is not yet integrated with startup or the runner.

CREATE SEQUENCE public.calls_id_seq START WITH 1;
CREATE SEQUENCE public.conversations_id_seq START WITH 1;
CREATE SEQUENCE public.conversation_turns_id_seq START WITH 1;
CREATE SEQUENCE public.appointments_id_seq START WITH 1;
CREATE SEQUENCE public.users_id_seq START WITH 1;
CREATE SEQUENCE public.tenants_id_seq START WITH 1;
CREATE SEQUENCE public.knowledge_articles_id_seq START WITH 1;
CREATE SEQUENCE public.analytics_events_id_seq START WITH 1;
CREATE SEQUENCE public.callers_id_seq START WITH 1;

CREATE TABLE public.calls (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval('public.calls_id_seq'::pg_catalog.regclass),
    call_sid VARCHAR(100) UNIQUE NOT NULL,
    tenant_id INTEGER DEFAULT 1,
    phone_number VARCHAR(50),
    caller_name VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    detected_language VARCHAR(10),
    direction VARCHAR(20) DEFAULT 'inbound',
    status VARCHAR(50) DEFAULT 'in_progress',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    conversation_turns INTEGER DEFAULT 0,
    transferred BOOLEAN DEFAULT FALSE,
    transferred_to VARCHAR(100),
    recording_url TEXT,
    transcription TEXT,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.conversations (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval('public.conversations_id_seq'::pg_catalog.regclass),
    call_sid VARCHAR(100)
        REFERENCES public.calls(call_sid) ON DELETE CASCADE,
    tenant_id INTEGER DEFAULT 1,
    state JSONB,
    intent_history JSONB,
    summary TEXT,
    sentiment VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.conversation_turns (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval(
            'public.conversation_turns_id_seq'::pg_catalog.regclass
        ),
    call_sid VARCHAR(100)
        REFERENCES public.calls(call_sid) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    audio_duration_ms INTEGER,
    stt_confidence FLOAT,
    intent VARCHAR(100),
    entities JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.appointments (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval(
            'public.appointments_id_seq'::pg_catalog.regclass
        ),
    call_sid VARCHAR(100)
        REFERENCES public.calls(call_sid) ON DELETE SET NULL,
    tenant_id INTEGER DEFAULT 1,
    client_name VARCHAR(100) NOT NULL,
    client_phone VARCHAR(50) NOT NULL,
    client_email VARCHAR(100),
    accountant_name VARCHAR(100),
    appointment_time TIMESTAMP WITH TIME ZONE NOT NULL,
    appointment_time_formatted VARCHAR(200),
    client_type VARCHAR(50),
    language VARCHAR(10) DEFAULT 'en',
    status VARCHAR(50) DEFAULT 'confirmed',
    booking_url TEXT,
    event_type_uri VARCHAR(255),
    service_id VARCHAR(100),
    staff_id VARCHAR(100),
    ms_booking_id VARCHAR(100),
    reminder_sent BOOLEAN DEFAULT FALSE,
    reminder_sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.users (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval('public.users_id_seq'::pg_catalog.regclass),
    tenant_id INTEGER DEFAULT 1,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    role VARCHAR(50) DEFAULT 'admin',
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(100),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.tenants (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval('public.tenants_id_seq'::pg_catalog.regclass),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(50) UNIQUE NOT NULL,
    industry VARCHAR(100),
    phone_number VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    settings JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.knowledge_articles (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval(
            'public.knowledge_articles_id_seq'::pg_catalog.regclass
        ),
    tenant_id INTEGER DEFAULT 1,
    category VARCHAR(100),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    priority INTEGER DEFAULT 0,
    tags TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.callers (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval('public.callers_id_seq'::pg_catalog.regclass),
    phone_number VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    call_count INTEGER DEFAULT 1,
    first_call TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    last_call TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    tenant_id INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.analytics_events (
    id INTEGER PRIMARY KEY
        DEFAULT pg_catalog.nextval(
            'public.analytics_events_id_seq'::pg_catalog.regclass
        ),
    tenant_id INTEGER DEFAULT 1,
    call_sid VARCHAR(100),
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    duration_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT pg_catalog.now()
);

CREATE TABLE public.admin_users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE public.mfa_codes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER
        REFERENCES public.admin_users(id) ON DELETE CASCADE,
    code VARCHAR(6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE
);

CREATE TABLE public.admin_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER
        REFERENCES public.admin_users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT
);

CREATE TABLE public.call_logs (
    id SERIAL PRIMARY KEY,
    call_sid VARCHAR(100) UNIQUE,
    phone_number VARCHAR(50),
    caller_name VARCHAR(255),
    language VARCHAR(10),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration_seconds INTEGER,
    status VARCHAR(50) DEFAULT 'in_progress',
    transfer_requested BOOLEAN DEFAULT FALSE,
    dtmf_count INTEGER DEFAULT 0,
    booking_made BOOLEAN DEFAULT FALSE,
    notes TEXT
);

CREATE TABLE public.bookings (
    id SERIAL PRIMARY KEY,
    call_sid VARCHAR(100),
    phone_number VARCHAR(50),
    client_name VARCHAR(255),
    client_email VARCHAR(255),
    accountant_name VARCHAR(255),
    appointment_time TIMESTAMP,
    client_type VARCHAR(50),
    language VARCHAR(10),
    status VARCHAR(50) DEFAULT 'confirmed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ms_booking_id VARCHAR(255),
    notes TEXT
);

CREATE TABLE public.sms_logs (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(50),
    client_name VARCHAR(255),
    message TEXT,
    provider VARCHAR(50),
    status VARCHAR(50),
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    booking_id INTEGER REFERENCES public.bookings(id),
    booking_link VARCHAR(500),
    error_message TEXT
);

CREATE TABLE public.api_usage (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    characters_used INTEGER DEFAULT 0,
    audio_seconds FLOAT DEFAULT 0,
    estimated_cost DECIMAL(10, 4) DEFAULT 0,
    recorded_at DATE DEFAULT CURRENT_DATE,
    UNIQUE (service_name, recorded_at)
);

CREATE TABLE public.system_metrics (
    id SERIAL PRIMARY KEY,
    cpu_percent FLOAT,
    memory_percent FLOAT,
    disk_percent FLOAT,
    active_calls INTEGER DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_calls_phone ON public.calls(phone_number);
CREATE INDEX idx_calls_status ON public.calls(status);
CREATE INDEX idx_calls_started ON public.calls(started_at);
CREATE INDEX idx_calls_tenant ON public.calls(tenant_id);
CREATE INDEX idx_conversations_call_sid ON public.conversations(call_sid);
CREATE INDEX idx_turns_call_sid ON public.conversation_turns(call_sid);
CREATE INDEX idx_turns_timestamp ON public.conversation_turns(timestamp);
CREATE INDEX idx_appointments_phone ON public.appointments(client_phone);
CREATE INDEX idx_appointments_time ON public.appointments(appointment_time);
CREATE INDEX idx_appointments_status ON public.appointments(status);
CREATE INDEX idx_knowledge_tenant ON public.knowledge_articles(tenant_id);
CREATE INDEX idx_knowledge_category ON public.knowledge_articles(category);
CREATE INDEX idx_knowledge_language ON public.knowledge_articles(language);
CREATE INDEX idx_callers_phone ON public.callers(phone_number);
CREATE INDEX idx_callers_tenant ON public.callers(tenant_id);
CREATE INDEX idx_callers_last_call ON public.callers(last_call);
CREATE INDEX idx_analytics_tenant ON public.analytics_events(tenant_id);
CREATE INDEX idx_analytics_type ON public.analytics_events(event_type);
CREATE INDEX idx_analytics_timestamp ON public.analytics_events(timestamp);
CREATE INDEX idx_call_logs_started_at ON public.call_logs(started_at);
CREATE INDEX idx_call_logs_phone_number ON public.call_logs(phone_number);
CREATE INDEX idx_bookings_appointment_time ON public.bookings(appointment_time);
CREATE INDEX idx_bookings_accountant ON public.bookings(accountant_name);
CREATE INDEX idx_sms_logs_sent_at ON public.sms_logs(sent_at);
CREATE INDEX idx_admin_sessions_token ON public.admin_sessions(session_token);
CREATE INDEX idx_mfa_codes_user_expires
    ON public.mfa_codes(user_id, expires_at);

CREATE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = pg_catalog.now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_calls_updated_at
    BEFORE UPDATE ON public.calls
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON public.conversations
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_appointments_updated_at
    BEFORE UPDATE ON public.appointments
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON public.tenants
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_knowledge_updated_at
    BEFORE UPDATE ON public.knowledge_articles
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_callers_updated_at
    BEFORE UPDATE ON public.callers
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
