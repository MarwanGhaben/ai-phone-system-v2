-- =====================================================
-- AI Voice Platform v2 - Dashboard Database Schema
-- =====================================================
-- Run this script to create all necessary tables for the dashboard

-- Admin users table
CREATE TABLE IF NOT EXISTS admin_users (
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

-- MFA codes table
CREATE TABLE IF NOT EXISTS mfa_codes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES admin_users(id) ON DELETE CASCADE,
    code VARCHAR(6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    used BOOLEAN DEFAULT FALSE
);

-- Sessions table
CREATE TABLE IF NOT EXISTS admin_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES admin_users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT
);

-- Call logs table (for detailed call tracking)
CREATE TABLE IF NOT EXISTS call_logs (
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

-- Bookings table
CREATE TABLE IF NOT EXISTS bookings (
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

-- SMS logs table
CREATE TABLE IF NOT EXISTS sms_logs (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(50),
    client_name VARCHAR(255),
    message TEXT,
    provider VARCHAR(50),
    status VARCHAR(50),
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    booking_id INTEGER REFERENCES bookings(id),
    booking_link VARCHAR(500),
    error_message TEXT
);

-- API usage tracking table
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    characters_used INTEGER DEFAULT 0,
    audio_seconds FLOAT DEFAULT 0,
    estimated_cost DECIMAL(10, 4) DEFAULT 0,
    recorded_at DATE DEFAULT CURRENT_DATE,
    UNIQUE(service_name, recorded_at)
);

-- System metrics table (for historical tracking)
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    cpu_percent FLOAT,
    memory_percent FLOAT,
    disk_percent FLOAT,
    active_calls INTEGER DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_call_logs_started_at ON call_logs(started_at);
CREATE INDEX IF NOT EXISTS idx_call_logs_phone_number ON call_logs(phone_number);
CREATE INDEX IF NOT EXISTS idx_bookings_appointment_time ON bookings(appointment_time);
CREATE INDEX IF NOT EXISTS idx_bookings_accountant ON bookings(accountant_name);
CREATE INDEX IF NOT EXISTS idx_sms_logs_sent_at ON sms_logs(sent_at);
CREATE INDEX IF NOT EXISTS idx_admin_sessions_token ON admin_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_mfa_codes_user_expires ON mfa_codes(user_id, expires_at);

-- Insert default admin user (password: admin123 - CHANGE THIS!)
-- Password hash is bcrypt of 'admin123'
INSERT INTO admin_users (username, email, password_hash, is_superuser)
VALUES ('admin', 'admin@flexibleaccounting.ca', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.V4ferE6EbLMOoS', TRUE)
ON CONFLICT (username) DO NOTHING;
