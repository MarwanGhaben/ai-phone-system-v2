# AI Voice Platform - SaaS Transformation Design Document

**Version**: 1.0  
**Date**: May 2026  
**Author**: Architecture Review  
**Status**: Planning

---

## Executive Summary

Transform the current single-tenant AI Voice Platform (built for Flexible Accounting) into a multi-tenant SaaS product that can serve multiple businesses, each with their own AI phone agent, branding, calendar integration, and isolated data.

**Current State**: Working single-tenant system with database schema partially prepared for multi-tenancy  
**Target State**: Production-ready multi-tenant SaaS with self-service onboarding  
**Estimated Timeline**: 10-14 weeks (MVP in 4-5 weeks)

---

## Table of Contents

1. [Infrastructure Strategy](#1-infrastructure-strategy)
2. [Architecture Overview](#2-architecture-overview)
3. [Database Changes](#3-database-changes)
4. [Core Code Changes](#4-core-code-changes)
5. [New Components to Build](#5-new-components-to-build)
6. [Security Considerations](#6-security-considerations)
7. [Implementation Phases](#7-implementation-phases)
8. [Cost Analysis](#8-cost-analysis)
9. [Risk Mitigation](#9-risk-mitigation)

---

## 1. Infrastructure Strategy

### Recommended Setup

| Environment | Purpose | Server | Domain |
|-------------|---------|--------|--------|
| **Production (Current)** | Flexible Accounting ONLY | Current server | aiagent.ghaben.ca |
| **Development/Staging** | SaaS development & testing | NEW server | dev.aivoiceplatform.com (example) |
| **Production (SaaS)** | Multi-tenant SaaS | NEW server (later) | app.aivoiceplatform.com (example) |

### Why Separate Servers?

1. **Zero risk to existing client** - Flexible Accounting keeps working
2. **Clean slate for multi-tenancy** - No migration headaches
3. **Different deployment cycles** - SaaS can iterate fast without affecting FA
4. **Eventually**: Migrate FA to be a tenant on the SaaS platform

### Development Server Specs (Recommended)

```
Cloud Provider: DigitalOcean / Linode / AWS Lightsail
Instance Type:  4 vCPU, 8GB RAM, 160GB SSD
OS:             Ubuntu 22.04 LTS
Monthly Cost:   ~$48-80/month

Services:
- Docker + Docker Compose
- PostgreSQL 16
- Redis 7
- Nginx (reverse proxy + SSL)
- Certbot (Let's Encrypt)
```

### Domain Structure for SaaS

```
app.aivoiceplatform.com          → Main SaaS application
api.aivoiceplatform.com          → API endpoints (Twilio webhooks)
admin.aivoiceplatform.com        → Super-admin panel (your control panel)

Each tenant gets:
- Unique Twilio phone number
- Dashboard access at: app.aivoiceplatform.com/dashboard (tenant-scoped by login)
```

---

## 2. Architecture Overview

### Current Architecture (Single-Tenant)

```
┌─────────────────────────────────────────────────────────┐
│                    CURRENT SYSTEM                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Twilio ──→ /api/incoming-call ──→ Orchestrator        │
│                     │                    │               │
│                     │              [HARDCODED]           │
│                     │           "Flexible Accounting"    │
│                     │              "Sarah"               │
│                     ▼                    │               │
│               PostgreSQL ◄───────────────┘               │
│            (tenant_id = 1 always)                        │
│                     │                                    │
│                     ▼                                    │
│               Dashboard                                  │
│         (sees ALL data, no isolation)                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Target Architecture (Multi-Tenant)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SAAS ARCHITECTURE                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Twilio Phone Numbers                                                │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐                                │
│   │+1-555-01│ │+1-555-02│ │+1-555-03│  ... N numbers                 │
│   └────┬────┘ └────┬────┘ └────┬────┘                                │
│        │           │           │                                      │
│        └───────────┼───────────┘                                      │
│                    ▼                                                  │
│   ┌────────────────────────────────────┐                             │
│   │      TENANT RESOLVER SERVICE       │                             │
│   │  phone_number → tenant_id lookup   │                             │
│   └────────────────┬───────────────────┘                             │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────────┐                             │
│   │      TENANT CONFIG LOADER          │                             │
│   │  Load from tenants table:          │                             │
│   │  - Business name, agent name       │                             │
│   │  - Services, hours, language       │                             │
│   │  - MS Bookings credentials         │                             │
│   │  - Voice ID, SMS number            │                             │
│   └────────────────┬───────────────────┘                             │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────────┐                             │
│   │         ORCHESTRATOR               │                             │
│   │  System prompt built DYNAMICALLY   │                             │
│   │  from tenant config                │                             │
│   └────────────────┬───────────────────┘                             │
│                    │                                                  │
│        ┌───────────┼───────────┐                                     │
│        ▼           ▼           ▼                                     │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐                                │
│   │   LLM   │ │Calendar │ │   SMS   │  All services receive          │
│   │(OpenAI) │ │(dynamic)│ │(dynamic)│  tenant_id context             │
│   └─────────┘ └─────────┘ └─────────┘                                │
│                    │                                                  │
│                    ▼                                                  │
│   ┌────────────────────────────────────┐                             │
│   │           PostgreSQL               │                             │
│   │  ALL writes include tenant_id      │                             │
│   │  ALL reads filter by tenant_id     │                             │
│   └────────────────────────────────────┘                             │
│                    │                                                  │
│        ┌───────────┴───────────┐                                     │
│        ▼                       ▼                                     │
│   ┌─────────────┐       ┌─────────────┐                              │
│   │  Tenant     │       │  Super      │                              │
│   │  Dashboard  │       │  Admin      │                              │
│   │ (isolated)  │       │  Panel      │                              │
│   └─────────────┘       └─────────────┘                              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Database Changes

### Existing Schema (Already Present)

The `tenants` table already exists:

```sql
-- ALREADY EXISTS in init_db.sql
CREATE TABLE IF NOT EXISTS tenants (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    industry VARCHAR(100),
    phone_number VARCHAR(20),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### New Schema Additions Required

```sql
-- =====================================================
-- TENANT CONFIGURATION (expand settings JSONB)
-- =====================================================

-- Example tenant settings structure:
{
    "business": {
        "name": "ABC Tax Services",
        "display_name": "ABC Tax",
        "industry": "accounting",
        "timezone": "America/Toronto",
        "languages": ["en", "ar"],
        "default_language": "en"
    },
    "agent": {
        "name": "Emma",
        "personality": "friendly and professional",
        "greeting_style": "warm"
    },
    "services": [
        {"name": "Personal Tax Filing", "name_ar": "تقديم الضرائب الشخصية"},
        {"name": "Corporate Tax", "name_ar": "ضرائب الشركات"},
        {"name": "Bookkeeping", "name_ar": "مسك الدفاتر"}
    ],
    "business_hours": {
        "monday": {"open": "09:00", "close": "17:00"},
        "tuesday": {"open": "09:00", "close": "17:00"},
        "wednesday": {"open": "09:00", "close": "17:00"},
        "thursday": {"open": "09:00", "close": "17:00"},
        "friday": {"open": "09:00", "close": "17:00"},
        "saturday": null,
        "sunday": null
    },
    "booking_rules": {
        "max_days_ahead": 2,
        "min_hours_notice": 1,
        "slot_duration_minutes": 30
    },
    "branding": {
        "sms_signature": "— ABC Tax Services",
        "email_footer": "ABC Tax Services | 123 Main St",
        "primary_color": "#1a73e8"
    },
    "voice": {
        "elevenlabs_voice_id": "Rachel",
        "speech_rate": 1.0
    }
}

-- =====================================================
-- TENANT CREDENTIALS (encrypted, separate table)
-- =====================================================

CREATE TABLE IF NOT EXISTS tenant_credentials (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    credential_type VARCHAR(50) NOT NULL,  -- 'ms_bookings', 'google_calendar', 'twilio', 'elevenlabs'
    credentials_encrypted BYTEA NOT NULL,   -- AES-256 encrypted JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, credential_type)
);

-- Index for fast lookup
CREATE INDEX idx_tenant_credentials_lookup ON tenant_credentials(tenant_id, credential_type);

-- =====================================================
-- PHONE NUMBER TO TENANT MAPPING
-- =====================================================

CREATE TABLE IF NOT EXISTS tenant_phone_numbers (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    phone_number VARCHAR(20) NOT NULL UNIQUE,  -- E.164 format: +14165551234
    is_primary BOOLEAN DEFAULT false,
    provider VARCHAR(20) DEFAULT 'twilio',     -- 'twilio', 'telnyx'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fast lookup by phone number (critical for call routing)
CREATE UNIQUE INDEX idx_phone_tenant ON tenant_phone_numbers(phone_number);

-- =====================================================
-- TENANT USAGE TRACKING (for billing)
-- =====================================================

CREATE TABLE IF NOT EXISTS tenant_usage (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    usage_date DATE NOT NULL,
    
    -- Call metrics
    total_calls INTEGER DEFAULT 0,
    total_call_minutes DECIMAL(10,2) DEFAULT 0,
    
    -- AI usage
    llm_tokens_input INTEGER DEFAULT 0,
    llm_tokens_output INTEGER DEFAULT 0,
    whisper_minutes DECIMAL(10,2) DEFAULT 0,
    elevenlabs_characters INTEGER DEFAULT 0,
    
    -- SMS
    sms_sent INTEGER DEFAULT 0,
    sms_received INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, usage_date)
);

CREATE INDEX idx_tenant_usage_date ON tenant_usage(tenant_id, usage_date);

-- =====================================================
-- SUPER ADMIN USERS (SaaS operators)
-- =====================================================

CREATE TABLE IF NOT EXISTS super_admins (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- =====================================================
-- TENANT SUBSCRIPTION/BILLING
-- =====================================================

CREATE TABLE IF NOT EXISTS tenant_subscriptions (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    plan_type VARCHAR(50) NOT NULL,  -- 'starter', 'professional', 'enterprise'
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'paused', 'cancelled'
    
    -- Limits
    max_monthly_calls INTEGER,
    max_monthly_minutes INTEGER,
    max_phone_numbers INTEGER,
    
    -- Billing
    monthly_base_price DECIMAL(10,2),
    per_minute_rate DECIMAL(10,4),
    billing_email VARCHAR(255),
    
    -- Dates
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_period_start DATE,
    current_period_end DATE,
    cancelled_at TIMESTAMP
);
```

### Migration Strategy

1. **Don't modify production** - These changes go to the new SaaS database only
2. **Seed with test tenant** - Create a test business to develop against
3. **Later**: Export Flexible Accounting data and import as tenant #1 on SaaS

---

## 4. Core Code Changes

### 4.1 Tenant Context Object

Create a new service that carries tenant context through the entire call flow:

```python
# services/tenant/tenant_context.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class TenantContext:
    """Immutable tenant context passed through entire call flow"""
    tenant_id: int
    business_name: str
    business_display_name: str
    agent_name: str
    timezone: str
    languages: List[str]
    default_language: str
    services: List[Dict[str, str]]
    business_hours: Dict[str, Any]
    booking_rules: Dict[str, Any]
    branding: Dict[str, str]
    voice_id: str
    
    # Credentials (loaded separately, not stored in context)
    ms_bookings_config: Optional[Dict] = None
    
    @classmethod
    def from_db_row(cls, tenant_row: dict) -> 'TenantContext':
        """Build context from database tenant record"""
        settings = tenant_row.get('settings', {})
        business = settings.get('business', {})
        agent = settings.get('agent', {})
        
        return cls(
            tenant_id=tenant_row['id'],
            business_name=business.get('name', tenant_row['name']),
            business_display_name=business.get('display_name', tenant_row['name']),
            agent_name=agent.get('name', 'Assistant'),
            timezone=business.get('timezone', 'America/Toronto'),
            languages=business.get('languages', ['en']),
            default_language=business.get('default_language', 'en'),
            services=settings.get('services', []),
            business_hours=settings.get('business_hours', {}),
            booking_rules=settings.get('booking_rules', {}),
            branding=settings.get('branding', {}),
            voice_id=settings.get('voice', {}).get('elevenlabs_voice_id', 'Rachel'),
        )
```

### 4.2 Tenant Resolver Service

```python
# services/tenant/tenant_resolver.py

from typing import Optional
from services.database import get_db_pool
from services.tenant.tenant_context import TenantContext
from loguru import logger

class TenantResolver:
    """Resolves tenant from incoming phone number"""
    
    _cache: dict = {}  # Simple in-memory cache
    
    @classmethod
    async def resolve_by_phone(cls, phone_number: str) -> Optional[TenantContext]:
        """
        Look up tenant by the Twilio 'To' phone number.
        This is the FIRST thing called when a call comes in.
        """
        # Normalize phone number to E.164
        normalized = cls._normalize_phone(phone_number)
        
        # Check cache first
        if normalized in cls._cache:
            return cls._cache[normalized]
        
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT t.* 
                FROM tenants t
                JOIN tenant_phone_numbers tpn ON t.id = tpn.tenant_id
                WHERE tpn.phone_number = $1
                AND t.is_active = true
            """, normalized)
            
            if not row:
                logger.error(f"No tenant found for phone number: {normalized}")
                return None
            
            context = TenantContext.from_db_row(dict(row))
            cls._cache[normalized] = context
            return context
    
    @classmethod
    async def resolve_by_id(cls, tenant_id: int) -> Optional[TenantContext]:
        """Look up tenant by ID (for dashboard, API calls)"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM tenants WHERE id = $1 AND is_active = true",
                tenant_id
            )
            if not row:
                return None
            return TenantContext.from_db_row(dict(row))
    
    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normalize to E.164 format"""
        import re
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        return f"+{digits}"
    
    @classmethod
    def clear_cache(cls):
        """Clear cache (call after tenant config updates)"""
        cls._cache.clear()
```

### 4.3 Modified Incoming Call Handler

```python
# api/main.py - modified /api/incoming-call endpoint

@app.post("/api/incoming-call")
async def incoming_call(request: Request):
    form_data = await request.form()
    
    # STEP 1: Resolve tenant from the "To" phone number
    to_number = form_data.get("To", "")
    tenant_context = await TenantResolver.resolve_by_phone(to_number)
    
    if not tenant_context:
        logger.error(f"Call to unregistered number: {to_number}")
        # Return a polite rejection or route to default
        response = VoiceResponse()
        response.say("Sorry, this number is not in service.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")
    
    # STEP 2: Create call record with tenant_id
    call_sid = form_data.get("CallSid")
    from_number = form_data.get("From")
    
    call_id = await create_call_record(
        call_sid=call_sid,
        from_number=from_number,
        to_number=to_number,
        tenant_id=tenant_context.tenant_id  # <-- NOW INCLUDED
    )
    
    # STEP 3: Initialize orchestrator WITH tenant context
    orchestrator = ConversationOrchestrator(
        call_id=call_id,
        caller_phone=from_number,
        tenant_context=tenant_context  # <-- PASSED THROUGH
    )
    
    # ... rest of call handling
```

### 4.4 Dynamic System Prompt Builder

```python
# services/conversation/prompt_builder.py

from services.tenant.tenant_context import TenantContext
from datetime import datetime
from pytz import timezone

class PromptBuilder:
    """Builds system prompts dynamically from tenant config"""
    
    @staticmethod
    def build_system_prompt(tenant: TenantContext, language: str = "en") -> str:
        """
        Generate the full system prompt for the AI agent.
        This replaces the hardcoded prompt in orchestrator.py
        """
        
        # Get current time in tenant's timezone
        tz = timezone(tenant.timezone)
        now = datetime.now(tz)
        
        # Build services list
        if language == "ar":
            services_list = "\n".join(
                f"- {s.get('name_ar', s['name'])}" 
                for s in tenant.services
            )
        else:
            services_list = "\n".join(
                f"- {s['name']}" 
                for s in tenant.services
            )
        
        # Build hours string
        hours_str = PromptBuilder._format_business_hours(
            tenant.business_hours, language
        )
        
        prompt = f"""You are {tenant.agent_name}, a friendly and professional AI receptionist for {tenant.business_name}.

## Your Role
You answer phone calls, help callers with inquiries, and book appointments with staff members.

## Business Information
- **Company**: {tenant.business_display_name}
- **Services Offered**:
{services_list}

## Business Hours
{hours_str}

## Booking Rules
- Maximum booking window: {tenant.booking_rules.get('max_days_ahead', 2)} business days ahead
- Minimum notice: {tenant.booking_rules.get('min_hours_notice', 1)} hour(s)
- Appointment duration: {tenant.booking_rules.get('slot_duration_minutes', 30)} minutes

## Communication Style
- Be warm, professional, and helpful
- Keep responses concise (this is a phone call)
- If the caller speaks Arabic, respond in Arabic
- Always confirm details before booking

## Current Date/Time
- Timezone: {tenant.timezone}
- Current time: {now.strftime('%I:%M %p')}
- Today: {now.strftime('%A, %B %d, %Y')}
"""
        return prompt
    
    @staticmethod
    def _format_business_hours(hours: dict, language: str) -> str:
        days_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_ar = ['الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد']
        
        days = days_ar if language == "ar" else days_en
        lines = []
        
        for i, day_key in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
            day_hours = hours.get(day_key)
            if day_hours:
                lines.append(f"- {days[i]}: {day_hours['open']} - {day_hours['close']}")
            else:
                closed = "مغلق" if language == "ar" else "Closed"
                lines.append(f"- {days[i]}: {closed}")
        
        return "\n".join(lines)
```

### 4.5 Dashboard Query Scoping

Every dashboard query must be scoped by tenant_id:

```python
# services/dashboard/dashboard_routes.py - BEFORE (unsafe)

@router.get("/api/dashboard/calls")
async def get_calls():
    query = "SELECT * FROM call_logs ORDER BY created_at DESC LIMIT 100"
    # DANGER: Returns ALL tenants' calls!

# AFTER (tenant-scoped)

@router.get("/api/dashboard/calls")
async def get_calls(request: Request):
    # Get tenant_id from authenticated session
    tenant_id = request.state.tenant_id  # Set by auth middleware
    
    query = """
        SELECT * FROM call_logs 
        WHERE tenant_id = $1 
        ORDER BY created_at DESC 
        LIMIT 100
    """
    results = await db.fetch(query, tenant_id)
    # Safe: Only returns this tenant's calls
```

---

## 5. New Components to Build

### 5.1 Super Admin Panel

A separate admin interface for YOU (the SaaS operator) to manage tenants.

**Features:**
- List all tenants with status, usage, subscription
- Add new tenant (wizard flow)
- Edit tenant configuration
- Assign phone numbers to tenants
- View cross-tenant analytics
- Manage subscriptions/billing
- System health monitoring

**Tech Stack:**
- FastAPI backend (separate routes, `/super-admin/*`)
- React or simple server-rendered templates
- Protected by separate super-admin auth

### 5.2 Tenant Onboarding Wizard

Step-by-step flow for adding a new business:

```
Step 1: Basic Info
├── Business name
├── Industry
├── Primary contact email
└── Timezone

Step 2: AI Agent Configuration  
├── Agent name (e.g., "Emma", "Sarah")
├── Personality style
├── Languages supported
└── Services offered (add/remove)

Step 3: Business Hours
├── Set hours for each day
└── Holiday handling

Step 4: Calendar Integration
├── Choose: Microsoft Bookings / Google Calendar / None
├── OAuth flow or credential entry
└── Test connection

Step 5: Phone Number
├── Provision new Twilio number, OR
├── Port existing number
└── Configure SMS settings

Step 6: Review & Activate
├── Review all settings
├── Test call (optional)
└── Activate tenant
```

### 5.3 Tenant Dashboard Enhancements

Each tenant's dashboard should show:
- Their own calls/appointments/analytics only
- Tenant-specific settings page
- Usage meter (calls this month, minutes used)
- Billing/invoice history (if self-service)

### 5.4 Credential Vault Service

Securely store and retrieve per-tenant API credentials:

```python
# services/tenant/credential_vault.py

from cryptography.fernet import Fernet
import os
import json

class CredentialVault:
    """Encrypt/decrypt tenant credentials"""
    
    def __init__(self):
        key = os.environ.get('CREDENTIAL_ENCRYPTION_KEY')
        if not key:
            raise ValueError("CREDENTIAL_ENCRYPTION_KEY not set")
        self.cipher = Fernet(key.encode())
    
    async def store_credentials(
        self, 
        tenant_id: int, 
        credential_type: str, 
        credentials: dict
    ):
        """Store encrypted credentials"""
        encrypted = self.cipher.encrypt(json.dumps(credentials).encode())
        
        await db.execute("""
            INSERT INTO tenant_credentials (tenant_id, credential_type, credentials_encrypted)
            VALUES ($1, $2, $3)
            ON CONFLICT (tenant_id, credential_type) 
            DO UPDATE SET credentials_encrypted = $3, updated_at = NOW()
        """, tenant_id, credential_type, encrypted)
    
    async def get_credentials(
        self, 
        tenant_id: int, 
        credential_type: str
    ) -> dict:
        """Retrieve and decrypt credentials"""
        row = await db.fetchrow("""
            SELECT credentials_encrypted 
            FROM tenant_credentials
            WHERE tenant_id = $1 AND credential_type = $2
        """, tenant_id, credential_type)
        
        if not row:
            return None
        
        decrypted = self.cipher.decrypt(row['credentials_encrypted'])
        return json.loads(decrypted)
```

---

## 6. Security Considerations

### 6.1 Tenant Isolation (CRITICAL)

| Layer | Isolation Method |
|-------|------------------|
| **Database** | All queries MUST include `WHERE tenant_id = $X` |
| **API** | Middleware extracts tenant_id from auth token, injects into request.state |
| **File Storage** | Separate folders per tenant: `/storage/{tenant_id}/recordings/` |
| **Logs** | Include tenant_id in all log entries for filtering |
| **Cache** | Prefix all cache keys with tenant_id: `tenant:{id}:key` |

### 6.2 Authentication Layers

```
┌─────────────────────────────────────────────┐
│              AUTHENTICATION                  │
├─────────────────────────────────────────────┤
│                                              │
│  Super Admin (you)                           │
│  └── Separate login: /super-admin/login     │
│  └── MFA required                            │
│  └── Access to all tenants                   │
│                                              │
│  Tenant Admin (business owners)              │
│  └── Login: /dashboard/login                 │
│  └── Scoped to their tenant_id only          │
│  └── Cannot see other tenants                │
│                                              │
│  Tenant Staff (optional, future)             │
│  └── Limited permissions within tenant       │
│                                              │
└─────────────────────────────────────────────┘
```

### 6.3 Credential Security

- All third-party credentials (MS Bookings, Twilio sub-accounts) encrypted at rest
- Encryption key stored in environment, not in code or DB
- Credentials never logged, never returned in API responses
- Rotate encryption key annually

### 6.4 Audit Logging

Log all sensitive operations with tenant context:

```python
logger.info(
    "Appointment booked",
    tenant_id=tenant_context.tenant_id,
    caller_phone=caller_phone,
    staff_name=staff_name,
    appointment_time=appointment_time
)
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Multi-tenant infrastructure without breaking changes

| Task | Est. Hours | Priority |
|------|------------|----------|
| Set up development server | 4h | P0 |
| Clone repo, configure new environment | 2h | P0 |
| Create new database with expanded schema | 4h | P0 |
| Implement TenantContext dataclass | 2h | P0 |
| Implement TenantResolver service | 4h | P0 |
| Create phone-to-tenant mapping table | 2h | P0 |
| Seed test tenant with sample config | 2h | P0 |
| **Subtotal** | **20h** | |

**Deliverable**: Can look up tenant from phone number

---

### Phase 2: Call Flow Integration (Weeks 3-4)
**Goal**: Calls route to correct tenant, use dynamic config

| Task | Est. Hours | Priority |
|------|------------|----------|
| Modify `/api/incoming-call` to resolve tenant | 4h | P0 |
| Pass tenant_context through Orchestrator | 6h | P0 |
| Build dynamic PromptBuilder service | 8h | P0 |
| Replace hardcoded system prompt | 4h | P0 |
| Update SMS service for dynamic branding | 4h | P1 |
| Update email service for dynamic branding | 4h | P1 |
| Test end-to-end with test tenant | 4h | P0 |
| **Subtotal** | **34h** | |

**Deliverable**: Test calls work with dynamic tenant config

---

### Phase 3: Dashboard Isolation (Week 5)
**Goal**: Each tenant sees only their data

| Task | Est. Hours | Priority |
|------|------------|----------|
| Add tenant_id extraction to auth middleware | 4h | P0 |
| Audit ALL dashboard queries (there are ~30) | 2h | P0 |
| Add `WHERE tenant_id = $X` to all queries | 8h | P0 |
| Update dashboard UI to show tenant name | 2h | P1 |
| Test with 2 tenants, verify isolation | 4h | P0 |
| Security review of isolation | 4h | P0 |
| **Subtotal** | **24h** | |

**Deliverable**: Dashboard is tenant-isolated

---

### Phase 4: Calendar Multi-Tenancy (Week 6)
**Goal**: Each tenant connects their own calendar

| Task | Est. Hours | Priority |
|------|------------|----------|
| Implement CredentialVault service | 6h | P0 |
| Modify MS Bookings service to accept credentials | 6h | P0 |
| Load tenant's MS Bookings creds at call time | 4h | P0 |
| (Optional) Add Google Calendar integration | 12h | P2 |
| Test booking flow with tenant-specific calendar | 4h | P0 |
| **Subtotal** | **20-32h** | |

**Deliverable**: Each tenant uses their own calendar

---

### Phase 5: Super Admin Panel (Weeks 7-8)
**Goal**: You can onboard new tenants

| Task | Est. Hours | Priority |
|------|------------|----------|
| Design super admin routes (`/super-admin/*`) | 2h | P0 |
| Super admin authentication | 6h | P0 |
| Tenant list view | 4h | P0 |
| Add tenant wizard (basic info) | 8h | P0 |
| Tenant config editor | 8h | P0 |
| Phone number assignment UI | 4h | P0 |
| Credential entry for MS Bookings | 6h | P0 |
| Tenant activation/deactivation | 4h | P1 |
| **Subtotal** | **42h** | |

**Deliverable**: Can onboard new tenants via admin panel

---

### Phase 6: Polish & Production Prep (Weeks 9-10)
**Goal**: Production-ready SaaS

| Task | Est. Hours | Priority |
|------|------------|----------|
| Usage tracking per tenant | 8h | P1 |
| Error handling improvements | 6h | P1 |
| Logging with tenant context everywhere | 4h | P1 |
| Performance testing with multiple tenants | 6h | P1 |
| Security audit | 8h | P0 |
| Documentation | 6h | P1 |
| Production deployment setup | 8h | P0 |
| **Subtotal** | **46h** | |

**Deliverable**: Ready for first real tenant

---

### Summary Timeline

```
Week 1-2:   Foundation (tenant resolution)
Week 3-4:   Call flow integration (dynamic prompts)
Week 5:     Dashboard isolation (security)
Week 6:     Calendar multi-tenancy
Week 7-8:   Super admin panel (onboarding)
Week 9-10:  Polish & production prep

Total: ~186 hours (~10-12 weeks at 15-20 hrs/week)
MVP (through Week 5): ~78 hours (~4-5 weeks)
```

---

## 8. Cost Analysis

### Development Costs (One-Time)

| Item | Cost |
|------|------|
| Development server (3 months) | ~$150-240 |
| Test Twilio numbers (2-3) | ~$6-9/month |
| Domain name (optional new one) | ~$12/year |
| SSL certificate | Free (Let's Encrypt) |
| Your time | Your hourly rate × ~186 hours |

### Operational Costs (Per Tenant, Monthly)

| Item | Est. Cost/Tenant |
|------|------------------|
| Twilio phone number | $1-2 |
| Twilio voice (100 min) | $1.30 |
| OpenAI GPT-4 (~500 calls) | $5-15 |
| ElevenLabs TTS | $5-11 |
| Whisper STT | $2-5 |
| SMS (50 messages) | $1-2 |
| **Total per tenant** | **~$15-35/month** |

### Suggested Pricing Tiers

| Plan | Monthly Price | Included | Target |
|------|---------------|----------|--------|
| **Starter** | $99/month | 200 calls, 1 number | Solo practitioners |
| **Professional** | $249/month | 500 calls, 2 numbers | Small firms |
| **Enterprise** | $499/month | 1500 calls, 5 numbers | Multi-location |

With ~$30 cost per tenant, you'd have healthy margins.

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Data leak between tenants** | Strict code review for all queries; automated tests that verify isolation |
| **Credential exposure** | Encryption at rest; never log credentials; access audit |
| **Single tenant takes down system** | Rate limiting per tenant; resource quotas |
| **Calendar OAuth expires** | Monitor token expiry; alert tenant to re-auth |
| **Twilio number spam** | Implement call screening; block known spam patterns |
| **OpenAI rate limits** | Queue system; per-tenant rate limiting |

---

## Appendix A: File Changes Summary

Files that need modification for multi-tenancy:

```
CRITICAL (must change):
├── api/main.py                           # Tenant resolution on incoming call
├── services/conversation/orchestrator.py  # Accept tenant context, dynamic prompt
├── services/dashboard/dashboard_routes.py # Add tenant_id to ALL queries
├── services/calendar/ms_bookings_service.py # Load credentials per tenant
├── services/sms/telnyx_sms_service.py    # Dynamic branding

NEW FILES:
├── services/tenant/
│   ├── tenant_context.py                 # TenantContext dataclass
│   ├── tenant_resolver.py                # Phone → tenant lookup
│   ├── credential_vault.py               # Encrypted credential storage
│   └── prompt_builder.py                 # Dynamic system prompt
├── api/super_admin/
│   ├── routes.py                         # Super admin API
│   ├── auth.py                           # Super admin auth
│   └── templates/                        # Admin panel UI

DATABASE:
├── scripts/init_db.sql                   # Add new tables (or migrations/)
```

---

## Appendix B: Development Server Setup Checklist

```bash
# 1. Provision server (DigitalOcean/Linode/AWS)
#    - Ubuntu 22.04 LTS
#    - 4 vCPU, 8GB RAM
#    - 160GB SSD

# 2. Initial setup
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git nginx certbot

# 3. Clone repository
git clone <your-repo> /opt/ai-voice-saas
cd /opt/ai-voice-saas

# 4. Create new .env with SaaS-specific values
cp .env.example .env
nano .env  # Configure for new environment

# 5. Generate credential encryption key
echo "CREDENTIAL_ENCRYPTION_KEY=$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" >> .env

# 6. Initialize database with new schema
docker-compose up -d postgres
docker exec -i <postgres-container> psql -U <user> -d <db> < scripts/init_db_saas.sql

# 7. Start all services
docker-compose up -d

# 8. Configure SSL
sudo certbot --nginx -d dev.yourdomain.com

# 9. Configure Twilio webhook to point to new server
# Twilio Console → Phone Numbers → Configure → Webhook URL
```

---

## Next Steps

1. **Approve this design** - Review and confirm approach
2. **Provision dev server** - Get infrastructure ready
3. **Start Phase 1** - Foundation work
4. **Weekly check-ins** - Track progress against timeline

---

*Document Version History*
- v1.0 (May 2026) - Initial design document
