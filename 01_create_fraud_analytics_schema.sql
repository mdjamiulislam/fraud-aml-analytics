-- ============================================================
-- PROJECT 5 - STEP 12
-- CREATE POSTGRESQL FRAUD ANALYTICS SCHEMA
-- ============================================================


-- ------------------------------------------------------------
-- 1. Create analytical schema
-- ------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS fraud_analytics;


-- ------------------------------------------------------------
-- 2. Main scored transaction fact table
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.fact_scored_transactions CASCADE;

CREATE TABLE fraud_analytics.fact_scored_transactions (

    transaction_id BIGINT PRIMARY KEY,

    step SMALLINT NOT NULL,

    transaction_day SMALLINT NOT NULL,

    hour_of_day SMALLINT NOT NULL,

    transaction_type VARCHAR(20) NOT NULL,

    amount NUMERIC(20,2) NOT NULL,

    origin_account VARCHAR(30) NOT NULL,

    destination_account VARCHAR(30) NOT NULL,

    oldbalance_org NUMERIC(20,2) NOT NULL,

    oldbalance_dest NUMERIC(20,2) NOT NULL,

    high_value_200k_flag SMALLINT NOT NULL,

    hgb_fraud_score DOUBLE PRECISION NOT NULL,

    risk_tier VARCHAR(10) NOT NULL,

    operational_alert SMALLINT NOT NULL,

    legacy_rule_flag SMALLINT NOT NULL,

    actual_fraud SMALLINT NOT NULL,

    CONSTRAINT chk_high_value_flag
        CHECK (high_value_200k_flag IN (0,1)),

    CONSTRAINT chk_operational_alert
        CHECK (operational_alert IN (0,1)),

    CONSTRAINT chk_legacy_rule
        CHECK (legacy_rule_flag IN (0,1)),

    CONSTRAINT chk_actual_fraud
        CHECK (actual_fraud IN (0,1)),

    CONSTRAINT chk_risk_tier
        CHECK (
            risk_tier IN (
                'Low',
                'Medium',
                'High',
                'Critical'
            )
        )
);


-- ------------------------------------------------------------
-- 3. Investigator-facing alert queue
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.investigator_alert_queue CASCADE;

CREATE TABLE fraud_analytics.investigator_alert_queue (

    investigation_rank INTEGER PRIMARY KEY,

    transaction_id BIGINT NOT NULL,

    step SMALLINT NOT NULL,

    transaction_day SMALLINT NOT NULL,

    hour_of_day SMALLINT NOT NULL,

    transaction_type VARCHAR(20) NOT NULL,

    amount NUMERIC(20,2) NOT NULL,

    origin_account VARCHAR(30) NOT NULL,

    destination_account VARCHAR(30) NOT NULL,

    oldbalance_org NUMERIC(20,2) NOT NULL,

    oldbalance_dest NUMERIC(20,2) NOT NULL,

    hgb_fraud_score DOUBLE PRECISION NOT NULL,

    risk_tier VARCHAR(10) NOT NULL,

    priority_band VARCHAR(10) NOT NULL,

    high_value_200k_flag SMALLINT NOT NULL,

    legacy_rule_flag SMALLINT NOT NULL
);


-- ------------------------------------------------------------
-- 4. Risk-tier thresholds
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.risk_tier_thresholds CASCADE;

CREATE TABLE fraud_analytics.risk_tier_thresholds (

    threshold_type VARCHAR(100) PRIMARY KEY,

    source VARCHAR(150),

    score_threshold DOUBLE PRECISION NOT NULL
);


-- ------------------------------------------------------------
-- 5. Risk-tier summary
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.risk_tier_summary CASCADE;

CREATE TABLE fraud_analytics.risk_tier_summary (

    risk_tier VARCHAR(10) PRIMARY KEY,

    transaction_count BIGINT NOT NULL,

    population_pct DOUBLE PRECISION,

    average_model_score DOUBLE PRECISION,

    transaction_value NUMERIC(24,2),

    fraud_count BIGINT,

    fraud_rate_pct DOUBLE PRECISION,

    fraud_capture_pct DOUBLE PRECISION,

    fraud_value NUMERIC(24,2),

    fraud_value_capture_pct DOUBLE PRECISION,

    operational_alerts BIGINT
);


-- ------------------------------------------------------------
-- 6. Operational alert summary
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.operational_alert_summary CASCADE;

CREATE TABLE fraud_analytics.operational_alert_summary (

    metric VARCHAR(150) PRIMARY KEY,

    value DOUBLE PRECISION
);


-- ------------------------------------------------------------
-- 7. Investigation capacity
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.investigation_capacity_summary CASCADE;

CREATE TABLE fraud_analytics.investigation_capacity_summary (

    investigation_capacity INTEGER PRIMARY KEY,

    alerts_reviewed INTEGER,

    fraud_found INTEGER,

    precision_pct DOUBLE PRECISION,

    total_test_fraud_captured_pct DOUBLE PRECISION
);


-- ------------------------------------------------------------
-- 8. Model comparison
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.model_comparison CASCADE;

CREATE TABLE fraud_analytics.model_comparison (

    model VARCHAR(120) PRIMARY KEY,

    threshold DOUBLE PRECISION,

    roc_auc DOUBLE PRECISION,

    average_precision DOUBLE PRECISION,

    accuracy DOUBLE PRECISION,

    precision DOUBLE PRECISION,

    recall DOUBLE PRECISION,

    f1 DOUBLE PRECISION,

    f2 DOUBLE PRECISION,

    true_negative BIGINT,

    false_positive BIGINT,

    false_negative BIGINT,

    true_positive BIGINT,

    alerts BIGINT,

    alert_rate_pct DOUBLE PRECISION,

    false_positive_rate_pct DOUBLE PRECISION
);


-- ------------------------------------------------------------
-- 9. HGB feature importance
-- ------------------------------------------------------------

DROP TABLE IF EXISTS fraud_analytics.hgb_feature_importance CASCADE;

CREATE TABLE fraud_analytics.hgb_feature_importance (

    feature VARCHAR(100) PRIMARY KEY,

    importance_mean DOUBLE PRECISION,

    importance_std DOUBLE PRECISION
);


-- ------------------------------------------------------------
-- 10. Indexes for Power BI / analytical queries
-- ------------------------------------------------------------

CREATE INDEX idx_scored_risk_tier
ON fraud_analytics.fact_scored_transactions (
    risk_tier
);


CREATE INDEX idx_scored_operational_alert
ON fraud_analytics.fact_scored_transactions (
    operational_alert
);


CREATE INDEX idx_scored_actual_fraud
ON fraud_analytics.fact_scored_transactions (
    actual_fraud
);


CREATE INDEX idx_scored_transaction_type
ON fraud_analytics.fact_scored_transactions (
    transaction_type
);


CREATE INDEX idx_scored_step
ON fraud_analytics.fact_scored_transactions (
    step
);


CREATE INDEX idx_scored_day_hour
ON fraud_analytics.fact_scored_transactions (
    transaction_day,
    hour_of_day
);


CREATE INDEX idx_scored_score
ON fraud_analytics.fact_scored_transactions (
    hgb_fraud_score DESC
);


CREATE INDEX idx_queue_score
ON fraud_analytics.investigator_alert_queue (
    hgb_fraud_score DESC
);


-- ------------------------------------------------------------
-- 11. Confirm objects
-- ------------------------------------------------------------

SELECT
    schemaname,
    tablename
FROM pg_tables
WHERE schemaname = 'fraud_analytics'
ORDER BY tablename;