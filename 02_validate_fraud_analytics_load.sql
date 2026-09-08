-- ============================================================
-- PROJECT 5 - STEP 12
-- VALIDATE POSTGRESQL DATA LOAD
-- ============================================================


-- 1. TABLE ROW COUNTS

SELECT
    'fact_scored_transactions' AS table_name,
    COUNT(*) AS row_count
FROM fraud_analytics.fact_scored_transactions

UNION ALL

SELECT
    'investigator_alert_queue',
    COUNT(*)
FROM fraud_analytics.investigator_alert_queue

UNION ALL

SELECT
    'risk_tier_thresholds',
    COUNT(*)
FROM fraud_analytics.risk_tier_thresholds

UNION ALL

SELECT
    'risk_tier_summary',
    COUNT(*)
FROM fraud_analytics.risk_tier_summary

UNION ALL

SELECT
    'operational_alert_summary',
    COUNT(*)
FROM fraud_analytics.operational_alert_summary

UNION ALL

SELECT
    'investigation_capacity_summary',
    COUNT(*)
FROM fraud_analytics.investigation_capacity_summary

UNION ALL

SELECT
    'model_comparison',
    COUNT(*)
FROM fraud_analytics.model_comparison

UNION ALL

SELECT
    'hgb_feature_importance',
    COUNT(*)
FROM fraud_analytics.hgb_feature_importance

ORDER BY table_name;


-- 2. MAIN FACT TABLE RECONCILIATION

SELECT
    COUNT(*) AS transactions,

    SUM(actual_fraud) AS actual_fraud,

    SUM(operational_alert) AS operational_alerts,

    SUM(
        CASE
            WHEN actual_fraud = 1
             AND operational_alert = 1
            THEN 1
            ELSE 0
        END
    ) AS true_positives,

    SUM(
        CASE
            WHEN actual_fraud = 0
             AND operational_alert = 1
            THEN 1
            ELSE 0
        END
    ) AS false_positives,

    SUM(
        CASE
            WHEN actual_fraud = 1
             AND operational_alert = 0
            THEN 1
            ELSE 0
        END
    ) AS false_negatives

FROM fraud_analytics.fact_scored_transactions;


-- 3. RISK-TIER DISTRIBUTION

SELECT
    risk_tier,

    COUNT(*) AS transaction_count,

    SUM(actual_fraud) AS fraud_count,

    ROUND(
        100.0 * SUM(actual_fraud) / COUNT(*),
        4
    ) AS fraud_rate_pct,

    SUM(operational_alert) AS operational_alerts

FROM fraud_analytics.fact_scored_transactions

GROUP BY risk_tier

ORDER BY
    CASE risk_tier
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Low' THEN 4
    END;


-- 4. INVESTIGATION QUEUE VALIDATION

SELECT
    COUNT(*) AS alert_queue_rows,
    MIN(investigation_rank) AS first_rank,
    MAX(investigation_rank) AS last_rank,
    MIN(hgb_fraud_score) AS minimum_alert_score,
    MAX(hgb_fraud_score) AS maximum_alert_score

FROM fraud_analytics.investigator_alert_queue;


-- 5. MODEL COMPARISON

SELECT
    model,

    ROUND(
        precision::numeric * 100,
        2
    ) AS precision_pct,

    ROUND(
        recall::numeric * 100,
        2
    ) AS recall_pct,

    ROUND(
        f2::numeric,
        4
    ) AS f2,

    true_positive,
    false_positive,
    false_negative,
    alerts

FROM fraud_analytics.model_comparison

ORDER BY recall DESC NULLS LAST;


-- 6. FEATURE IMPORTANCE

SELECT
    feature,

    ROUND(
        importance_mean::numeric,
        4
    ) AS importance_mean

FROM fraud_analytics.hgb_feature_importance

ORDER BY importance_mean DESC;