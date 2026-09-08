-- ============================================================
-- PROJECT 5 - STEP 13
-- VALIDATE POWER BI REPORTING VIEWS
-- ============================================================


-- 1. Executive KPIs

SELECT *
FROM fraud_analytics.vw_executive_fraud_kpis;


-- 2. Risk tier analysis

SELECT *
FROM fraud_analytics.vw_risk_tier_analysis
ORDER BY risk_tier_sort;


-- 3. Transaction type

SELECT *
FROM fraud_analytics.vw_transaction_type_risk
ORDER BY fraud_rate_pct DESC;


-- 4. Daily trend

SELECT *
FROM fraud_analytics.vw_daily_fraud_trend
ORDER BY transaction_day;


-- 5. Hourly pattern

SELECT *
FROM fraud_analytics.vw_hourly_fraud_pattern
ORDER BY hour_of_day;


-- 6. Confusion matrix

SELECT *
FROM fraud_analytics.vw_confusion_matrix
ORDER BY outcome_sort;


-- 7. Model performance

SELECT *
FROM fraud_analytics.vw_model_performance;


-- 8. Investigation capacity

SELECT *
FROM fraud_analytics.vw_investigation_capacity;


-- 9. Highest priority alerts

SELECT *
FROM fraud_analytics.vw_investigator_queue
LIMIT 20;


-- 10. Feature importance

SELECT *
FROM fraud_analytics.vw_feature_importance;