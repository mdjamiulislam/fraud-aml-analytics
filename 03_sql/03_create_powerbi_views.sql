-- ============================================================
-- PROJECT 5 - STEP 13
-- POWER BI ANALYTICAL VIEWS
-- ============================================================


-- ============================================================
-- 1. EXECUTIVE FRAUD KPI VIEW
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_executive_fraud_kpis AS

SELECT

    COUNT(*) AS total_transactions,

    ROUND(
        SUM(amount),
        2
    ) AS total_transaction_value,

    SUM(actual_fraud) AS fraud_transactions,

    ROUND(
        SUM(
            CASE
                WHEN actual_fraud = 1
                THEN amount
                ELSE 0
            END
        ),
        2
    ) AS fraud_transaction_value,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS fraud_rate_pct,

    SUM(operational_alert) AS operational_alerts,

    ROUND(
        (
            100.0
            * SUM(operational_alert)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS alert_rate_pct,

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
    ) AS false_negatives,

    ROUND(
        (
            100.0
            * SUM(
                CASE
                    WHEN actual_fraud = 1
                     AND operational_alert = 1
                    THEN 1
                    ELSE 0
                END
            )
            /
            NULLIF(
                SUM(operational_alert),
                0
            )
        )::numeric,
        2
    ) AS model_precision_pct,

    ROUND(
        (
            100.0
            * SUM(
                CASE
                    WHEN actual_fraud = 1
                     AND operational_alert = 1
                    THEN 1
                    ELSE 0
                END
            )
            /
            NULLIF(
                SUM(actual_fraud),
                0
            )
        )::numeric,
        2
    ) AS model_recall_pct,

    SUM(legacy_rule_flag) AS legacy_rule_alerts,

    SUM(
        CASE
            WHEN legacy_rule_flag = 1
             AND actual_fraud = 1
            THEN 1
            ELSE 0
        END
    ) AS legacy_rule_true_positives

FROM fraud_analytics.fact_scored_transactions;

-- ============================================================
-- 2. RISK TIER ANALYSIS
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_risk_tier_analysis AS

SELECT

    risk_tier,

    CASE risk_tier
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Low' THEN 4
    END AS risk_tier_sort,

    COUNT(*) AS transaction_count,

    ROUND(
        (
            100.0
            * COUNT(*)
            /
            SUM(COUNT(*)) OVER ()
        )::numeric,
        4
    ) AS population_pct,

    ROUND(
        AVG(hgb_fraud_score)::numeric,
        6
    ) AS average_model_score,

    ROUND(
        SUM(amount),
        2
    ) AS transaction_value,

    SUM(actual_fraud) AS fraud_count,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS fraud_rate_pct,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            /
            NULLIF(
                SUM(
                    SUM(actual_fraud)
                ) OVER (),
                0
            )
        )::numeric,
        4
    ) AS fraud_capture_pct,

    ROUND(
        SUM(
            CASE
                WHEN actual_fraud = 1
                THEN amount
                ELSE 0
            END
        ),
        2
    ) AS fraud_value,

    ROUND(
        (
            100.0
            *
            SUM(
                CASE
                    WHEN actual_fraud = 1
                    THEN amount
                    ELSE 0
                END
            )
            /
            NULLIF(
                SUM(
                    SUM(
                        CASE
                            WHEN actual_fraud = 1
                            THEN amount
                            ELSE 0
                        END
                    )
                ) OVER (),
                0
            )
        )::numeric,
        4
    ) AS fraud_value_capture_pct,

    SUM(operational_alert) AS operational_alerts

FROM fraud_analytics.fact_scored_transactions

GROUP BY risk_tier;

-- ============================================================
-- 3. TRANSACTION TYPE RISK
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_transaction_type_risk AS

SELECT

    transaction_type,

    COUNT(*) AS transaction_count,

    ROUND(
        SUM(amount),
        2
    ) AS transaction_value,

    ROUND(
        AVG(amount),
        2
    ) AS average_transaction_amount,

    SUM(actual_fraud) AS fraud_count,

    ROUND(
        SUM(
            CASE
                WHEN actual_fraud = 1
                THEN amount
                ELSE 0
            END
        ),
        2
    ) AS fraud_value,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS fraud_rate_pct,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            /
            NULLIF(
                SUM(
                    SUM(actual_fraud)
                ) OVER (),
                0
            )
        )::numeric,
        4
    ) AS fraud_capture_pct,

    SUM(operational_alert) AS operational_alerts,

    ROUND(
        (
            100.0
            * SUM(operational_alert)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS alert_rate_pct,

    ROUND(
        AVG(hgb_fraud_score)::numeric,
        6
    ) AS average_model_score

FROM fraud_analytics.fact_scored_transactions

GROUP BY transaction_type;

-- ============================================================
-- 4. DAILY FRAUD TREND
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_daily_fraud_trend AS

SELECT

    transaction_day,

    COUNT(*) AS transaction_count,

    ROUND(
        SUM(amount),
        2
    ) AS transaction_value,

    SUM(actual_fraud) AS fraud_count,

    ROUND(
        SUM(
            CASE
                WHEN actual_fraud = 1
                THEN amount
                ELSE 0
            END
        ),
        2
    ) AS fraud_value,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS fraud_rate_pct,

    SUM(operational_alert) AS operational_alerts,

    ROUND(
        AVG(hgb_fraud_score)::numeric,
        6
    ) AS average_model_score

FROM fraud_analytics.fact_scored_transactions

GROUP BY transaction_day

ORDER BY transaction_day;

-- ============================================================
-- 5. HOURLY FRAUD PATTERN
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_hourly_fraud_pattern AS

SELECT

    hour_of_day,

    COUNT(*) AS transaction_count,

    ROUND(
        SUM(amount),
        2
    ) AS transaction_value,

    SUM(actual_fraud) AS fraud_count,

    ROUND(
        SUM(
            CASE
                WHEN actual_fraud = 1
                THEN amount
                ELSE 0
            END
        ),
        2
    ) AS fraud_value,

    ROUND(
        (
            100.0
            * SUM(actual_fraud)
            / NULLIF(COUNT(*), 0)
        )::numeric,
        4
    ) AS fraud_rate_pct,

    SUM(operational_alert) AS operational_alerts,

    ROUND(
        AVG(hgb_fraud_score)::numeric,
        6
    ) AS average_model_score

FROM fraud_analytics.fact_scored_transactions

GROUP BY hour_of_day

ORDER BY hour_of_day;

-- ============================================================
-- 6. CHAMPION MODEL CONFUSION MATRIX
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_confusion_matrix AS

SELECT

    'True Positive' AS outcome,

    1 AS outcome_sort,

    COUNT(*) AS transaction_count

FROM fraud_analytics.fact_scored_transactions

WHERE actual_fraud = 1
  AND operational_alert = 1


UNION ALL


SELECT

    'False Positive',

    2,

    COUNT(*)

FROM fraud_analytics.fact_scored_transactions

WHERE actual_fraud = 0
  AND operational_alert = 1


UNION ALL


SELECT

    'False Negative',

    3,

    COUNT(*)

FROM fraud_analytics.fact_scored_transactions

WHERE actual_fraud = 1
  AND operational_alert = 0


UNION ALL


SELECT

    'True Negative',

    4,

    COUNT(*)

FROM fraud_analytics.fact_scored_transactions

WHERE actual_fraud = 0
  AND operational_alert = 0;

 -- ============================================================
-- 7. MODEL PERFORMANCE
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_model_performance AS

SELECT

    model,

    threshold,

    ROUND(
        roc_auc::numeric,
        4
    ) AS roc_auc,

    ROUND(
        average_precision::numeric,
        4
    ) AS average_precision,

    ROUND(
        accuracy::numeric * 100,
        2
    ) AS accuracy_pct,

    ROUND(
        precision::numeric * 100,
        2
    ) AS precision_pct,

    ROUND(
        recall::numeric * 100,
        2
    ) AS recall_pct,

    ROUND(
        f1::numeric,
        4
    ) AS f1,

    ROUND(
        f2::numeric,
        4
    ) AS f2,

    true_negative,

    false_positive,

    false_negative,

    true_positive,

    alerts,

    ROUND(
        alert_rate_pct::numeric,
        4
    ) AS alert_rate_pct,

    ROUND(
        false_positive_rate_pct::numeric,
        4
    ) AS false_positive_rate_pct

FROM fraud_analytics.model_comparison;

-- ============================================================
-- 8. INVESTIGATION CAPACITY ANALYSIS
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_investigation_capacity AS

SELECT

    investigation_capacity,

    alerts_reviewed,

    fraud_found,

    ROUND(
        precision_pct::numeric,
        2
    ) AS precision_pct,

    ROUND(
        total_test_fraud_captured_pct::numeric,
        2
    ) AS fraud_capture_pct

FROM fraud_analytics.investigation_capacity_summary

ORDER BY investigation_capacity;

-- ============================================================
-- 9. INVESTIGATOR ALERT QUEUE
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_investigator_queue AS

SELECT

    investigation_rank,

    transaction_id,

    step,

    transaction_day,

    hour_of_day,

    transaction_type,

    amount,

    origin_account,

    destination_account,

    oldbalance_org,

    oldbalance_dest,

    ROUND(
        hgb_fraud_score::numeric,
        6
    ) AS fraud_risk_score,

    risk_tier,

    priority_band,

    high_value_200k_flag,

    legacy_rule_flag,

    CASE

        WHEN hgb_fraud_score >= 0.99
        THEN 'Immediate Review'

        WHEN hgb_fraud_score >= 0.98
        THEN 'Urgent Review'

        ELSE 'Priority Review'

    END AS recommended_action

FROM fraud_analytics.investigator_alert_queue

ORDER BY investigation_rank;

-- ============================================================
-- 10. MODEL FEATURE IMPORTANCE
-- ============================================================

CREATE OR REPLACE VIEW fraud_analytics.vw_feature_importance AS

SELECT

    feature,

    ROUND(
        importance_mean::numeric,
        4
    ) AS importance_mean,

    ROUND(
        importance_std::numeric,
        4
    ) AS importance_std,

    ROW_NUMBER() OVER (
        ORDER BY importance_mean DESC
    ) AS importance_rank

FROM fraud_analytics.hgb_feature_importance

ORDER BY importance_mean DESC;

-- ============================================================
-- 11. CONFIRM ALL POWER BI VIEWS
-- ============================================================

SELECT

    table_schema,

    table_name

FROM information_schema.views

WHERE table_schema = 'fraud_analytics'

ORDER BY table_name;