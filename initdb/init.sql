CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    divorce_probability FLOAT NOT NULL,
    features JSONB NOT NULL,
    endpoint VARCHAR(20) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON prediction_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_request_id ON prediction_logs (request_id);