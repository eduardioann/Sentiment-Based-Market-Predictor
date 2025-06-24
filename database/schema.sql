CREATE TABLE IF NOT EXISTS tesla_predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    current_price FLOAT NOT NULL,
    predicted_5d_change FLOAT,
    predicted_5d_price FLOAT,
    predicted_7d_change FLOAT,
    predicted_7d_price FLOAT,
    direction_5d_prob FLOAT,
    direction_5d VARCHAR(10),
    direction_7d_prob FLOAT,
    direction_7d VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);