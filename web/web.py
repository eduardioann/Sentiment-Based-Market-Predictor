from flask import Flask, render_template_string, url_for
import psycopg2

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sentiment-Based Market Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@800&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='web.png') }}">
    <style>
        body {
            background: linear-gradient(135deg, #232526 0%, #0f2027 100%);
            font-family: 'Roboto', sans-serif;
            min-height: 100vh;
        }
        .header-img {
            max-width: 110px;
            margin-bottom: 0.5em;
            filter: drop-shadow(0 4px 16px #000a);
            border-radius: 1.5em;
        }
        .glass-card {
            background: rgba(255,255,255,0.13);
            border-radius: 1em;
            box-shadow: 0 4px 16px 0 rgba(31,38,135,0.25);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.13);
            margin-bottom: 1.2em;
            font-size: 1.05em;
            transition: box-shadow 0.2s, transform 0.2s;
        }
        .glass-card:hover {
            box-shadow: 0 8px 32px 0 rgba(31,38,135,0.35);
            transform: translateY(-4px) scale(1.02);
        }
        .prediction-title {
            font-size: 1.18em;
            font-weight: bold;
            color: #00c3ff;
            letter-spacing: 1px;
        }
        .mini-ticker {
            font-size: 1.18em;
            font-weight: 700;
            color: #e53935;
            letter-spacing: 2px;
            margin-bottom: 0.2em;
            text-align: center;
        }
        .value-badge {
            font-size: 1.1em;
            font-weight: 500;
            padding: 0.3em 0.7em;
            border-radius: 0.5em;
        }
        .up { background: #43e97b; color: #222; }
        .down { background: #e53935; color: #fff; }
        .neutral { background: #f7971e; color: #fff; }
        .prob-badge { background: #ffe066; color: #222; font-weight: 600; }
        h1 {
            font-family: 'Montserrat', sans-serif;
            font-weight: 800;
            color: #e0e3e8;
            letter-spacing: 1.5px;
            margin-bottom: 0.2em;
            font-size: 2.7em;
            text-shadow: 0 2px 12px #0006;
            text-align: center;
        }
        .byline {
            color: #e0e3e8;
            font-size: 1.15em;
            margin-bottom: 1.2em;
            font-family: 'Montserrat', sans-serif;
            text-align: center;
        }
        .technical-indicators {
            margin: 0 auto 2em auto;
            max-width: 600px;
            padding: 1.2em 1.5em 0.7em 1.5em;
        }
        .technical-indicators h4 {
            color: #00c3ff;
            font-size: 1.18em;
            margin-bottom: 0.7em;
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
        }
        .footer {
            color: #e0e3e8;
            font-size: 1.05em;
            margin-top: 2em;
            text-align: center;
        }
        .footer a { color: #ffe066; text-decoration: underline; font-weight: 600; }
        .footer .footer-link-desc { color: #e0e3e8; font-size: 0.98em; display: block; margin-bottom: 0.2em; }
        .footer .disclaimer { color: #e53935; font-size: 1.01em; margin-top: 1.2em; font-weight: 600; }
        @media (max-width: 900px) {
            .col-md-6 { flex: 0 0 100%; max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container py-3">
        <div class="text-center">
            <img src="{{ url_for('static', filename='web.png') }}" class="header-img" alt="TradingView Candlestick Chart"/>
            <h1><i class="bi bi-bar-chart-line-fill"></i> Sentiment-Based Market Predictor </h1>
            <div class="byline">Next-Gen Deep Learning Ensemble for Financial Prediction</div>
        </div>
        <div class="row justify-content-center">
            {% for pred in predictions %}
            <div class="col-md-6 d-flex align-items-stretch justify-content-center">
                <div class="card glass-card mb-4 w-100">
                    <div class="card-body text-center">
                        <div class="mini-ticker">TSLA</div>
                        <div class="prediction-title mb-2">
                            <i class="bi bi-calendar-event"></i> Prediction for <b>{{ pred.prediction_date }}</b>
                        </div>
                        <div class="mb-2">
                            <span class="badge value-badge neutral">Current Price: ${{ '{:.2f}'.format(pred.current_price) }}</span>
                        </div>
                        <div class="row text-center mb-2 justify-content-center">
                            <div class="col">
                                <div class="mb-1" style="color:#e0e3e8; font-weight:600;"><i class="bi bi-graph-up-arrow"></i> <b>5-Day</b></div>
                                <span class="badge value-badge {% if pred.predicted_5d_change >= 0 %}up{% else %}down{% endif %}" style="display:inline-block; margin-bottom:0.5em;">{{ '{:+.2f}%'.format(pred.predicted_5d_change*100) }}</span>
                                <div class="small" style="color:#e0e3e8; font-weight:600; margin-bottom:0.5em;">Predicted: ${{ '{:.2f}'.format(pred.predicted_5d_price) }}</div>
                                <span class="badge prob-badge" style="display:inline-block; margin-bottom:0.5em;">Probability Up: {{ '{:.1f}%'.format(pred.direction_5d_prob*100) }}</span>
                                <div style="color:#e0e3e8; font-weight:600;">Direction: <b>{{ pred.direction_5d }}</b></div>
                            </div>
                            <div class="col">
                                <div class="mb-1" style="color:#e0e3e8; font-weight:600;"><i class="bi bi-graph-up-arrow"></i> <b>7-Day</b></div>
                                <span class="badge value-badge {% if pred.predicted_7d_change >= 0 %}up{% else %}down{% endif %}" style="display:inline-block; margin-bottom:0.5em;">{{ '{:+.2f}%'.format(pred.predicted_7d_change*100) }}</span>
                                <div class="small" style="color:#e0e3e8; font-weight:600; margin-bottom:0.5em;">Predicted: ${{ '{:.2f}'.format(pred.predicted_7d_price) }}</div>
                                <span class="badge prob-badge" style="display:inline-block; margin-bottom:0.5em;">Probability Up: {{ '{:.1f}%'.format(pred.direction_7d_prob*100) }}</span>
                                <div style="color:#e0e3e8; font-weight:600;">Direction: <b>{{ pred.direction_7d }}</b></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        <div class="technical-indicators glass-card text-center mb-4">
            <h4><i class="bi bi-activity"></i> Technical Indicators (TSLA)</h4>
            <div class="row justify-content-center">
                <div class="col-auto">
                    <div class="glass-card p-3 mb-3">
                        <h5 class="mb-2" style="color:#e0e3e8; font-weight:600;"><i class="bi bi-bar-chart"></i> Volume</h5>
                        <span class="ms-1" style="color:#e53935; font-weight:600;">77.1M <i class="bi bi-arrow-down"></i></span>
                    </div>
                </div>
                <div class="col-auto">
                    <div class="glass-card p-3 mb-3">
                        <h5 class="mb-2" style="color:#e0e3e8; font-weight:600;"><i class="bi bi-graph-up"></i> MACD</h5>
                        <span class="ms-1" style="color:#e53935; font-weight:600;">-3.19</span>
                    </div>
                </div>
                <div class="col-auto">
                    <div class="glass-card p-3 mb-3">
                        <h5 class="mb-2" style="color:#e0e3e8; font-weight:600;"><i class="bi bi-activity"></i> RSI</h5>
                        <span class="ms-1" style="color:#ffe066; font-weight:600;">38.36</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="footer">
            <span class="footer-link-desc">
                For real-time price charts and technical analysis, visit
                <a href="https://www.tradingview.com/" target="_blank">TradingView</a>.
            </span>
            <span class="footer-link-desc">
                For social sentiment and psychology insights, visit
                <a href="https://x.com/" target="_blank">X</a>.
            </span>
            <span class="footer-link-desc">
                For backend, AI model, and data pipeline details, see
                <a href="https://github.com/eduardioann/Sentiment-Based-Market-Predictor" target="_blank">Sentiment Based Market Predictor</a>.
            </span>
            <div class="disclaimer">
                The information presented does not constitute financial advice and is for informational purposes only.
            </div>
            <span style="color:#e0e3e8; font-size:0.98em; display:block; margin-top:0.5em;">
                &copy; 2025 Eduard Ioan Racoare. All rights reserved. Unauthorized use or reproduction is prohibited.
            </span>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def show_predictions():
    conn = psycopg2.connect(
        dbname='sentiment_db',
        user='postgres',
        password='postgres123!',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT prediction_date, current_price, predicted_5d_change, predicted_5d_price,
               predicted_7d_change, predicted_7d_price, direction_5d_prob, direction_5d,
               direction_7d_prob, direction_7d, created_at
        FROM tesla_predictions
        ORDER BY prediction_date DESC, created_at DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    predictions = []
    for row in rows:
        predictions.append({
            'prediction_date': row[0],
            'current_price': row[1],
            'predicted_5d_change': row[2],
            'predicted_5d_price': row[3],
            'predicted_7d_change': row[4],
            'predicted_7d_price': row[5],
            'direction_5d_prob': row[6],
            'direction_5d': row[7],
            'direction_7d_prob': row[8],
            'direction_7d': row[9],
            'created_at': row[10],
        })
    indicators = {
        'volume': '187.74M',
        'macd': '+2.15',
        'rsi': '50.23'
    }
    return render_template_string(HTML, predictions=predictions, indicators=indicators)

if __name__ == '__main__':
    app.run(debug=True)