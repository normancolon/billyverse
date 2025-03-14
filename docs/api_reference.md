# BillieVerse API Reference

## 📚 Table of Contents
- [Authentication](#authentication)
- [Market Data API](#market-data-api)
- [Trading API](#trading-api)
- [Portfolio API](#portfolio-api)
- [Analytics API](#analytics-api)
- [WebSocket API](#websocket-api)
- [Rate Limits](#rate-limits)
- [Error Handling](#error-handling)

## 🔐 Authentication

### API Key Authentication
```http
Authorization: Bearer <your_api_key>
```

### Example Request
```python
import requests

headers = {
    'Authorization': 'Bearer your_api_key_here',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.billieverse.com/v1/market/prices', headers=headers)
```

## 📊 Market Data API

### Get Current Prices
```http
GET /api/v1/market/prices
```

Query Parameters:
- `pairs` (required): Comma-separated list of trading pairs
- `timeframe` (optional): Time interval (1m, 5m, 15m, 1h, 4h, 1d)
- `limit` (optional): Number of candles to return (default: 100)

Response:
```json
{
    "status": "success",
    "timestamp": 1634567890,
    "data": {
        "ETH/USDC": {
            "price": 1850.45,
            "volume_24h": 1250000.0,
            "change_24h": 2.5
        },
        "ETH/USDT": {
            "price": 1851.20,
            "volume_24h": 980000.0,
            "change_24h": 2.4
        }
    }
}
```

### Get Historical Data
```http
GET /api/v1/market/history
```

Query Parameters:
- `pair` (required): Trading pair
- `start_time` (required): Start timestamp
- `end_time` (required): End timestamp
- `interval` (optional): Candle interval (default: 1h)

Response:
```json
{
    "status": "success",
    "data": [
        {
            "timestamp": 1634567890,
            "open": 1850.45,
            "high": 1855.20,
            "low": 1848.30,
            "close": 1852.15,
            "volume": 125.5
        }
    ]
}
```

## 💹 Trading API

### Place New Order
```http
POST /api/v1/trading/order
```

Request Body:
```json
{
    "pair": "ETH/USDC",
    "side": "BUY",
    "type": "LIMIT",
    "amount": "1.0",
    "price": "1850.0",
    "time_in_force": "GTC"
}
```

Response:
```json
{
    "status": "success",
    "data": {
        "order_id": "0x123...",
        "status": "PENDING",
        "timestamp": 1634567890,
        "fills": []
    }
}
```

### Cancel Order
```http
DELETE /api/v1/trading/order/{order_id}
```

Response:
```json
{
    "status": "success",
    "data": {
        "order_id": "0x123...",
        "status": "CANCELLED"
    }
}
```

### Get Order Status
```http
GET /api/v1/trading/order/{order_id}
```

Response:
```json
{
    "status": "success",
    "data": {
        "order_id": "0x123...",
        "pair": "ETH/USDC",
        "side": "BUY",
        "type": "LIMIT",
        "amount": "1.0",
        "filled_amount": "0.5",
        "price": "1850.0",
        "status": "PARTIALLY_FILLED",
        "timestamp": 1634567890
    }
}
```

## 📈 Portfolio API

### Get Portfolio Balance
```http
GET /api/v1/portfolio/balance
```

Response:
```json
{
    "status": "success",
    "timestamp": 1634567890,
    "data": {
        "total_value_usd": 125000.50,
        "assets": {
            "ETH": {
                "amount": 10.5,
                "value_usd": 19429.72
            },
            "USDC": {
                "amount": 50000.0,
                "value_usd": 50000.0
            }
        }
    }
}
```

### Get Trading History
```http
GET /api/v1/portfolio/history
```

Query Parameters:
- `start_time` (optional): Start timestamp
- `end_time` (optional): End timestamp
- `limit` (optional): Number of trades (default: 100)

Response:
```json
{
    "status": "success",
    "data": {
        "trades": [
            {
                "trade_id": "0x456...",
                "order_id": "0x123...",
                "pair": "ETH/USDC",
                "side": "BUY",
                "amount": "1.0",
                "price": "1850.0",
                "timestamp": 1634567890,
                "gas_used": 150000
            }
        ]
    }
}
```

## 📊 Analytics API

### Get Performance Metrics
```http
GET /api/v1/analytics/performance
```

Query Parameters:
- `timeframe` (optional): Analysis period (1d, 7d, 30d, YTD)

Response:
```json
{
    "status": "success",
    "data": {
        "total_profit_usd": 15250.45,
        "roi_percent": 12.5,
        "sharpe_ratio": 2.1,
        "max_drawdown": 5.2,
        "win_rate": 0.68,
        "trade_count": 145
    }
}
```

### Get Risk Metrics
```http
GET /api/v1/analytics/risk
```

Response:
```json
{
    "status": "success",
    "data": {
        "var_95": 12500.0,
        "expected_shortfall": 15000.0,
        "portfolio_beta": 1.2,
        "correlation_matrix": {
            "ETH": {
                "USDC": -0.2,
                "USDT": -0.2
            }
        }
    }
}
```

## 🔌 WebSocket API

### Connection
```javascript
ws = new WebSocket('wss://api.billieverse.com/ws')
```

### Subscribe to Market Updates
```javascript
// Subscribe message
{
    "op": "subscribe",
    "channel": "market",
    "pairs": ["ETH/USDC", "ETH/USDT"]
}

// Market update message
{
    "channel": "market",
    "data": {
        "pair": "ETH/USDC",
        "price": 1850.45,
        "volume": 125.5,
        "timestamp": 1634567890
    }
}
```

### Subscribe to Order Updates
```javascript
// Subscribe message
{
    "op": "subscribe",
    "channel": "orders",
    "order_id": "0x123..."
}

// Order update message
{
    "channel": "orders",
    "data": {
        "order_id": "0x123...",
        "status": "FILLED",
        "filled_amount": "1.0",
        "timestamp": 1634567890
    }
}
```

## ⚡ Rate Limits

| Endpoint Category | Rate Limit |
|------------------|------------|
| Market Data      | 100/min    |
| Trading          | 50/min     |
| Portfolio        | 30/min     |
| Analytics        | 20/min     |

Rate limit headers in response:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1634567890
```

## ❌ Error Handling

### Error Response Format
```json
{
    "status": "error",
    "error": {
        "code": "INSUFFICIENT_FUNDS",
        "message": "Insufficient funds for trade",
        "details": {
            "required": "2.0 ETH",
            "available": "1.5 ETH"
        }
    }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| INVALID_API_KEY | Invalid or expired API key |
| INSUFFICIENT_FUNDS | Insufficient funds for trade |
| INVALID_PARAMS | Invalid request parameters |
| ORDER_NOT_FOUND | Order ID not found |
| RATE_LIMIT_EXCEEDED | Rate limit exceeded |
| MARKET_CLOSED | Market is currently closed |
| INVALID_SIGNATURE | Invalid request signature |
| SYSTEM_UNAVAILABLE | System temporarily unavailable |

### Example Error Handling
```python
try:
    response = requests.post(
        'https://api.billieverse.com/v1/trading/order',
        headers=headers,
        json=order_data
    )
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    error_data = e.response.json()
    print(f"Error: {error_data['error']['message']}")
```

Would you like me to add more details to any section or include additional API endpoints? 