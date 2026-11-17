import os

# Render dynamically assigns $PORT (typically 10000). Default to 10000 if not set.
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Free-tier resource optimization (512MB RAM limit)
workers = 1
threads = 4
worker_class = "gthread"
timeout = 180
keepalive = 5

# Logging to standard output so logs appear directly in Render dashboard
accesslog = "-"
errorlog = "-"
loglevel = "info"
