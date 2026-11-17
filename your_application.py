"""
WSGI shim to ensure compatibility with Render default start command:
gunicorn your_application.wsgi
"""
import sys
import types
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

from app import app

# Export as both 'wsgi' and 'application' for any WSGI server
wsgi = app
application = app

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
