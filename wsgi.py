"""
Standard WSGI entry point for Flask production deployment.
"""
import sys
import types
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

from app import app

application = app
wsgi = app

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
