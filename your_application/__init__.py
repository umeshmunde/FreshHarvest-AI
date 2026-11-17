"""
your_application package shim
"""
import sys
import types
if 'torchvision._meta_registrations' not in sys.modules:
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

from app import app

wsgi = app
application = app
