#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import_list = dir()
if ("AverageMeter" not in import_list) and ("ProgressMeter" not in import_list) and ("RecorderMeter1" not in import_list) and ("RecorderMeter" not in import_list):
    from api.src.main_obj import AverageMeter, ProgressMeter, RecorderMeter1, RecorderMeter
    
from django.core.management.commands.runserver import Command as Runserver

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'imood_scanner.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
