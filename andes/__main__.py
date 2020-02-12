#!/usr/bin/env python3

"""
Andes main entry point Redirection to main.py
This makes the package callable with python -m andes
"""

from andes.cli import main

if __name__ == '__main__':
    main()
