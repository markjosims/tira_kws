"""
Build HF dataset of keywords for KWS evaluation.
Segment keywords from sentence audio using timestamps from MFA output.

This script adds the `keyword_idcs` key to the existing KEYWORD_LIST JSON file.
See JSON structure in `keyword_list_builder.py`.
"""

import argparse
import json