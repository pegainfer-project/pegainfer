"""Campaign planning and collection tools for simulator calibration.

The directory is intentionally a script package rather than a runtime
dependency.  The modules use only the Python standard library so a calibration
environment does not need to modify the Rust workspace environment.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

