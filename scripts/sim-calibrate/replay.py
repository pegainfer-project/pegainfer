#!/usr/bin/env python3
"""Read current held-out evidence; execution remains owned by plan_driver."""
from __future__ import annotations

import json
from pathlib import Path

PURE_COVERAGE_KINDS = ('cold_prefill', 'steady_decode')


class ReplayError(ValueError):
    """A replay cannot establish complete campaign evidence."""


def replay_campaign(root: Path) -> dict:
    from report import report_campaign
    return report_campaign(root)


def audit_publication_coverage(cells: list[dict], *, required_splits=('sealed_test',),
                               required_kinds=PURE_COVERAGE_KINDS) -> dict:
    """Retain the publication audit API without interpreting legacy measurements."""
    scoped = [cell for cell in cells if cell.get('split') in required_splits
              and cell.get('kind') in required_kinds]
    failures = []
    counts = dict(cells=len(scoped), in_domain=0, out_of_domain=0, unsupported=0,
                  sim_failed=0, pairing_failed=0)
    identities = [cell.get('cell_id') for cell in scoped]
    if len(set(identities)) != len(identities):
        failures.append({'reason': 'duplicate publication cell IDs'})
    for cell in scoped:
        cell_id = cell.get('cell_id')
        coverage = cell.get('coverage') or {}
        status = coverage.get('status')
        if status in ('in_domain', 'out_of_domain', 'unsupported'):
            counts[status] += 1
        if status != 'in_domain':
            failures.append({'cell_id': cell_id, 'reason': coverage.get('reason') or status or 'missing coverage'})
        for role in ('target', 'zero_cost', 'sim'):
            if (cell.get(role) or {}).get('status') != 'succeeded':
                if role == 'sim':
                    counts['sim_failed'] += 1
                failures.append({'cell_id': cell_id, 'reason': f'{role} replay did not succeed'})
        pairing = cell.get('pairing') or {}
        if pairing.get('target_zero_cost') is not True or pairing.get('target_sim') is not True:
            counts['pairing_failed'] += 1
            failures.append({'cell_id': cell_id, 'reason': 'request attribution pairing failed'})
    missing_splits = [split for split in required_splits if not any(cell.get('split') == split for cell in scoped)]
    failures.extend({'split': split, 'reason': 'no required pure prefill/decode cells'} for split in missing_splits)
    return {'status': 'failed' if failures else 'passed', 'required_splits': list(required_splits),
            'required_kinds': list(required_kinds), 'counts': counts,
            'missing_splits': missing_splits, 'failures': failures}


def require_publication_coverage(report: dict, *, required_splits=('sealed_test',), required_kinds=None) -> dict:
    """Publication also requires the complete, source-verified current report."""
    from report import verify_report
    verify_report(report)
    if report.get('status') != 'succeeded':
        raise ReplayError('publication requires a complete held-out replay report')
    audit = audit_publication_coverage(report['cells'], required_splits=required_splits,
                                       required_kinds=tuple(required_kinds or report.get('coverage_required_kinds')
                                                            or PURE_COVERAGE_KINDS))
    if audit['status'] != 'passed':
        raise ReplayError('publication coverage gate failed: ' + json.dumps(audit, sort_keys=True))
    return audit
