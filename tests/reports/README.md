# Test Reports

This directory contains generated test reports and coverage analysis.

## Generated Files (Auto-created)
- `coverage_report.html` - Test coverage analysis
- `performance_report.json` - Performance benchmarks  
- `accuracy_report.json` - Model accuracy results

Run tests to generate reports:
```bash
python -m pytest tests/ --cov=. --cov-report=html
```