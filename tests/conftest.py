import os
import pytest
from datetime import datetime
from src.utils.helpers import save_report

def pytest_sessionfinish(session, exitstatus):
    """
    Hook to save test results in JSON format using the project's Dual Persistence Pattern.
    """
    # 1. Identify test type from session arguments
    # session.config.args contains the paths passed to pytest
    test_args = session.config.args
    test_type = "all_tests"
    
    if any("tests/unit" in str(arg) for arg in test_args):
        test_type = "unit_tests"
    elif any("tests/integration" in str(arg) for arg in test_args):
        test_type = "integration_tests"
    
    # 2. Collect statistics and details
    total_tests = session.testscollected
    passed = 0
    failed = 0
    skipped = 0
    results = []

    for item in session.items:
        # We look for the outcomes stored by pytest_runtest_makereport
        # Note: a test has setup, call, teardown. We focus on 'call' for the main outcome.
        rep_call = getattr(item, "rep_call", None)
        rep_setup = getattr(item, "rep_setup", None)
        
        outcome = "unknown"
        duration = 0
        error_msg = None

        if rep_call:
            outcome = rep_call.outcome
            duration = rep_call.duration
            if rep_call.failed:
                error_msg = str(rep_call.longrepr)
        elif rep_setup and rep_setup.failed:
            outcome = "error_in_setup"
            duration = rep_setup.duration
            error_msg = str(rep_setup.longrepr)
        elif rep_setup and rep_setup.skipped:
            outcome = "skipped"
            duration = rep_setup.duration

        # Update stats
        if outcome == "passed": passed += 1
        elif outcome == "failed": failed += 1
        elif outcome == "skipped": skipped += 1
        elif outcome == "error_in_setup": failed += 1 # Counts as failure

        results.append({
            "node_id": item.nodeid,
            "outcome": outcome,
            "duration_secs": round(float(duration), 4),
            "error": error_msg
        })

    report_data = {
        "phase": "testing",
        "test_type": test_type,
        "timestamp": datetime.now().isoformat(),
        "description": f"Automated test execution report for {test_type}",
        "summary": {
            "total_collected": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "exit_status": int(exitstatus)
        },
        "results": results
    }

    # 3. Define output directory and save
    # Projects rules say outputs go to specific folders, but for tests
    # the user explicitly asked for tests/reports
    reports_dir = os.path.join(os.getcwd(), "tests", "reports")
    
    save_report(report_data, reports_dir, test_type)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to make test results available to pytest_sessionfinish.
    """
    outcome = yield
    rep = outcome.get_result()
    # Set a report attribute for each phase ("setup", "call", "teardown")
    setattr(item, "rep_" + rep.when, rep)
