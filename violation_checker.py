# violation_checker.py

def compute_compliance_rate(workers):
    """
    % of required PPE items actually present, given a list of worker dicts
    that already have a "missing" list on them (i.e. after check_violations
    has run). This is the single formula used everywhere a compliance % is
    shown — per-image, per-report, and in the aggregate summary.

    Previously, per-image "compliance_rate" was computed separately as
    safe_workers / total_workers — a *different* formula from the one used
    for the overall summary. For images with a single worker that's a
    strictly binary number (0% or 100%): if that one worker is missing even
    a single item, the image reads as "0% compliant" even if they're
    wearing 4 of 5 required items. Using the same "items present / items
    required" formula everywhere fixes that and keeps every compliance
    number in the app consistent with each other.
    """
    total_required = len(workers) * 5
    if total_required <= 0:
        return 100.0
    total_missing = sum(len(w.get("missing", [])) for w in workers)
    return round(((total_required - total_missing) / total_required) * 100, 2)


# The 5 required PPE items, as (worker-dict key, display label) pairs.
REQUIRED_PPE = (
    ("helmet", "Helmet"),
    ("vest", "Vest"),
    ("gloves", "Gloves"),
    ("boots", "Boots"),
    ("goggles", "Goggles"),
)

# How many of the 5 required items a worker must have on to count as
# "Safe". Set to all 5: any missing item makes the worker Unsafe, which is
# the strict reading of a PPE requirement — partial protection is not
# compliance. Lower it here (e.g. 3) if a site wants a partial-credit rule
# instead; every surface that shows a Safe/Unsafe verdict reads this
# constant through evaluate_worker, so this one line is the whole change.
#
# Note the practical consequence: the detector has to find all five items
# in the same worker, so anything it misses (goggles and gloves are the
# hardest) shows up as a violation. That's the intended behaviour, but it
# does mean detection recall now drives the compliance number directly.
SAFE_MIN_ITEMS_PRESENT = 5  # out of 5 (100%)


def evaluate_worker(worker):
    """
    Fill in one worker dict's "missing" list and Safe/Unsafe "status" from
    its per-item PPE booleans, and return it.

    Split out of check_violations so code that builds worker-shaped dicts
    outside the per-frame pipeline — e.g. the cumulative whole-video rows
    in api.py — reaches its verdict through the same threshold instead of
    re-implementing the rule and drifting from it.
    """
    missing = [label for key, label in REQUIRED_PPE if not worker.get(key)]
    worker["missing"] = missing
    worker["status"] = (
        "Safe"
        if (len(REQUIRED_PPE) - len(missing)) >= SAFE_MIN_ITEMS_PRESENT
        else "Unsafe"
    )
    return worker


def check_violations(workers):
    """
    Check PPE violations for each worker.

    Args:
        workers: List of worker dictionaries from associate_ppe()

    Returns:
        Dictionary with workers, violations, and summary statistics
    """

    violations = []

    summary = {
        "helmet": 0,
        "vest": 0,
        "gloves": 0,
        "boots": 0,
        "goggles": 0,
        "safe_workers": 0,
        "unsafe_workers": 0
    }

    for worker in workers:

        evaluate_worker(worker)

        for key, _label in REQUIRED_PPE:
            if not worker.get(key):
                summary[key] += 1

        if worker["status"] == "Safe":

            summary["safe_workers"] += 1

        else:

            summary["unsafe_workers"] += 1

            violations.append(
                {
                    "worker_id": worker["worker_id"],
                    "missing": worker["missing"],
                    "status": "Unsafe"
                }
            )

    # ✅ DEBUG: Print violation check results
    print("\n===== VIOLATION CHECK =====")
    for worker in workers:
        print(worker)
    print("Summary:", summary)
    print("===========================\n")

    return {
        "workers": workers,
        "violations": violations,
        "summary": summary
    }