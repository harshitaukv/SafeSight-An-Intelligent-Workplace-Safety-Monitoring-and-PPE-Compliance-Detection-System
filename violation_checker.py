# violation_checker.py

def check_violations(workers):

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

        missing = []

        if not worker["helmet"]:
            missing.append("Helmet")
            summary["helmet"] += 1

        if not worker["vest"]:
            missing.append("Vest")
            summary["vest"] += 1

        if not worker["gloves"]:
            missing.append("Gloves")
            summary["gloves"] += 1

        if not worker["boots"]:
            missing.append("Boots")
            summary["boots"] += 1

        if not worker["goggles"]:
            missing.append("Goggles")
            summary["goggles"] += 1

        if len(missing) == 0:

            worker["status"] = "Safe"
            worker["missing"] = []

            summary["safe_workers"] += 1

        else:

            worker["status"] = "Unsafe"
            worker["missing"] = missing

            summary["unsafe_workers"] += 1

            violations.append(
                {
                    "worker_id": worker["worker_id"],
                    "missing": missing,
                    "status": "Unsafe"
                }
            )

    return {
        "workers": workers,
        "violations": violations,
        "summary": summary
    }