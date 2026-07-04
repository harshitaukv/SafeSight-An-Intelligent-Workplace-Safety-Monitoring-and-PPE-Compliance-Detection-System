def check_violations(worker_data):
    violations = []

    for worker in worker_data:
        if not worker["helmet"]:
            violations.append(f"Worker {worker['worker_id']} missing helmet")

        if not worker["gloves"]:
            violations.append(f"Worker {worker['worker_id']} missing gloves")

        if not worker["vest"]:
            violations.append(f"Worker {worker['worker_id']} missing vest")

        if not worker["boots"]:
            violations.append(f"Worker {worker['worker_id']} missing boots")

        if not worker["goggles"]:
            violations.append(f"Worker {worker['worker_id']} missing goggles")

    if not violations:
        return ["No violations"]

    return violations