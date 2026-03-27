import numpy as np
import pandas as pd
import heapq

#HELLO TEST TEST TEST
#TEST LINE 2
FILE_PATH = "CallCounts.xls"
SEED = 12345
N_REP = 5000

#Branch test

def estimate_hourly_rates(file_path, sheet_name=0):

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    labels = df.columns.tolist()
    hourly_rates = df.mean(axis=0).to_numpy(dtype=float)
    return df, labels, hourly_rates


def sample_erlang(mean, k, rng):
    return rng.gamma(shape=k, scale=mean / k)


def generate_base_day(hourly_rates, rng, p_financial=0.59):

    arrival_times = []


    for h, lam in enumerate(hourly_rates):
        n_calls = rng.poisson(lam)
        times_in_hour = h * 60 + rng.uniform(0, 60, size=n_calls)
        arrival_times.extend(times_in_hour.tolist())

    arrival_times.sort()

    jobs = []
    for t in arrival_times:
        if rng.random() < p_financial:
            call_type = "financial"
            service_time = sample_erlang(mean=5.0, k=2, rng=rng)
        else:
            call_type = "contact"
            service_time = sample_erlang(mean=5.0, k=3, rng=rng)

        jobs.append((t, call_type, service_time))

    return jobs


def simulate_fcfs_pool(jobs, num_servers):

    server_heap = [0.0] * num_servers
    heapq.heapify(server_heap)

    waits = []
    system_times = []

    for arrival_time, _, service_time in jobs:
        next_available = heapq.heappop(server_heap)
        service_start = max(arrival_time, next_available)
        departure_time = service_start + service_time
        heapq.heappush(server_heap, departure_time)

        wait = service_start - arrival_time
        time_in_system = departure_time - arrival_time

        waits.append(wait)
        system_times.append(time_in_system)

    return np.array(waits), np.array(system_times)


def summarize_performance(waits, system_times):

    return {
        "n_calls": len(waits),
        "avg_wait": float(np.mean(waits)) if len(waits) else 0.0,
        "avg_system_time": float(np.mean(system_times)) if len(system_times) else 0.0,
        "prob_wait_gt_20s": float(np.mean(waits > (20 / 60))) if len(waits) else 0.0,
        "service_level_20s": float(np.mean(waits <= (20 / 60))) if len(waits) else 1.0,
        "max_wait": float(np.max(waits)) if len(waits) else 0.0,
    }



def simulate_current_system(base_jobs, fin_servers=4, contact_servers=3):

    fin_jobs = [job for job in base_jobs if job[1] == "financial"]
    contact_jobs = [job for job in base_jobs if job[1] == "contact"]

    fin_waits, fin_system = simulate_fcfs_pool(fin_jobs, fin_servers)
    con_waits, con_system = simulate_fcfs_pool(contact_jobs, contact_servers)

    waits = np.concatenate([fin_waits, con_waits])
    system_times = np.concatenate([fin_system, con_system])

    return summarize_performance(waits, system_times)


def simulate_cross_trained_system(base_jobs, num_servers):

    cross_jobs = [(t, call_type, 1.05 * service_time) for (t, call_type, service_time) in base_jobs]
    waits, system_times = simulate_fcfs_pool(cross_jobs, num_servers)
    return summarize_performance(waits, system_times)


def average_dicts(dict_list):
    keys = dict_list[0].keys()
    return {k: float(np.mean([d[k] for d in dict_list])) for k in keys}


def run_simulation(hourly_rates, n_rep=5000, seed=12345):
    rng = np.random.default_rng(seed)

    current_results = []
    cross6_results = []
    cross7_results = []
    cross8_results = []

    for _ in range(n_rep):

        base_jobs = generate_base_day(hourly_rates, rng)

        current_results.append(simulate_current_system(base_jobs, fin_servers=4, contact_servers=3))
        cross6_results.append(simulate_cross_trained_system(base_jobs, num_servers=6))
        cross7_results.append(simulate_cross_trained_system(base_jobs, num_servers=7))
        cross8_results.append(simulate_cross_trained_system(base_jobs, num_servers=8))

    summary = pd.DataFrame([
        {"system": "Current: 4 financial + 3 contact", **average_dicts(current_results)},
        {"system": "Cross-trained: 6 agents", **average_dicts(cross6_results)},
        {"system": "Cross-trained: 7 agents", **average_dicts(cross7_results)},
        {"system": "Cross-trained: 8 agents", **average_dicts(cross8_results)},
    ])

    return summary


if __name__ == "__main__":
    df, labels, hourly_rates = estimate_hourly_rates(FILE_PATH)

    rate_table = pd.DataFrame({
        "Hour": labels,
        "Estimated rate (calls/hour)": hourly_rates,
        "Estimated rate (calls/minute)": hourly_rates / 60
    })

    print("\nEstimated NHPP hourly rates:")
    print(rate_table.to_string(index=False))

    print("\nTotal expected calls per day:")
    print(f"{hourly_rates.sum():.4f}")

    summary = run_simulation(hourly_rates, n_rep=N_REP, seed=SEED)

    print("\nSimulation results:")
    print(summary.to_string(index=False))