import numpy as np
import matplotlib.pyplot as plt


times = np.array([
    0.08,0.1,0.15,0.34,0.54,0.94,1.27,1.31,1.93,2.31,2.34,3.1,
    3.61,3.67,4.07,4.65,7.76,8.31,9.37,9.51,10.5,11.12,14.39,14.73,25.19,
    30.12,31.76,32.02,33.27,33.58,36.62,36.88,36.91,37.11,37.15,37.92,
    38.55,39.21,39.55,39.77,39.98,
    0.15,0.62,1.22,1.39,1.61,1.97,3.2,3.38,3.91,4.02,4.2,4.25,4.31,
    4.33,5.25,5.47,5.71,5.79,5.93,6.53,6.79,7.86,9.06,10.35,10.36,11.59,
    17.19,25.78,27.84,27.88,31.01,33.65,34.88,35.02,35.16,35.17,35.5,36,
    36.83,37.5,37.76,38.05,38.49
])

T = 40.0


split = np.where(np.diff(times) < 0)[0][0] + 1
day1 = np.sort(times[:split])
day2 = np.sort(times[split:])
days = [day1, day2]

print("Day 1 arrivals:", len(day1))
print("Day 2 arrivals:", len(day2))


grid = np.unique(np.concatenate(([0.0, T], day1, day2)))
grid.sort()

Lambda_vals = []
for t in grid:
    avg_count = np.mean([np.sum(d <= t) for d in days])
    Lambda_vals.append(avg_count)

Lambda_vals = np.array(Lambda_vals)

dt = np.diff(grid)
dLambda = np.diff(Lambda_vals)
slopes = np.divide(dLambda, dt, out=np.zeros_like(dLambda), where=dt > 0)


def Lambda_hat(t):
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)

    for j, x in enumerate(t):
        if x <= grid[0]:
            out[j] = Lambda_vals[0]
        elif x >= grid[-1]:
            out[j] = Lambda_vals[-1]
        else:
            i = np.searchsorted(grid, x) - 1
            out[j] = Lambda_vals[i] + slopes[i] * (x - grid[i])

    return out if out.size > 1 else out[0]


def Lambda_inv(y):
    y = np.asarray(y)
    out = np.zeros_like(y, dtype=float)

    for j, target in enumerate(y):
        if target <= Lambda_vals[0]:
            out[j] = grid[0]
        elif target >= Lambda_vals[-1]:
            out[j] = grid[-1]
        else:
            i = np.searchsorted(Lambda_vals, target, side='right') - 1


            if slopes[i] == 0:
                out[j] = grid[i+1]
            else:
                out[j] = grid[i] + (target - Lambda_vals[i]) / slopes[i]

    return out if out.size > 1 else out[0]


rng = np.random.default_rng(12345)

Lambda_T = Lambda_vals[-1]
M = rng.poisson(Lambda_T)
Y = np.sort(rng.uniform(0, Lambda_T, size=M))
sim_arrivals = Lambda_inv(Y)

print("\nEstimated total mean arrivals over 40 hours =", Lambda_T)
print("Simulated number of arrivals =", M)
print("Simulated arrival times:")
print(np.round(sim_arrivals, 3))


t_plot = np.linspace(0, T, 1000)
plt.figure(figsize=(8, 5))
plt.plot(t_plot, Lambda_hat(t_plot), label='Fitted piecewise-linear $\\Lambda(t)$')
plt.step(grid, Lambda_vals, where='post', alpha=0.6, label='Empirical mean count')
plt.xlabel("Time (hours)")
plt.ylabel("Integrated rate $\\Lambda(t)$")
plt.title("Fitted integrated rate function for the 40-hour sale")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 2))
plt.eventplot(sim_arrivals, orientation='horizontal')
plt.xlim(0, T)
plt.xlabel("Time (hours)")
plt.yticks([])
plt.title("Simulated arrival times for one 40-hour sale")
plt.grid(True, axis='x')
plt.show()