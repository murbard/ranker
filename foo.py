# convert the following C code to python

import numpy as np
from collections import defaultdict, Counter

# int draw_index(int n) {
#     // Draw a random integer from 1 to n-1 with p ~ i (n - i)
#     // but subtract 1 to be 0-based
#     q = n * (n * n - 1);
#     u = rand() * q / RAND_MAX;
#     int s = (3 - 4 * n + n * n + u) / (n * n + 2 *n - 3);
#     while (True) {
#         int lo = floor(s);
#         int hi = lo + 1;
#         if ((lo * (1 + lo) * (3 * n - 2 * lo - 1) <= u)  and
#             (hi * (1 + hi) * (3 * n - 2 * hi - 1) > u)) {
#             return lo;
#         }
#         s = (s * s * (3 - 3 * n + 4 * s) - u) / (1 + 6 * s * (1 + s) - 3 * n * (1 + 2 * s));
#     }
# }

avg_iter = 0
total_calls = 0

def draw_index(n):
    global total_calls
    global avg_iter
    total_calls += 1
    # Draw a random integer from 1 to n-1 with p ~ i (n - i)
    # but subtract 1 to be 0-based
    q = n * (n * n - 1)
    u = np.random.rand() * q
    # when does (3n-2k-1)k(1+k)/6 == u?
    s = (3 - 4 * n + n * n + u) / (n * n + 2 *n - 3)
    while True:
        lo = np.floor(s)
        hi = lo + 1
        if ((lo * (1 + lo) * (3 * n - 2 * lo - 1) <= u)  and
            (hi * (1 + hi) * (3 * n - 2 * hi - 1) > u)):
            return lo

        avg_iter += 1

        s = (s * s * (3 - 3 * n + 4 * s) - u) / (1  + 6 * s * (1 + s) - 3 * n * (1 + 2 * s))
        if s >= n - 1:
            s = n - 1
        if s <= 0:
            s = 0



n = 1000
N = 1000000
zz = Counter([1 + draw_index(n) for _ in range(N)])
count = Counter(zz)

# check the relative frequencies and do a chi-squared test
# to see if they are close to ~ i (n - i)

f = np.array([i * (n - i) for i in range(1, n)])
f = f / f.sum()
o = np.array([count[i] for i in range(1, n)])
o = o / o.sum()

# chi-squared test
chi = N * ((f - o)**2 / f).sum()

# use scipy to determine the p-value
from scipy.stats import chi2

print(avg_iter / total_calls)

p = 1 - chi2.cdf(chi, n - 2)
# print results
print('chi-squared p-value =', p)
