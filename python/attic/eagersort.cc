#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>


// int draw_index(int n) {
//     // Draw a random integer from 1 to n-1 with p ~ i (n - i)
//     // but subtract 1 to be 0-based
//     int q = n * (n * n - 1);
//     double u = (rand()  / (1.0 + RAND_MAX)) * q;
//     double s = (3 - 4 * n + n * n + u) / (n * n + 2 *n - 3.0);
//     while (true) {
//         int lo = floor(s);
//         if ((lo * (1 + lo) * (3 * n - 2 * lo - 1) <= u)  and
//             ((1 + lo) * (2 + lo) * (3 * n - 2 * lo - 3) > u)) {
//             return lo;
//         }
//         // Newton's update
//         s = (s * s * (3 - 3 * n + 4 * s) - u) / (1 + 6 * s * (1 + s) - 3 * n * (1 + 2 * s));
//         s = s < 0 ? 0 : s > n - 1 ? n - 1 : s;
//     }
// }

// TODO: implement perfect sampling
// following https://www.sciencedirect.com/science/article/pii/S0012365X06000033
double sample(int n, int* poset, int* out) {
    // First, find a linear extension of the poset by doing a topological sort
    // of the poset.

    // copy the poset using memcpy
    int* poset_copy = new int[n * n];
    memcpy(poset_copy, poset, n * n * sizeof(int));

    // topological sort
    int k = 0 ;
    while(k < n) {
        for(int i = 0; i < n; ++i) {
            bool no_predecessors = true;
            for(int j = 0; j < n; ++j) {
                if (poset_copy[j * n + i]) {
                    no_predecessors = false;
                    break;
                }
            }
            if (no_predecessors) {
                out[k++] = i;
                for(int j = 0; j < n; ++j) {
                    poset_copy[i * n + j] = 0;
                }
            }
        }
    }

    // count inversions
    int inversions = 0;
    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            if (out[i] > out[j])
                inversions++;
        }
    }
    double expected_inversions = inversions;


    int burn_iterations = n * n * n * log(n) * 500;
    int total_iterations = burn_iterations * 10;

    // sample
    for(int w = 0; w < total_iterations; ++w) {
        int p = rand() % (n - 1);
        int c = rand() % 2;
        if (c == 1 && poset[out[p] * n + out[p+1]] == 0) {
            int tmp = out[p];
            out[p] = out[p+1];
            out[p+1] = tmp;
            if (out[p] > out[p+1]) {
                inversions++;
            } else {
                inversions--;
            }
        }
        if (w > burn_iterations)
            expected_inversions += inversions;
    }

    expected_inversions /= (total_iterations - burn_iterations);
    return expected_inversions;
}

int main(int argc, char** argv) {
    int n = 10;
    int* poset = new int[n * n];
    int* out = new int[n];
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            poset[i * n + j] = 0;
        }
    }
    poset[0 * n + 1] = 1; // s[0] < s[1]
    poset[0 * n + 2] = 1; // s[0] < s[2]

    int* s = new int[n];
    for(int w = 0; w < 1000; w++) {

        // sample a random permutation via Fisher-Yates
        for(int i = 0; i < n ; ++i) {
            s[i] = i;
        }
        for(int i = 0; i < n - 1; ++i) {
            int j = i + rand() % (n - i);
            int tmp = s[i];
            s[i] = s[j];
            s[j] = tmp;
        }
        double einv = n * (n - 1) / 4.0 ;
        while (einv > 0.1) {
            for(int i = 0; i < n; ++i) {
                for(int j = i + 1; j < n; ++j) {
                    // what happens if I compare s[i] and s[j]?
                    poset[s[i] * n + s[j]] = 1;
                    double einv_gt = sample(n, poset, out);
                    poset[s[i] * n + s[j]] = 0;
                    poset[s[j] * n + s[i]] = 1;
                    double einv_lt = sample(n, poset, out);

                }
            }
        }


    }
}