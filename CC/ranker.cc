//todo, do as much as possible directly in compressed format

// include boost ublas for matrices and vectors
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/assign/std/vector.hpp>

// boost special functions erf, erfc, polygamma, digamma, lngamma
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>


// boost random
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/gamma_distribution.hpp>


#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iostream>

// boost optional
#include <boost/optional.hpp>

using namespace boost;
using namespace boost::numeric;
using namespace boost::numeric::ublas;
using namespace boost::math;

const double Mα = 1.2;
const double Mβ = 2.0;

// --- correction to the erf approximation of the logistic-normal integral ---
// corr(z) = sum_i CORR_C[i] (1 + CORR_A[i] z^2) exp(-CORR_A[i] z^2) makes approx+corr match
// log(1+e^{-z}) to ~3e-7 (k=5), and each term still integrates against a Gaussian in closed form.
// Switching k is purely a matter of replacing these arrays (k = CORR_K). See python/fit_correction.py.
const int CORR_K = 5;
const double CORR_A[CORR_K] = {0.04802229968, 0.08966847198, 0.1575268951, 0.252698338, 0.4630803829};
const double CORR_C[CORR_K] = {0.0007259888421, 0.00854921256, 0.02779792245, 0.017919263, 0.001535080506};

// Generalized bump integral:  ∫ N(μδ, σδ²) Σ_k (p_k + q_k a_k z²) e^{-a_k z²} dz  and its plain
// derivatives wrt (μδ, σδ).  Used for the erf-approximation correction (p=q=c) and for the tie
// model's premixed atom bumps.  order==0 fills only T.
inline void bump_eval(int K, const double* A, const double* P_, const double* Q_,
                      double μδ, double σδ, int order,
                      double& T, double& tμ, double& tσ, double& tμμ, double& tμσ, double& tσσ) {
    double s = σδ*σδ, μ = μδ;
    T = tμ = tσ = tμμ = tμσ = tσσ = 0.0;
    if (order == 0) {
        for (int k = 0; k < K; k++) {
            double a = A[k], D = 1 + 2*a*s;
            T += exp(-a*μ*μ/D)/sqrt(D)*(P_[k] + Q_[k]*(a*μ*μ/(D*D) + a*s/D));
        }
        return;
    }
    double Tμ = 0, Ts = 0, Tμμ = 0, Tμs = 0, Tss = 0;   // derivatives wrt μ and s = σδ²
    for (int k = 0; k < K; k++) {
        double a = A[k], p = P_[k], q = Q_[k];
        double D = 1 + 2*a*s, E = exp(-a*μ*μ/D), P = E/sqrt(D);
        double B = p + q*(a*μ*μ/(D*D) + a*s/D);
        double Pμ = P*(-2*a*μ/D), R = -a/D + 2*a*a*μ*μ/(D*D), Ps = P*R;
        double Pμμ = P*(4*a*a*μ*μ/(D*D) - 2*a/D), Rs = 2*a*a/(D*D) - 8*a*a*a*μ*μ/(D*D*D), Pss = P*(R*R + Rs);
        double Pμs = Ps*(-2*a*μ/D) + P*4*a*a*μ/(D*D);
        double Bμ = q*2*a*μ/(D*D), Bs = q*(a/(D*D) - 4*a*a*μ*μ/(D*D*D)), Bμμ = q*2*a/(D*D);
        double Bμs = q*(-8*a*a*μ/(D*D*D)), Bss = q*(-4*a*a/(D*D*D) + 24*a*a*a*μ*μ/(D*D*D*D));
        T   += P*B;
        Tμ  += Pμ*B + P*Bμ;
        Ts  += Ps*B + P*Bs;
        Tμμ += Pμμ*B + 2*Pμ*Bμ + P*Bμμ;
        Tμs += Pμs*B + Pμ*Bs + Ps*Bμ + P*Bμs;
        Tss += Pss*B + 2*Ps*Bs + P*Bss;
    }
    tμ = Tμ; tσ = Ts*2*σδ; tμμ = Tμμ; tμσ = Tμs*2*σδ; tσσ = 4*s*Tss + 2*Ts;
}

// Correction to  ∫ N(μδ, σδ²) log(1+e^{-z}) dz  and its derivatives wrt (μδ, σδ), summed over terms.
// Same (μδ, σδ) convention as the erf kernels gμδ/gσδ/hμμδ/hμσδ/hσσδ, so they simply add.
inline void correction(double μδ, double σδ, int order,
                       double& T, double& tμ, double& tσ, double& tμμ, double& tμσ, double& tσσ) {
    bump_eval(CORR_K, CORR_A, CORR_C, CORR_C, μδ, σδ, order, T, tμ, tσ, tμμ, tμσ, tσσ);
}

// F(μδ, σδ) = ∫ N(μδ, σδ²) log(1+e^{-δ}) dδ --- the error-function approximation plus the k-term
// correction --- with PLAIN derivatives.  Single source of truth for the observation kernel,
// consumed by evaluate(), compute_dobs(), build_Hlin() and select_logdomain() (which negate where
// their legacy conventions store -dF/dσδ and -d²F).  order==0 fills only F.
inline void F_kernel(double μδ, double σδ, int order,
                     double& F, double& Fμ, double& Fσ, double& Fμμ, double& Fμσ, double& Fσσ) {
    const double h = sqrt(16.0 / M_PI + 2 * σδ * σδ);
    const double sqrt_π_h = sqrt(M_PI) * h;
    const double μδ_over_h = μδ / h;
    const double e = exp(-μδ_over_h * μδ_over_h);
    double Tc, tμ, tσ, tμμ, tμσ, tσσ;
    correction(μδ, σδ, order, Tc, tμ, tσ, tμμ, tμσ, tσσ);
    F = e * h / (2 * sqrt(M_PI)) - 0.5 * μδ * (1 - erf(μδ_over_h)) + Tc;
    if (order == 0) { Fμ = Fσ = Fμμ = Fμσ = Fσσ = 0.0; return; }
    Fμ = - 0.5 * (1 - erf(μδ_over_h)) + tμ;
    Fσ = e * σδ / sqrt_π_h + tσ;
    Fμμ = e / sqrt_π_h + tμμ;
    Fμσ = - 2 * μδ * σδ * e / (sqrt(M_PI) * h * h * h) + tμσ;
    double sqrt_π_h5 = sqrt_π_h * sqrt_π_h; sqrt_π_h5 *= sqrt_π_h5; sqrt_π_h5 *= sqrt_π_h;
    Fσσ = e * (256 + 4 * M_PI * σδ * σδ * (8 + M_PI * μδ * μδ)) / sqrt_π_h5 + tσσ;
}

// order-0 per-basis-term integrals (for the tie model's CAVI accumulators)
inline void basis_phi(int K, const double* A, double μδ, double σδ, double* φp, double* φq) {
    double s = σδ*σδ, μ = μδ;
    for (int k = 0; k < K; k++) {
        double a = A[k], D = 1 + 2*a*s, E = exp(-a*μ*μ/D)/sqrt(D);
        φp[k] = E;
        φq[k] = E*(a*μ*μ/(D*D) + a*s/D);
    }
}

// boost variate generator
typedef boost::mt19937 RNGType;
typedef boost::random::normal_distribution<> Normal;
typedef boost::random::variate_generator<RNGType&, Normal> NormalGen;

typedef boost::random::gamma_distribution<> Gamma;
typedef boost::random::variate_generator<RNGType&, Gamma> GammaGen;
RNGType rng;

GammaGen gammagen(rng, Gamma(Mα, Mβ));  // 1/gamma ~ InvGamma(Mα, scale=1/Mβ)
NormalGen normalgen(rng, Normal(0, 1));

class Instance {
    public:
        double v;
        vector<double> z;

        Instance(int n) : z(n), v(0) {}
        ~Instance() {}
        Instance(const Instance& other) : z(other.z), v(other.v) {}
        Instance(Instance&& other) : z(std::move(other.z)), v(std::move(other.v)) {}
        Instance& operator=(const Instance& other) {
            z = other.z;
            v = other.v;
            return *this;
        }
        Instance& operator=(Instance&& other) {
            z = std::move(other.z);
            v = std::move(other.v);
            return *this;
        }
        // static method that generates a random instnace
        static Instance random(int n) {
            Instance inst(n);
            // draw a random value for v from an inverse gamma
            // distribution with parameters Mα and scale=1/Mβ
            inst.v = 1.0 / gammagen();
            // draw a random vector z from a normal distribution with mean 0 and variance v
            for (int i = 0; i < n; i++) {
                inst.z(i) = normalgen() * sqrt(inst.v);
            }
            return inst;
        }
};

class Observations {
    public:
        // mapped matrix X: directed win counts.  T: tie counts, stored at (min,max), used only
        // when the model has tie support enabled (empty otherwise -- zero overhead).
        mapped_matrix<int> X;
        mapped_matrix<int> T;
        // optional coordinate matrix that matches X
        optional<coordinate_matrix<int> > Xcoord;

        Observations(int n) : X(n, n), T(n, n) {}
        Observations(int n, const Instance& instance, int count) : X(n,n), T(n,n) {
            X.reserve(count);
            // draw from the binomial distribution with count = count and
            // n(n-1)/2 buckets
            const int m = n*(n-1)/2;
            for(int c = 0 ; c < count ; c++) {
                int i = rand() % n;
                int j = rand() % n;
                while (i == j) {
                    j = rand() % n;
                }
                double p = 1.0 / (1.0 + exp(-instance.z(i)+instance.z(j)));
                double r = (double)rand() / (double)RAND_MAX;
                if (r < p) {
                    X(i,j) += 1;
                } else {
                    X(j,i) += 1;
                }
            }
        }
        // canonical tie insertion: always stores in the upper triangle (reads tolerate either)
        void add_tie(int i, int j) { T(std::min(i, j), std::max(i, j)) += 1; }

        // Davidson generator: win/tie/loss with true tie parameter λ
        Observations(int n, const Instance& instance, int count, double λ) : X(n,n), T(n,n) {
            X.reserve(count);
            for(int c = 0 ; c < count ; c++) {
                int i = rand() % n;
                int j = rand() % n;
                while (i == j) { j = rand() % n; }
                double d = instance.z(i) - instance.z(j);
                double Z = 2*cosh(d/2) + λ;
                double r = (double)rand() / (double)RAND_MAX * Z;
                if (r < exp(d/2)) X(i,j) += 1;
                else if (r < exp(d/2) + λ) add_tie(i, j);
                else X(j,i) += 1;
            }
        }
        ~Observations() {}
        Observations(const Observations& other) : X(other.X), T(other.T) {}
        Observations(Observations&& other) : X(std::move(other.X)), T(std::move(other.T)) {}
        Observations& operator=(const Observations& other) {
            X = other.X;
            T = other.T;
            return *this;
        }
        Observations& operator=(Observations&& other) {
            X = std::move(other.X);
            T = std::move(other.T);
            return *this;
        }

        void computeCoordinateMatrix() {
            Xcoord = optional<coordinate_matrix<int>>(X);
        }
};

static const char* TIE_ATOMS_DEFAULT = "fit_results/tie_atoms_J16.txt";

// --- tie model: Davidson tie parameter λ with a categorical variational posterior over J atoms ---
// Atoms sit at prior quantiles of p0 = λ/(2+λ) (see python/fit_tie_atoms.py), each with prior
// mass 1/J.  Every atom's bump  b_λ(δ) = log(1 + λ/(2cosh(δ/2)))  is represented in a shared
// basis {e^{-a_l δ²}, a_l δ² e^{-a_l δ²}}, so the w-mixture premixes into ONE bump_eval call per
// pair: the per-pair cost is independent of J.
struct TieModel {
    int J = 0, L = 0;
    std::vector<double> a;                    // L shared widths
    std::vector<double> λ, logλ, π, logπ;     // J atoms: location, log, prior mass, its log
    std::vector<double> P, Q;                 // (J × L) coefficients, atom j at [j*L, j*L+L)
    std::vector<double> w;                    // categorical posterior weights
    std::vector<double> pmix, qmix;           // premixed coefficients Σ_j w_j (P_j, Q_j)
    double wlogλ = 0;                         // Σ_j w_j log λ_j
    double KLwπ = 0;                          // Σ_j w_j log(w_j/π_j): the categorical KL term

    static TieModel load(const std::string& path) {
        TieModel t;
        std::string used = path;              // the path actually opened, for honest error messages
        std::ifstream f(path);
        if (!f && path == TIE_ATOMS_DEFAULT) {
            used = "../" + path;              // allow running the binary from CC/ (default file only:
            f.open(used);                     //  never silently substitute an explicit user path)
            if (!f) throw std::runtime_error("cannot open tie atom file " + path + " (also tried ../)");
        }
        if (!f) throw std::runtime_error("cannot open tie atom file " + path);
        f >> t.J >> t.L;
        t.a.resize(t.L);
        for (auto& v : t.a) f >> v;
        t.λ.resize(t.J); t.logλ.resize(t.J); t.π.resize(t.J); t.logπ.resize(t.J);
        t.P.resize((size_t)t.J*t.L); t.Q.resize((size_t)t.J*t.L);
        for (int j = 0; j < t.J; j++) {
            f >> t.λ[j] >> t.π[j];
            t.logλ[j] = log(t.λ[j]);
            t.logπ[j] = log(t.π[j]);
            for (int l = 0; l < t.L; l++) f >> t.P[(size_t)j*t.L+l] >> t.Q[(size_t)j*t.L+l];
        }
        if (!f) throw std::runtime_error("bad tie atom file " + used);
        t.w.assign(t.J, 1.0/t.J);
        t.pmix.resize(t.L); t.qmix.resize(t.L);
        t.premix();
        return t;
    }
    double E_lambda() const {                 // posterior mean of the tie parameter
        double E = 0;
        for (int j = 0; j < J; j++) E += w[j]*λ[j];
        return E;
    }
    void premix() {
        wlogλ = 0; KLwπ = 0;
        for (int j = 0; j < J; j++) {
            wlogλ += w[j]*logλ[j];
            if (w[j] > 0) KLwπ += w[j]*(log(w[j]) - logπ[j]);
        }
        for (int l = 0; l < L; l++) {
            double sp = 0, sq = 0;
            for (int j = 0; j < J; j++) { sp += w[j]*P[(size_t)j*L+l]; sq += w[j]*Q[(size_t)j*L+l]; }
            pmix[l] = sp; qmix[l] = sq;
        }
    }
};



class Hessian {
    public:
        coordinate_matrix<double> obs;
        vector<double> diag;
        double αβ;
        vector<double> μσαβ;

        Hessian(int n) : obs(2 * n, 2 * n), diag(2 * n + 2), αβ(0), μσαβ(2 * n) {}
        ~Hessian() {}
        Hessian(const Hessian& other) : obs(other.obs), diag(other.diag), αβ(other.αβ), μσαβ(other.μσαβ) {}
        Hessian(Hessian&& other) : obs(std::move(other.obs)), diag(std::move(other.diag)), αβ(std::move(other.αβ)), μσαβ(std::move(other.μσαβ)) {}
        Hessian& operator=(const Hessian& other) {
            obs = other.obs;
            diag = other.diag;
            αβ = other.αβ;
            μσαβ = other.μσαβ;
            return *this;
        }
        Hessian& operator=(Hessian&& other) {
            obs = std::move(other.obs);
            diag = std::move(other.diag);
            αβ = std::move(other.αβ);
            μσαβ = std::move(other.μσαβ);
            return *this;
        }

        vector<double> dot(const vector<double>& x) {
            vector<double> y(x.size());
            int n = (x.size() - 2)/2;
            // the dimensions are 2 * n + 2

            // view of diag, x, and out, excluding last two elements, called diag_μσ, x_μσ, and out_μσ
            // uses project

            auto diag_μσ = project(diag, ublas::range(0, 2 * n));
            auto x_μσ = project(x, ublas::range(0, 2 * n));
            auto y_μσ = project(y, ublas::range(0, 2 * n));

            // y_μσ = diag_μσ * x_μσ + μσαβ * (x[2 * n] + x[2 * n + 1])
            y_μσ = element_prod(diag_μσ, x_μσ) + μσαβ * (x[2 * n] + x[2 * n + 1]);

            // y[2 * n] = μσαβ . x_μσ + diag[2 * n] * x[2 * n] + αβ * x[2 * n + 1]
            // y[2 * n + 1] = μσαβ . x_μσ + diag[2 * n + 1] * x[2 * n + 1] + αβ * x[2 * n]
            double μσαβx = inner_prod(μσαβ, x_μσ);
            y[2 * n] = μσαβx + diag[2 * n] * x[2 * n] + αβ * x[2 * n + 1];
            y[2 * n + 1] = μσαβx + αβ * x[2 * n] + diag[2 * n + 1] * x[2 * n + 1];

            // y[:-2] += h.obs @ x[:-2] + h.obs.T @ x[:-2]
            // remember that out is a sparse upper triangular matrix representing a symmetric matrix

            y_μσ += prod(obs, x_μσ);
            // now with the transpose
            y_μσ += prod(trans(obs), x_μσ);
            return y;
        }

        matrix<double> toDense() {
            const int n = (diag.size() - 2) / 2;
            matrix<double> dense(2 * n + 2, 2 * n + 2, 0.0);

            // fill the diagonal
            for (int i = 0; i < 2 * n + 2; i++) {
                dense(i, i) = diag(i);
            }
            // fill the off-diagonal
            for (int i = 0; i < 2 * n ; i++) {
                for (int j = i + 1; j < 2 * n ; j++) {
                    dense(i, j) += obs(i, j);
                    dense(j, i) += obs(i, j);
                }
            }
            // fill in the αβ and βα terms
            dense(2 * n, 2 * n + 1) += αβ;
            dense(2 * n + 1, 2 * n) += αβ;


            // fill in the μσαβ and αβμσ terms
            for (int i = 0; i < 2 * n; i++) {
                dense(i, 2 * n ) += μσαβ(i);
                dense(i, 2 * n + 1) += μσαβ(i);
                dense(2 * n, i) += μσαβ(i);
                dense(2 * n + 1 , i) += μσαβ(i);
            }

            return dense;
        }
        vector<double> cg(const vector<double>& b, const double λ, const double tol, const int max_iter) {
            const int size = b.size();
            vector<double> x(size, 0.0);
            vector<double> Hp(size);
            vector<double> r(b);
            vector<double> z(b);
            z = element_div(z, diag);
            z *= 1.0/(1.0 + λ);
            vector<double> p = vector<double>(z);

            int k = 0;
            while(k < max_iter) {
                Hp = dot(p) + λ * element_prod(diag, p);
                double pkTHpk = inner_prod(p, Hp);
                double αk = inner_prod(r, z) / pkTHpk;
                double rkTzk = inner_prod(r, z);
                x += αk * p;
                r -=  αk * Hp;
                if (norm_2(r) < tol)
                    break;
                z = element_div(r, diag);
                z *= 1.0 / (1.0 + λ);
                double βk = inner_prod(r, z) / rkTzk;
                p = z + βk * p;
                k++;
            }
            return x;
        }
};

class Ranker {
    public:
        int n;
        vector<double> params;
        optional<TieModel> ties;   // engaged => Davidson tie model; empty => plain Bradley-Terry

        Ranker(int n) : n(n), params(2 * n + 2), hessian(), gradient(), val() {}
        Ranker(int n, const vector<double> &params) : n(n), params(params), hessian(), gradient(), val() {}

        ~Ranker() {}
        Ranker(const Ranker& other) : n(other.n), params(other.params), ties(other.ties), hessian(other.hessian), gradient(other.gradient) {}
        Ranker(Ranker&& other) : n(std::move(other.n)), params(std::move(other.params)), ties(std::move(other.ties)), hessian(std::move(other.hessian)), gradient(std::move(other.gradient)) {}
        Ranker& operator=(const Ranker& other) {
            n = other.n;
            params = other.params;
            ties = other.ties;
            hessian = other.hessian;
            gradient = other.gradient;
            return *this;
        }
        Ranker& operator=(Ranker&& other) {
            n = std::move(other.n);
            params = std::move(other.params);
            ties = std::move(other.ties);
            hessian = std::move(other.hessian);
            gradient = std::move(other.gradient);
            return *this;
        }

        // visit each observed unordered pair (a<b) once with counts (n_w, n_l, n_t);
        // tie counts participate only when the tie model is enabled and are accepted in either
        // triangle of T (canonicalised here); diagonal entries are skipped
        template<class F>
        void each_pair(const Observations& obs, F f) const {
            const bool tie_on = bool(ties);
            auto nt = [&](int a, int b) -> int {   // a < b; count ties stored in either triangle
                return tie_on ? obs.T(a, b) + obs.T(b, a) : 0;
            };
            for (auto it = obs.X.begin1(); it != obs.X.end1(); ++it)
                for (auto el = it.begin(); el != it.end(); ++el) {
                    const int i = el.index1(), j = el.index2();
                    if (i == j) continue;
                    if (i < j) f(i, j, *el, obs.X(j, i), nt(i, j));
                    else if (!obs.X.find_element(j, i)) f(j, i, 0, *el, nt(j, i));
                }
            if (tie_on)
                for (auto it = obs.T.begin1(); it != obs.T.end1(); ++it)
                    for (auto el = it.begin(); el != it.end(); ++el) {
                        const int i = el.index1(), j = el.index2();
                        if (i == j) continue;
                        const int a = std::min(i, j), b = std::max(i, j);
                        if (i > j && obs.T.find_element(a, b)) continue;   // counted at the upper entry
                        if (obs.X.find_element(a, b) || obs.X.find_element(b, a)) continue;  // handled above
                        f(a, b, 0, 0, nt(a, b));
                    }
        }
        std::string print_gradient() {
            std::stringstream ss;
            if (!gradient)
                ss << "none";
            else {
                ss << "{";
                for (int i = 0; i < gradient->size(); i++) {
                    ss << (*gradient)[i];
                    if (i < gradient->size() - 1)
                        ss << ", ";
                }
            }
            ss << "}";
            return ss.str();
        }
        std::string print_hessian() {
            std::stringstream ss;
            if (!hessian)
                ss << "none";
            else {
                auto dense = hessian->toDense();
                ss << "{";
                for (int i = 0; i < dense.size1(); i++) {
                    ss << "{";
                    for (int j = 0; j < dense.size2(); j++) {
                        ss << dense(i, j) ;
                        if (j != dense.size2() - 1)
                            ss << ", ";
                    }
                    ss << "}";
                    if (i != dense.size1() - 1)
                        ss << ", ";
                }
                ss << "}";
            }
            return ss.str();
        }

    public:
        optional<double> val;
        optional<vector<double>> gradient;
        optional<Hessian> hessian;

        void evaluate(const Observations& obs, const bool compute_gradient = false, const bool compute_hessian = false) {

            const double ln_α = params(2 * n);
            const double ln_β = params(2 * n + 1);
            const double α = exp(ln_α);
            const double β = exp(ln_β);
            const double αβ = α * β;
            const double psi_α = digamma(α);
            const double psi1_α = trigamma(α);
            const double psi2_α = polygamma(2, α);
            // μ is a vector slice of the virst n elements of params. It's just a view on the object and provides
            // a lightweight proxy, no copy happens. This uses thje boost function "project"
            vector_range<vector<double>> μ = project(params, ublas::range(0, n));
            vector_range<vector<double>> ln_σ = project(params, ublas::range(n, 2 * n));
            vector<double> σ2(n);
            optional<vector_range<vector<double>>> g_μ;
            optional<vector_range<vector<double>>> g_σ;


            if (compute_gradient) {
                // if gradient is null initialize it
                if (!gradient) {
                    gradient = vector<double>(2 * n + 2);
                }
                g_μ = project(*(gradient), ublas::range(0, n));
                g_σ = project(*(gradient), ublas::range(n, 2 * n));
            }
            if (compute_hessian) {
                // if hessian is null initialize it
                if (!hessian) {
                    hessian = Hessian(n);
                }
                // it it's not null, erase the obs part, the rest is just overwritten
                hessian->obs.clear();
            }

            // Compute sum of μi² + σi², and sum of ln(σi)
            double Σ_μ2_σ2 = 0.0;
            double Σ_lnσ = 0.0;
            for(int i = 0; i < n; ++i) {
                double μi = μ(i);
                double lnσi = ln_σ(i);
                double σi2 = exp(2 * lnσi);
                σ2(i) = σi2;
                double μi2 = μi * μi;
                Σ_μ2_σ2 += μi2 + σi2;
                Σ_lnσ += lnσi;
            }

            double v = - α + αβ / Mβ + Mα * log(Mβ) - 0.5 * n * (1 + log(2 * M_PI)) +
             ln_β + lgamma(Mα) - lgamma(α) + (α - Mα) * psi_α - (1 + Mα) * ln_β
             - Σ_lnσ + 0.5 * (αβ * Σ_μ2_σ2  - n * (ln_β + psi_α - log(2 * M_PI)));

             if (compute_gradient) {

                // Compute gradient
                // dv/dα
                (*gradient)(2 * n) = 0.5 * α *( -2 + 2 * β / Mβ - (2 * Mα + n - 2 * α) * psi1_α + β * Σ_μ2_σ2);
                // dv/dβ
                (*gradient)(2 * n + 1) = - Mα - 0.5 * n + αβ / Mβ + 0.5 * αβ * Σ_μ2_σ2;

                // dv/dμi and dv/dlnσi
                // set the first n elements of gh.g to αβ * ranker.μ using daxpy
                *g_μ = αβ * μ;
                // set the elements n to 2n of gh.g to -1 + αβ * σ2
                *g_σ = αβ * σ2;
                for (auto& el : *g_σ) { el -= 1.0; }
             }

             if (compute_hessian) {

                // Compute Hessian
                // d²v/dμi² and d²v/dlnσi²
                // set the first n elements of gh.h.diag to 1 and the next n elements to 2 * σ2
                vector_range<vector<double>> diag_μ = project(hessian->diag, ublas::range(0, n));
                diag_μ = vector<double>(n, 1.0);
                vector_range<vector<double>> diag_σ = project(hessian->diag, ublas::range(n, 2 * n));
                diag_σ = 2.0 * σ2;

                // scale the whole thing by αβ
                hessian->diag *= αβ;

                // d²v/dα²
                hessian->diag(2 * n) =
                    0.5 * α *(-2 + 2 * β / Mβ - (2 * Mα + n - 4 * α) * psi1_α + β * Σ_μ2_σ2 - (2 * Mα + n - 2 * α) * α * psi2_α);

                // d²v/dαdβ
                hessian->αβ = 0.5 * αβ * (2 / Mβ + Σ_μ2_σ2);

                // d²v/dβ²
                hessian->diag(2 * n + 1) = hessian->αβ; // same value as the cross term

                // d²v/d{μ|σ}id{α|β}
                // place vector ranker.μ as the first half of the gh.h.μσαβ vector
                vector_range<vector<double>> αβ_μ = project(hessian->μσαβ, ublas::range(0, n));
                αβ_μ = μ;
                // place vector σ2 as the second half of the gh.h.μσαβ vector
                vector_range<vector<double>> αβ_σ = project(hessian->μσαβ, ublas::range(n, 2 * n));
                αβ_σ = σ2;
                // scale gh.h.μσαβ by αβ
                hessian->μσαβ *= αβ;
            }

            // Now we add observations, visiting each unordered pair (i<j) once.  Davidson form:
            // with counts (n_w, n_l, n_t) and count = n_w+n_l+n_t,
            //   v += count (F(μδ,σδ) + E[b_λ](μδ,σδ)) + (2 n_l + n_t) μδ/2 − n_t Σ_j w_j log λ_j,
            // where E[b_λ] is the premixed tie bump (absent without tie support).  For n_t = 0 this
            // equals the Bradley-Terry  n_w F(μδ) + n_l F(−μδ)  exactly, since F(−μ) = μ + F(μ).
            const bool tie_on = bool(ties);
            const double wlogλ = tie_on ? ties->wlogλ : 0.0;   // loop-invariant
            if (tie_on) v += ties->KLwπ;   // categorical KL: constant during Newton, but needed
                                           // for v to be the true -ELBO and monotone across CAVI
            coordinate_matrix<double> hobs(2 * n, 2 * n, (obs.X.nnz() + obs.T.nnz()) * 6);

            each_pair(obs, [&](int i, int j, int n_w, int n_l, int n_t) {
                    const int count = n_w + n_l + n_t;
                    if (count == 0) return;

                    const double σδ = sqrt(σ2(i) + σ2(j));
                    const double μδ = μ(i) - μ(j);
                    double F, Fμ, Fσ, Fμμ, Fμσ, Fσσ;
                    F_kernel(μδ, σδ, compute_gradient ? 2 : 0, F, Fμ, Fσ, Fμμ, Fμσ, Fσσ);
                    if (tie_on) {   // premixed atom bumps enter exactly like the correction terms
                        double Bt, bμ, bσ, bμμ, bμσ, bσσ;
                        bump_eval(ties->L, ties->a.data(), ties->pmix.data(), ties->qmix.data(), μδ, σδ,
                                  compute_gradient ? 2 : 0, Bt, bμ, bσ, bμμ, bμσ, bσσ);
                        F += Bt; Fμ += bμ; Fσ += bσ; Fμμ += bμμ; Fμσ += bμσ; Fσσ += bσσ;
                    }
                    const double gμδ = Fμ;    // legacy conventions of this loop: gμδ stores +dF/dμδ
                    const double gσδ = - Fσ;  // gσδ stores -dF/dσδ
                    const double lin = 0.5 * (2.0 * n_l + n_t);             // linear-in-μδ coefficient

                    v += count * F + lin * μδ - n_t * wlogλ;

                    if (compute_gradient) {
                        // dv/dμi, dv/dμj
                        (*g_μ)(i) += count * gμδ + lin;
                        (*g_μ)(j) -= count * gμδ + lin;

                        // dv/dlnσi, dv/dlnσj
                        (*g_σ)(i) -= count * (gσδ * σ2(i) / σδ);
                        (*g_σ)(j) -= count * (gσδ * σ2(j) / σδ);
                    }

                    if (compute_hessian) {

                        const double hμμδ = - Fμμ;   // h*δ store the NEGATED second derivatives
                        const double hμσδ = - Fμσ;   // (legacy convention of this loop)
                        const double hσσδ = - Fσσ;

                        // diagonal hessian terms
                        hessian->diag(i) -= count * hμμδ;
                        hessian->diag(j) -= count * hμμδ;
                        hessian->diag(n+i) -= count * σ2(i) / σδ * (gσδ * σ2(j) / (σδ * σδ) + hσσδ * σ2(i) / σδ + gσδ);
                        hessian->diag(n+j) -= count * σ2(j) / σδ * (gσδ * σ2(i) / (σδ * σδ) + hσσδ * σ2(j) / σδ + gσδ);

                        // hμiμj, hσiσj   (i < j always here)
                        hobs.append_element(i, j, count * hμμδ);
                        hobs.append_element(n + i, n + j, - count * (σ2(i) * σ2(j) / (σδ * σδ) * (hσσδ - gσδ / σδ)));

                        // hμiσi, hμjσi
                        double chσ = count * hμσδ * σ2(i) / σδ;
                        hobs.append_element(i, n + i, - chσ);
                        hobs.append_element(j, n + i, chσ);

                        // hμiσj, hμjσj
                        chσ = count * hμσδ * σ2(j) / σδ;
                        hobs.append_element(i, n + j, - chσ);
                        hobs.append_element(j, n + j, chσ);
                    }
            });
            if (compute_hessian)
                hessian->obs = std::move(hobs);   // once, after all observation rows

            if (!val)
                val = optional<double>(v);
            else
                *val = v;
        }

        void newton(const Observations& obs, const double tol = 1e-8, const int max_iter = 1e6, const bool verbose = false) {
            int k = 0;
            double last_val = 0; bool have_last = false;
            while (k < max_iter) {
                evaluate(obs, true, true);
                // stop on small relative change in the objective (matches python) or tiny gradient
                if (have_last && std::fabs(last_val - *val) / std::fabs(*val) < tol) break;
                if (norm_2(*gradient) < tol) break;
                last_val = *val; have_last = true;
                bool positive = true;
                for (int i = 0; i < 2 * n + 2; i++) {
                    if (hessian->diag(i) <= 0) {
                        positive = false;
                        break;
                    }
                }
                if (!positive) {
                    // throw "ruhroh" exception
                    throw std::runtime_error("non positive diagonal");
                }
                double λ = 1e-10;
                Ranker other(n);
                bool converged = false;
                do {
                    // cg (hessian, gradient, dx,  1e-10, tol, 1000 * n );
                    // use the ublas cg solver with diagonal preconditioner
                    auto dx = std::move(hessian->cg(*gradient, λ, tol, 1000 * n));
                    other = std::move(Ranker(n, params - dx));
                    other.ties = ties;              // the trial point keeps the tie model
                    other.evaluate(obs);
                    if ((*other.val) < (*val)) {
                        break;
                    } else {
                        λ *= 10;
                        if (λ > 1e10) { converged = true; break; }  // no decreasing step => at optimum
                    }
                } while (true);
                if (converged) break;
                *this = std::move(other);
                if (verbose) std::cout << "iter: " << k << ", val:" << *val << ", λ: " << λ << std::endl;
                k++;
            }
        }

        // Exact CAVI coordinate update of the tie weights:  w_j ∝ π_j exp(v_j), where
        // v_j = (Σ n_t) log λ_j − Σ_pairs count · E[b_{λ_j}]  (the data term is linear in w).
        // The per-pair basis integrals are accumulated once (L values); v_j is then an L-dot per
        // atom, so the pass is J-free over pairs.  Returns max |Δw_j|.
        double cavi(const Observations& obs) {
            if (!ties) return 0.0;
            TieModel& t = *ties;
            const int J = t.J, L = t.L;
            std::vector<double> s2c(n);                    // hoisted: n exp calls, not 2 per pair
            for (int i = 0; i < n; i++) s2c[i] = exp(2*params(n+i));
            std::vector<double> accp(L, 0.0), accq(L, 0.0), φp(L), φq(L);
            double nt_tot = 0;
            each_pair(obs, [&](int a, int b, int n_w, int n_l, int n_t) {
                const int count = n_w + n_l + n_t;
                if (count == 0) return;
                nt_tot += n_t;
                const double σδ = sqrt(s2c[a] + s2c[b]);
                const double μδ = params(a) - params(b);
                basis_phi(L, t.a.data(), μδ, σδ, φp.data(), φq.data());
                for (int l = 0; l < L; l++) { accp[l] += count*φp[l]; accq[l] += count*φq[l]; }
            });
            std::vector<double> vj(J);
            double mx = -1e300;
            for (int j = 0; j < J; j++) {
                double s = 0;
                for (int l = 0; l < L; l++) s += t.P[(size_t)j*L+l]*accp[l] + t.Q[(size_t)j*L+l]*accq[l];
                vj[j] = nt_tot*t.logλ[j] - s + t.logπ[j];
                mx = std::max(mx, vj[j]);
            }
            double Z = 0, dw = 0;
            for (int j = 0; j < J; j++) { vj[j] = exp(vj[j] - mx); Z += vj[j]; }
            for (int j = 0; j < J; j++) {
                double nw = vj[j]/Z;
                dw = std::max(dw, std::fabs(nw - t.w[j]));
                t.w[j] = nw;
            }
            t.premix();
            return dw;
        }

        void fit(const Observations& obs, const double tol = 1e-8, const int max_iter = 1e6, const bool verbose = false) {
            if (!ties) { newton(obs, tol, max_iter, verbose); return; }
            const double loose = std::max(tol, 1e-4);     // intermediate rounds need not fully converge
            double dw = 1.0;
            for (int r = 0; r < 8 && dw >= 1e-9; r++) {   // alternate Newton and the exact w-update
                newton(obs, loose, max_iter, verbose);
                dw = cavi(obs);
            }
            if (dw >= 1e-9)
                std::cerr << "warning: tie-weight alternation hit its 8-round cap without stationarity\n";
            // always end on a full-tolerance Newton pass, so the returned parameters (and cached
            // val/gradient/hessian) correspond to the stored weights
            newton(obs, tol, max_iter, verbose);
        }
};

// ===================== acquisition strategies (LINEAR domain) =====================
#include <cmath>
#include <fstream>
#include <thread>
#include <chrono>
#include <boost/math/special_functions/polygamma.hpp>

static const double GHx[21] = {-7.8493828951138225, -6.7514447187174609, -5.8293820073044715, -4.9949639447820253, -4.2143439816884216, -3.4698466904753764, -2.7505929810523733, -2.0491024682571628, -1.3597658232112302, -0.67804569244064405, 0, 0.67804569244064405, 1.3597658232112302, 2.0491024682571628, 2.7505929810523733, 3.4698466904753764, 4.2143439816884216, 4.9949639447820253, 5.8293820073044715, 6.7514447187174609, 7.8493828951138225};
static const double GHw[21] = {2.0989912195656709e-14, 4.975368604121718e-11, 1.4506612844930856e-08, 1.2253548361482524e-06, 4.2192347425516618e-05, 0.00070804779548153682, 0.0064396970514087751, 0.03395272978654286, 0.10839228562641948, 0.21533371569505969, 0.27026018357287701, 0.21533371569505969, 0.10839228562641948, 0.03395272978654286, 0.0064396970514087751, 0.00070804779548153682, 4.2192347425516618e-05, 1.2253548361482524e-06, 1.4506612844930856e-08, 4.975368604121718e-11, 2.0989912195656709e-14};

// extract linear params from a log-domain Ranker
static void extract(const ublas::vector<double>& p, int n, std::vector<double>& mu, std::vector<double>& sg, double& al, double& be) {
    mu.resize(n); sg.resize(n);
    for (int i = 0; i < n; i++) { mu[i] = p(i); sg[i] = exp(p(n + i)); }
    al = exp(p(2 * n)); be = exp(p(2 * n + 1));
}

static int actual_inversions(const std::vector<double>& mu, const std::vector<double>& z, int n) {
    int inv = 0;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
        if ((mu[i] - mu[j]) * (z[i] - z[j]) <= 0) inv++;
    return inv;
}

static double prob_win(const std::vector<double>& mu, const std::vector<double>& sg, int i, int j) {
    double m = mu[i] - mu[j];
    double s2 = sg[i] * sg[i] + sg[j] * sg[j];
    return 0.5 * (1.0 + std::erf(sqrt(M_PI) * m / (4.0 * sqrt(1.0 + M_PI * s2 / 8.0))));
}

// expected number of inversions + (optional) gradient wrt (mu,sigma) [FIXED |mu_delta| version]
static double expected_inversions(const std::vector<double>& mu, const std::vector<double>& sg, int n, double* grad) {
    double inv = 0;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        double sd = sqrt(sg[i] * sg[i] + sg[j] * sg[j]);
        double md = mu[i] - mu[j];
        double z = fabs(md) / sd;
        inv += 0.5 * erfc(z / sqrt(2.0));
        if (grad) {
            double dinv_dz = -exp(-z * z / 2.0) / sqrt(2.0 * M_PI);
            double sgn = (md > 0) ? 1.0 : ((md < 0) ? -1.0 : 0.0);
            double dmi = dinv_dz * sgn / sd;
            grad[i] += dmi; grad[j] -= dmi;
            grad[n + i] += -dinv_dz * fabs(md) * sg[i] / (sd * sd * sd);
            grad[n + j] += -dinv_dz * fabs(md) * sg[j] / (sd * sd * sd);
        }
    }
    return inv;
}

static double Hb(double q) { if (q < 1e-12) q = 1e-12; if (q > 1 - 1e-12) q = 1 - 1e-12; return -q * log(q) - (1 - q) * log(1 - q); }
static double eig_pair(double md, double sd) {
    double p = 0, al = 0;
    for (int k = 0; k < 21; k++) { double s = 1.0 / (1.0 + exp(-(md + sd * GHx[k]))); p += GHw[k] * s; al += GHw[k] * Hb(s); }
    return Hb(p) - al;
}

// dobs (dense, n*n*2) from mu,sigma
static void compute_dobs(const std::vector<double>& mu, const std::vector<double>& sg, int n, std::vector<double>& d0, std::vector<double>& d1) {
    d0.assign(n * n, 0); d1.assign(n * n, 0);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        double sd = sqrt(sg[i] * sg[i] + sg[j] * sg[j]);
        double md = mu[i] - mu[j];
        double F, Fμ, Fσ, Fμμ, Fμσ, Fσσ;
        F_kernel(md, sd, 1, F, Fμ, Fσ, Fμμ, Fμσ, Fσσ);
        d0[i * n + j] = Fσ / sd;
        d1[i * n + j] = Fμ;
    }
}

// build dense LINEAR-domain Hessian H (N x N, N=2n+2), porting models.py eval(compute_hessian)
static void build_Hlin(const std::vector<double>& mu, const std::vector<double>& sg, double al, double be,
                       const Observations& obs, int n, std::vector<double>& H) {
    using boost::math::polygamma;
    int N = 2 * n + 2;
    H.assign((size_t)N * N, 0.0);
    auto A = [&](int r, int c) -> double& { return H[(size_t)r * N + c]; };
    double albe = al * be;
    double sum_m2s2 = 0; for (int i = 0; i < n; i++) sum_m2s2 += mu[i] * mu[i] + sg[i] * sg[i];
    // analytic diagonal
    for (int i = 0; i < n; i++) { A(i, i) += albe; A(n + i, n + i) += 1.0 / (sg[i] * sg[i]) + albe; }
    A(2 * n, 2 * n)     += polygamma(1, al) + (1 + al) * polygamma(2, al)   // entropy
                         - (1 + Mα) * polygamma(2, al)                       // cross IG
                         - 0.5 * n * polygamma(2, al);                       // gaussian
    A(2 * n + 1, 2 * n + 1) += -1.0 / (be * be) + (1 + Mα) / (be * be) + n / (2.0 * be * be);
    // alpha-beta coupling
    double hab = 1.0 / Mβ + 0.5 * sum_m2s2;
    A(2 * n, 2 * n + 1) += hab; A(2 * n + 1, 2 * n) += hab;
    // mu/sigma <-> alpha/beta
    for (int i = 0; i < n; i++) {
        A(i, 2 * n) += be * mu[i];     A(2 * n, i) += be * mu[i];
        A(i, 2 * n + 1) += al * mu[i]; A(2 * n + 1, i) += al * mu[i];
        A(n + i, 2 * n) += be * sg[i];     A(2 * n, n + i) += be * sg[i];
        A(n + i, 2 * n + 1) += al * sg[i]; A(2 * n + 1, n + i) += al * sg[i];
    }
    // observation terms
    for (auto it = obs.X.begin1(); it != obs.X.end1(); ++it)
        for (auto el = it.begin(); el != it.end(); ++el) {
            int i = el.index1(), j = el.index2();
            if (i == j) continue;
            double count = *el;
            double sd = sqrt(sg[i] * sg[i] + sg[j] * sg[j]);
            double md = mu[i] - mu[j];
            double F, Fμ, Fσ, Fμμ, Fμσ, Fσσ;
            F_kernel(md, sd, 2, F, Fμ, Fσ, Fμμ, Fμσ, Fσσ);
            double gsd = -Fσ;
            double hmm = -Fμμ;
            double hms = -Fμσ;
            double hss = -Fσσ;
            // diagonal (subtracted)
            A(i, i)         -= count * hmm;
            A(j, j)         -= count * hmm;
            A(n + i, n + i) -= count * (hss * (sg[i] / sd) * (sg[i] / sd) + gsd * sg[j] * sg[j] / (sd * sd * sd));
            A(n + j, n + j) -= count * (hss * (sg[j] / sd) * (sg[j] / sd) + gsd * sg[i] * sg[i] / (sd * sd * sd));
            // off-diagonal entries (each added symmetrically: H[r][c]+=v, H[c][r]+=v)
            int lo = std::min(i, j), hi = std::max(i, j);
            auto sym = [&](int r, int c, double v) { A(r, c) += v; A(c, r) += v; };
            sym(lo, hi, count * hmm);
            sym(n + lo, n + hi, -count * (hss * sg[i] * sg[j] / (sd * sd) - gsd * sg[i] * sg[j] / (sd * sd * sd)));
            sym(i, n + i, -count * hms * sg[i] / sd);
            sym(i, n + j, -count * hms * sg[j] / sd);
            sym(j, n + i,  count * hms * sg[i] / sd);
            sym(j, n + j,  count * hms * sg[j] / sd);
        }
}

// solve H y = b (dense, Gaussian elimination w/ partial pivot); returns y
static std::vector<double> dense_solve(std::vector<double> H, std::vector<double> b, int N) {
    auto A = [&](int r, int c) -> double& { return H[(size_t)r * N + c]; };
    for (int col = 0; col < N; col++) {
        int piv = col; double best = fabs(A(col, col));
        for (int r = col + 1; r < N; r++) if (fabs(A(r, col)) > best) { best = fabs(A(r, col)); piv = r; }
        if (piv != col) { for (int c = 0; c < N; c++) std::swap(A(col, c), A(piv, c)); std::swap(b[col], b[piv]); }
        double d = A(col, col);
        for (int r = 0; r < N; r++) if (r != col) {
            double f = A(r, col) / d;
            if (f != 0.0) { for (int c = col; c < N; c++) A(r, c) -= f * A(col, c); b[r] -= f * b[col]; }
        }
    }
    for (int i = 0; i < N; i++) b[i] /= A(i, i);
    return b;
}

// factor = -H_lin^{-1} df  for a given df (length N)
static std::vector<double> acq_factor(const std::vector<double>& mu, const std::vector<double>& sg, double al, double be,
                                      const Observations& obs, int n, const std::vector<double>& df) {
    int N = 2 * n + 2;
    std::vector<double> H; build_Hlin(mu, sg, al, be, obs, n, H);
    std::vector<double> y = dense_solve(H, df, N);
    for (auto& v : y) v = -v;
    return y;
}

// select pair for var/inv strategies
static std::pair<int,int> select_factor_based(const std::vector<double>& mu, const std::vector<double>& sg, double al, double be,
                                              const Observations& obs, int n, bool inv_target) {
    int N = 2 * n + 2;
    std::vector<double> df(N, 0.0);
    if (inv_target) expected_inversions(mu, sg, n, df.data());
    else for (int i = 0; i < n; i++) df[n + i] = 2.0 * sg[i];
    std::vector<double> factor = acq_factor(mu, sg, al, be, obs, n, df);
    std::vector<double> d0, d1; compute_dobs(mu, sg, n, d0, d1);
    double best = 1e300; std::pair<int,int> bp(0, 1);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        double ifw = (factor[n + i] * sg[i] + factor[n + j] * sg[j]) * d0[i * n + j] + (factor[i] - factor[j]) * d1[i * n + j];
        double ifl = (factor[n + j] * sg[j] + factor[n + i] * sg[i]) * d0[j * n + i] + (factor[j] - factor[i]) * d1[j * n + i];
        double p = prob_win(mu, sg, i, j);
        double g = p * ifw + (1 - p) * ifl;
        if (g < best) { best = g; bp = {i, j}; }
    }
    return bp;
}

static std::pair<int,int> select_eig(const std::vector<double>& mu, const std::vector<double>& sg, int n) {
    double best = -1e300; std::pair<int,int> bp(0, 1);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        double sd = sqrt(sg[i] * sg[i] + sg[j] * sg[j]);
        double e = eig_pair(mu[i] - mu[j], sd);
        if (e > best) { best = e; bp = {i, j}; }
    }
    return bp;
}

// ---- 3-outcome EIG for the tie model ----
// Targets information about the SCORES: λ is marginalised out of the predictive before the
// entropy is taken (Hc stores H3 of the λ-mixed probabilities), so the gain is MI(outcome; δ)
// and no comparisons are spent learning the nuisance λ.  The λ-mixed quantities depend on δ
// only, so they are tabulated once per selection (cost ∝ J, shared across all candidate pairs);
// each candidate then costs 21 interpolated Gauss-Hermite nodes, independent of J.
static inline double xlogx(double p) { return p > 1e-300 ? p*log(p) : 0.0; }
static inline double H3(double pw, double pt) {
    double pl = 1.0 - pw - pt;
    return -(xlogx(pw) + xlogx(pt) + xlogx(pl));
}
struct TieEigTable {
    static const int NPTS = 2501;
    static constexpr double LO = -25.0, STEP = 0.02;
    std::vector<double> Pw, Pt, Hc;
    void build(const TieModel& t) {
        Pw.assign(NPTS, 0); Pt.assign(NPTS, 0); Hc.assign(NPTS, 0);
        for (int k = 0; k < NPTS; k++) {
            double d = LO + STEP*k, A = 2*cosh(d/2), e = exp(d/2);
            double pw = 0, pt = 0;
            for (int j = 0; j < t.J; j++) {
                double Z = A + t.λ[j];
                pw += t.w[j]*e/Z; pt += t.w[j]*t.λ[j]/Z;
            }
            Pw[k] = pw; Pt[k] = pt;
            Hc[k] = H3(pw, pt);   // entropy OF the λ-marginalised predictive: score-targeted MI
        }
    }
    inline void interp(double d, double& pw, double& pt, double& hc) const {
        double x = (d - LO)/STEP;
        if (x <= 0) { pw = Pw[0]; pt = Pt[0]; hc = Hc[0]; return; }
        if (x >= NPTS-1) { pw = Pw[NPTS-1]; pt = Pt[NPTS-1]; hc = Hc[NPTS-1]; return; }
        int k = (int)x; double f = x - k;
        pw = Pw[k]*(1-f) + Pw[k+1]*f; pt = Pt[k]*(1-f) + Pt[k+1]*f; hc = Hc[k]*(1-f) + Hc[k+1]*f;
    }
};
static double eig3_pair(double md, double sd, const TieEigTable& tab) {
    double pw = 0, pt = 0, hc = 0, a, b, c;
    for (int k = 0; k < 21; k++) {
        tab.interp(md + sd*GHx[k], a, b, c);
        pw += GHw[k]*a; pt += GHw[k]*b; hc += GHw[k]*c;
    }
    return H3(pw, pt) - hc;
}
static std::pair<int,int> select_eig_ties(const std::vector<double>& mu, const std::vector<double>& sg,
                                          int n, const TieModel& t) {
    TieEigTable tab; tab.build(t);
    double best = -1e300; std::pair<int,int> bp(0, 1);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        double sd = sqrt(sg[i] * sg[i] + sg[j] * sg[j]);
        double e = eig3_pair(mu[i] - mu[j], sd, tab);
        if (e > best) { best = e; bp = {i, j}; }
    }
    return bp;
}

// ---- LOG-domain native acquisition: reuses the fit's log-domain Hessian (rk.hessian) + cg ----
// gain df/do is coordinate-invariant, so this yields the same selected pair & gain as Python's linear domain.
// Requires rk to be at a fitted optimum (so H_log == S H_lin S on the invariants).
static std::pair<int,int> select_logdomain(Ranker& rk, const Observations& obs, int n, bool inv_target, double* out_gain = nullptr, std::vector<double>* out_factor = nullptr) {
    int N = 2 * n + 2;
    rk.evaluate(obs, true, true);  // populate log-domain gradient + Hessian at current params
    std::vector<double> mu, sg; double al, be; extract(rk.params, n, mu, sg, al, be);
    vector<double> negdf(N); for (int k = 0; k < N; k++) negdf(k) = 0.0;
    if (inv_target) {
        std::vector<double> g(N, 0.0); expected_inversions(mu, sg, n, g.data());
        for (int i = 0; i < n; i++) { negdf(i) = -g[i]; negdf(n + i) = -(sg[i] * g[n + i]); }
    } else {
        for (int i = 0; i < n; i++) negdf(n + i) = -(2.0 * sg[i] * sg[i]);
    }
    vector<double> factor = rk.hessian->cg(negdf, 0.0, 1e-10, 1000 * n);
    if (out_factor) { out_factor->resize(N); for (int k = 0; k < N; k++) (*out_factor)[k] = factor(k); }
    double best = 1e300; std::pair<int,int> bp(0, 1);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
        double sd = sqrt(sg[i] * sg[i] + sg[j] * sg[j]);
        double md = mu[i] - mu[j];
        double Fw, Fμw, Fσw, u3, u4, u5, Fl, Fμl, Fσl;
        F_kernel(md, sd, 1, Fw, Fμw, Fσw, u3, u4, u5);    // win uses μδ=md
        F_kernel(-md, sd, 1, Fl, Fμl, Fσl, u3, u4, u5);   // lose uses μδ=-md
        double gmd  = Fμw;    // win: i beats j
        double gmd2 = Fμl;    // lose: j beats i
        double gsd  = -Fσw;   // Fσ even in md
        double sig_part = -(gsd / sd) * (factor(n + i) * sg[i] * sg[i] + factor(n + j) * sg[j] * sg[j]);
        double ifwin  = (factor(i) - factor(j)) * gmd  + sig_part;
        double iflose = (factor(j) - factor(i)) * gmd2 + sig_part;
        double p = prob_win(mu, sg, i, j);
        double g = p * ifwin + (1 - p) * iflose;
        if (g < best) { best = g; bp = {i, j}; }
    }
    if (out_gain) *out_gain = best;
    return bp;
}

// initial sigma per VBayes.__init__
static double init_sigma() { return sqrt(Mβ / Mα) * (1.0 + 3.0 / (6.0 * Mα + 2.0 * Mα * Mα)); }

// prior-mode starting point (μ=0, σ=init_sigma, α=Mα, β=Mβ), shared by fits that start cold
static ublas::vector<double> init_params(int n) {
    ublas::vector<double> p(2 * n + 2);
    double s0 = init_sigma();
    for (int i = 0; i < n; i++) { p(i) = 0.0; p(n + i) = log(s0); }
    p(2 * n) = log(Mα); p(2 * n + 1) = log(Mβ);
    return p;
}

// ----- verify2: fit the n=6 reference obs, then check the FITTED-state reference (log-native) -----
static int verify2() {
    int n = 6;
    Ranker rk(n, init_params(n));
    Observations obs(n);
    int O[9][3] = {{0,1,3},{1,0,1},{2,3,2},{0,4,2},{4,5,1},{5,2,2},{3,1,1},{2,5,2},{4,0,1}};
    for (auto& o : O) obs.X(o[0], o[1]) = o[2];
    rk.fit(obs);
    std::vector<double> mu, sg; double al, be; extract(rk.params, n, mu, sg, al, be);
    std::vector<double> z = {0.4, -0.6, 0.05, 1.0, -0.3, 0.5};
    std::cout.precision(8); std::cout << std::fixed;
    std::cout << "mu="; for (double v : mu) std::cout << v << " "; std::cout << "\n";
    std::cout << "sigma="; for (double v : sg) std::cout << v << " "; std::cout << "\n";
    std::cout << "alpha=" << al << " beta=" << be << "\n";
    std::cout << "actual_inversions=" << actual_inversions(mu, z, n) << "\n";
    std::cout << "prob_win(0,1)=" << prob_win(mu, sg, 0, 1) << "  prob_win(2,3)=" << prob_win(mu, sg, 2, 3) << "\n";
    std::cout << "expected_inversions=" << expected_inversions(mu, sg, n, nullptr) << "\n";
    std::cout << "EIG(0,1)=" << eig_pair(mu[0]-mu[1], sqrt(sg[0]*sg[0]+sg[1]*sg[1]))
              << "  EIG(2,3)=" << eig_pair(mu[2]-mu[3], sqrt(sg[2]*sg[2]+sg[3]*sg[3]));
    auto ep = select_eig(mu, sg, n); std::cout << "  EIG selected pair=(" << ep.first << "," << ep.second << ")\n";
    for (int tgt = 0; tgt < 2; tgt++) {
        double gain; std::vector<double> fac;
        auto bp = select_logdomain(rk, obs, n, tgt == 1, &gain, &fac);
        std::cout << (tgt ? "inv" : "var") << ": factor_mu[:6]=";
        for (int i = 0; i < n; i++) std::cout << fac[i] << " ";
        std::cout << " selected=(" << bp.first << "," << bp.second << ") gain=" << gain << "\n";
    }
    return 0;
}

// ----- verify mode: reproduce python reference values on the fixed n=6 state -----
static int verify() {
    int n = 6;
    ublas::vector<double> p(2 * n + 2);
    double mu0[6] = {0.30, -0.50, 0.10, 0.80, -0.20, 0.40};
    double sg0[6] = {0.50, 0.70, 0.45, 0.90, 0.60, 0.55};
    for (int i = 0; i < n; i++) { p(i) = mu0[i]; p(n + i) = log(sg0[i]); }
    p(2 * n) = log(1.5); p(2 * n + 1) = log(2.5);
    Observations obs(n);
    obs.X(0, 1) = 3; obs.X(1, 0) = 1; obs.X(2, 3) = 2; obs.X(0, 4) = 2; obs.X(4, 5) = 1; obs.X(5, 2) = 2; obs.X(3, 1) = 1;
    std::vector<double> mu, sg; double al, be; extract(p, n, mu, sg, al, be);
    std::vector<double> z = {0.4, -0.6, 0.05, 1.0, -0.3, 0.5};
    std::cout.precision(6); std::cout << std::fixed;
    std::cout << "inversions = " << actual_inversions(mu, z, n) << "\n";
    std::cout << "prob_win(0,1) = " << prob_win(mu, sg, 0, 1) << "  prob_win(2,3) = " << prob_win(mu, sg, 2, 3) << "\n";
    std::vector<double> g(2 * n + 2, 0.0);
    std::cout << "expected_inversions = " << expected_inversions(mu, sg, n, g.data()) << "\n";
    std::cout << "Einv grad ="; for (double v : g) std::cout << " " << v; std::cout << "\n";
    std::cout << "EIG(0,1) = " << eig_pair(mu[0] - mu[1], sqrt(sg[0]*sg[0]+sg[1]*sg[1])) << "\n";
    std::cout << "EIG(2,3) = " << eig_pair(mu[2] - mu[3], sqrt(sg[2]*sg[2]+sg[3]*sg[3])) << "\n";
    std::vector<double> d0, d1; compute_dobs(mu, sg, n, d0, d1);
    std::cout << "dobs[0,1] = " << d0[0*n+1] << " " << d1[0*n+1] << "  dobs[1,0] = " << d0[1*n+0] << " " << d1[1*n+0] << "\n";
    for (int tgt = 0; tgt < 2; tgt++) {
        std::vector<double> df(2*n+2, 0.0);
        if (tgt == 1) expected_inversions(mu, sg, n, df.data());
        else for (int i = 0; i < n; i++) df[n+i] = 2.0 * sg[i];
        std::vector<double> factor = acq_factor(mu, sg, al, be, obs, n, df);
        std::pair<int,int> bp = select_factor_based(mu, sg, al, be, obs, n, tgt == 1);
        // recompute best gain for print
        std::vector<double> dd0, dd1; compute_dobs(mu, sg, n, dd0, dd1);
        double best = 1e300;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
            double ifw = (factor[n+i]*sg[i]+factor[n+j]*sg[j])*dd0[i*n+j] + (factor[i]-factor[j])*dd1[i*n+j];
            double ifl = (factor[n+j]*sg[j]+factor[n+i]*sg[i])*dd0[j*n+i] + (factor[j]-factor[i])*dd1[j*n+i];
            double pp = prob_win(mu, sg, i, j); double gg = pp*ifw+(1-pp)*ifl; if (gg<best) best=gg;
        }
        std::cout << (tgt ? "inv" : "var") << ": factor[:4]=" << factor[0] << " " << factor[1] << " " << factor[2] << " " << factor[3]
                  << " selected_pair=(" << bp.first << ", " << bp.second << ") best_gain=" << best << "\n";
    }
    return 0;
}

// ----- verify3: tie-model checks -- FD gradients (with and without ties), fit, reference values -----
static int verify3(const std::string& atomfile) {
    int n = 6;
    Observations obs(n);
    int O[9][3] = {{0,1,3},{1,0,1},{2,3,2},{0,4,2},{4,5,1},{5,2,2},{3,1,1},{2,5,2},{4,0,1}};
    for (auto& o : O) obs.X(o[0], o[1]) = o[2];
    int TT[4][3] = {{0,1,2},{2,3,1},{1,4,2},{3,5,1}};
    for (auto& t : TT) obs.T(t[0], t[1]) = t[2];

    ublas::vector<double> p(2 * n + 2);
    double mu0[6] = {0.30, -0.50, 0.10, 0.80, -0.20, 0.40};
    double sg0[6] = {0.50, 0.70, 0.45, 0.90, 0.60, 0.55};
    for (int i = 0; i < n; i++) { p(i) = mu0[i]; p(n + i) = log(sg0[i]); }
    p(2 * n) = log(1.5); p(2 * n + 1) = log(2.5);

    std::cout.precision(8); std::cout << std::fixed;
    for (int tie_on = 0; tie_on <= 1; tie_on++) {
        Ranker rk(n, p);
        if (tie_on) {
            rk.ties = TieModel::load(atomfile);
            for (int j = 0; j < rk.ties->J; j++) rk.ties->w[j] = (1.0 + j % 3);  // non-uniform w
            double Z = 0; for (double x : rk.ties->w) Z += x;
            for (auto& x : rk.ties->w) x /= Z;
            rk.ties->premix();
        }
        rk.evaluate(obs, true, false);
        double v0 = *rk.val;
        ublas::vector<double> g0 = *rk.gradient;
        double h = 1e-6, maxerr = 0;
        Ranker rp(n, p), rm(n, p);                 // constructed once; only params(k) varies
        rp.ties = rk.ties; rm.ties = rk.ties;
        for (int k = 0; k < 2 * n + 2; k++) {
            rp.params(k) = p(k) + h; rm.params(k) = p(k) - h;
            rp.evaluate(obs); rm.evaluate(obs);
            maxerr = std::max(maxerr, std::fabs((*rp.val - *rm.val) / (2 * h) - g0(k)));
            rp.params(k) = p(k); rm.params(k) = p(k);
        }
        std::cout << (tie_on ? "ties   " : "no-ties") << ": val=" << v0 << "  FD grad max err=" << std::scientific << maxerr << std::fixed << "\n";
    }
    // fit with ties and report the tie posterior
    Ranker rk(n, p);
    rk.ties = TieModel::load(atomfile);
    rk.fit(obs);
    double Eλ = rk.ties->E_lambda();
    std::vector<double> mu(n), sg(n);
    for (int i = 0; i < n; i++) { mu[i] = rk.params(i); sg[i] = exp(rk.params(n + i)); }
    std::cout << "fitted: val=" << *rk.val << "  E[λ]=" << Eλ << "\nmu=";
    for (int i = 0; i < n; i++) std::cout << mu[i] << " ";
    auto bp = select_eig_ties(mu, sg, n, *rk.ties);
    std::cout << "\nEIG3 selected pair=(" << bp.first << "," << bp.second << ")\n";
    return 0;
}

// ----- tiebench: time a single fit with tie support -----
static int tiebench(const std::string& atomfile, int nb, int count, double λtrue) {
    Instance inst = Instance::random(nb);
    Observations obs(nb, inst, count, λtrue);
    Ranker ranker(nb, init_params(nb));
    ranker.ties = TieModel::load(atomfile);
    auto t0 = std::chrono::high_resolution_clock::now();
    ranker.fit(obs, 1e-8, 1e6, false);
    auto t1 = std::chrono::high_resolution_clock::now();
    double Eλ = ranker.ties->E_lambda();
    std::vector<double> mu(nb), z(nb);
    for (int i = 0; i < nb; i++) { mu[i] = ranker.params(i); z[i] = inst.z(i); }
    std::cout << "n=" << nb << " comparisons=" << count << " λ*=" << λtrue
              << "  fit " << std::chrono::duration<double>(t1 - t0).count() << " s"
              << "  E[λ]=" << Eλ << "  inversions=" << actual_inversions(mu, z, nb) << "\n";
    return 0;
}

// ----- one experiment run; fills traj[strat][step] for this run -----
static const int MAXSTEPS = 2048;   // trajectory buffer size; steps must not exceed this
static void run_once(int n, int steps, unsigned long seed, double traj[4][MAXSTEPS]) {
    RNGType rng(seed);
    // v ~ InvGamma(Mα, scale=1/Mβ)  <=>  1/v ~ Gamma(shape=Mα, scale=Mβ)
    GammaGen gg(rng, Gamma(Mα, Mβ));
    NormalGen ng(rng, Normal(0, 1));
    boost::uniform_01<RNGType&> unif(rng);
    // instance
    double v = 1.0 / gg();
    std::vector<double> z(n);
    for (int i = 0; i < n; i++) z[i] = ng() * sqrt(v);
    for (int strat = 0; strat < 4; strat++) {  // 0 random,1 eig,2 var,3 inv
        Observations obs(n);
        Ranker rk(n, init_params(n));
        for (int t = 0; t < steps; t++) {
            rk.fit(obs);
            std::vector<double> mu, sg; double al, be; extract(rk.params, n, mu, sg, al, be);
            traj[strat][t] += actual_inversions(mu, z, n);
            int i, j;
            if (strat == 0) { i = (int)(unif() * n); do { j = (int)(unif() * n); } while (j == i); }
            else if (strat == 1) { auto pr = select_eig(mu, sg, n); i = pr.first; j = pr.second; }
            else { auto pr = select_logdomain(rk, obs, n, strat == 3); i = pr.first; j = pr.second; }
            double pij = 1.0 / (1.0 + exp(-(z[i] - z[j])));
            if (unif() < pij) obs.X(i, j) += 1; else obs.X(j, i) += 1;
        }
    }
}

int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "verify";
    if (mode == "verify") return verify();
    if (mode == "verify2") return verify2();
    if (mode == "verify3") return verify3(argc > 2 ? argv[2] : TIE_ATOMS_DEFAULT);
    if (mode == "tiebench") {   // tiebench <n> <comparisons> <lambda_true> [atomfile]
        int nb = argc > 2 ? atoi(argv[2]) : 200;
        int count = argc > 3 ? atoi(argv[3]) : 160000;
        double λt = argc > 4 ? atof(argv[4]) : 0.75;
        return tiebench(argc > 5 ? argv[5] : TIE_ATOMS_DEFAULT, nb, count, λt);
    }
    if (mode == "n3test") {
        const int n = 3;
        ublas::vector<double> params(2 * n + 2);
        params <<= 0.1, -0.2, 0.3, log(1.1), log(2.2), log(3.3), log(1.5), log(3.0);
        Ranker ranker(n, params);
        Observations obs(n);
        obs.X(0, 2) = 2.0; obs.X(1, 2) = 2.0; obs.X(2, 0) = 3.0; obs.X(1, 0) = 1.0;
        ranker.fit(obs, 1e-8, 1e6, false);
        std::cout << "n=3 converged val = " << *ranker.val << " (expect 6.08618 with correction)\n";
        return 0;
    }
    if (mode == "bench") {   // bench <n> <comparisons> : time a single fit
        int nb = argc > 2 ? atoi(argv[2]) : 200;
        int count = argc > 3 ? atoi(argv[3]) : 160000;
        Instance inst = Instance::random(nb);
        Observations obs(nb, inst, count);
        ublas::vector<double> params(2 * nb + 2);
        for (int i = 0; i < nb; i++) { params(i) = 0.0; params(nb + i) = 0.0; }  // mu=0, ln sigma=0
        params(2 * nb) = log(1.2); params(2 * nb + 1) = log(2.0);                  // ln alpha, ln beta
        Ranker ranker(nb, params);
        auto t0 = std::chrono::high_resolution_clock::now();
        ranker.fit(obs, 1e-8, 1e6, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "n=" << nb << " comparisons=" << count << " nnz=" << obs.X.nnz()
                  << "  fit " << secs << " s\n";
        return 0;
    }
    // experiment mode: experiment <n> <steps> <nruns> <seed0> <outfile>
    int n = argc > 2 ? atoi(argv[2]) : 12;
    int steps = argc > 3 ? atoi(argv[3]) : 500;
    if (steps > MAXSTEPS) { std::cerr << "steps " << steps << " exceeds MAXSTEPS " << MAXSTEPS << "\n"; return 1; }
    int nruns = argc > 4 ? atoi(argv[4]) : 300;
    unsigned long seed0 = argc > 5 ? strtoul(argv[5], 0, 10) : 1;
    std::string outfile = argc > 6 ? argv[6] : "experiment_out.csv";

    static double sum[4][MAXSTEPS];
    for (int s = 0; s < 4; s++) for (int t = 0; t < MAXSTEPS; t++) sum[s][t] = 0;

    int nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    std::vector<std::vector<std::vector<double>>> tls(nthreads, std::vector<std::vector<double>>(4, std::vector<double>(MAXSTEPS, 0.0)));
    std::atomic<int> done(0);
    auto worker = [&](int tid) {
        for (int r = tid; r < nruns; r += nthreads) {
            static thread_local double tr[4][MAXSTEPS];
            for (int s = 0; s < 4; s++) for (int t = 0; t < MAXSTEPS; t++) tr[s][t] = 0;
            run_once(n, steps, seed0 + 1000003UL * (unsigned long)r, tr);
            for (int s = 0; s < 4; s++) for (int t = 0; t < steps; t++) tls[tid][s][t] += tr[s][t];
            done++;
        }
    };
    for (int t = 0; t < nthreads; t++) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    for (int tid = 0; tid < nthreads; tid++) for (int s = 0; s < 4; s++) for (int t = 0; t < steps; t++) sum[s][t] += tls[tid][s][t];

    const char* names[4] = {"random", "eig", "var", "inv"};
    std::ofstream f(outfile);
    f << "step,random,eig,var,inv\n";
    for (int t = 0; t < steps; t++) { f << t; for (int s = 0; s < 4; s++) f << "," << sum[s][t] / nruns; f << "\n"; }
    f.close();
    std::cout.precision(3); std::cout << std::fixed;
    for (int s = 0; s < 4; s++) {
        double final = sum[s][steps-1] / nruns, auc = 0; for (int t = 0; t < steps; t++) auc += sum[s][t] / nruns; auc /= steps;
        std::cout << names[s] << ": final=" << final << " auc=" << auc << "\n";
    }
    std::cout << "runs=" << nruns << " threads=" << nthreads << " -> " << outfile << "\n";
    return 0;

}