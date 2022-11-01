// include boost ublas for matrices and vectors
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

// boost special functions erf, erfc, polygamma, digamma, lngamma
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

// boost optional
#include <boost/optional.hpp>

using namespace boost;
using namespace boost::numeric;
using namespace boost::numeric::ublas;
using namespace boost::math;

const double Mα = 1.2;
const double Mβ = 2.0;

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
};

class Observations {
    public:
        // mapped matrix X
        mapped_matrix<int> X;

        Observations(int n, int m) : X(n, m) {}
        ~Observations() {}
        Observations(const Observations& other) : X(other.X) {}
        Observations(Observations&& other) : X(std::move(other.X)) {}
        Observations& operator=(const Observations& other) {
            X = other.X;
            return *this;
        }
        Observations& operator=(Observations&& other) {
            X = std::move(other.X);
            return *this;
        }
};

class Hessian {
    public:
        coordinate_matrix<int> obs;
        vector<double> diag;
        double αβ;
        vector<double> μσαβ;

        Hessian(int n) : obs(n, 2 * n + 2), diag(n), αβ(0), μσαβ(2 * n + 2) {}
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
};

class Ranker {
    public:
        int n;
        vector<double> params;

        Ranker(int n) : n(n), params(2 * n + 2), hessian(), gradient(), val() {}
        Ranker(int n, vector<double> params) : n(n), params(params), hessian(n), gradient(2 * n + 2) {}

        ~Ranker() {}
        Ranker(const Ranker& other) : n(other.n), params(other.params), hessian(other.hessian), gradient(other.gradient) {}
        Ranker(Ranker&& other) : n(std::move(other.n)), params(std::move(other.params)), hessian(std::move(other.hessian)), gradient(std::move(other.gradient)) {}
        Ranker& operator=(const Ranker& other) {
            n = other.n;
            params = other.params;
            hessian = other.hessian;
            gradient = other.gradient;
            return *this;
        }
        Ranker& operator=(Ranker&& other) {
            n = std::move(other.n);
            params = std::move(other.params);
            hessian = std::move(other.hessian);
            gradient = std::move(other.gradient);
            return *this;
        }

    private:
        optional<double> val;
        optional<vector<double>> gradient;
        optional<Hessian> hessian;

        void evaluate(const Observations& obs, const bool compute_gradient, const bool compute_hessian) {
            const int n = this->n;
            const double ln_α = this->params(2 * n);
            const double ln_β = this->params(2 * n + 1);
            const double α = exp(ln_α);
            const double β = exp(ln_β);
            const double αβ = α * β;
            const double psi_α = digamma(α);
            const double psi1_α = trigamma(α);
            const double psi2_α = polygamma(2, α);
            // μ is a vector slice of the virst n elements of params. It's just a view on the object and provides
            // a lightweight proxy, no copy happens. This uses thje boost function "project"
            vector_range<vector<double>> μ = project(this->params, ublas::range(0, n));
            vector_range<vector<double>> ln_σ = project(this->params, ublas::range(n, 2 * n));
            vector<double> σ2(n);
            vector_range<vector<double>> g_μ = project(*(this->gradient), ublas::range(0, n));
            vector_range<vector<double>> g_σ = project(*(this->gradient), ublas::range(n, 2 * n));

            if (compute_gradient) {
                // if gradient is null initialize it
                if (!this->gradient) {
                    this->gradient = vector<double>(2 * n + 2);
                }
            }
            if (compute_hessian) {
                // if hessian is null initialize it
                if (!this->hessian) {
                    this->hessian = Hessian(n);
                }
                // it it's not null, erase the obs part, the rest is just overwritten
                this->hessian->obs.clear();
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
                (*(this->gradient))(2 * n) = 0.5 * α *( -2 + 2 * β / Mβ - (2 * Mα + n - 2 * α) * psi1_α + β * Σ_μ2_σ2);
                // dv/dβ
                (*(this->gradient))(2 * n + 1) = - Mα - 0.5 * n + αβ / Mβ + 0.5 * αβ * Σ_μ2_σ2;

                // dv/dμi and dv/dlnσi
                // set the first n elements of gh.g to αβ * ranker.μ using daxpy
                g_μ = αβ * μ;
                // set the elements n to 2n of gh.g to -1 + αβ * σ2
                g_σ = αβ * σ2;
                for (auto& el : g_σ) { el -= 1.0; }
             }

             if (compute_hessian) {

                // Compute Hessian
                // d²v/dμi² and d²v/dlnσi²
                // set the first n elements of gh.h.diag to 1 and the next n elements to 2 * σ2
                vector_range<vector<double>> diag_μ = project(this->hessian->diag, ublas::range(0, n));
                diag_μ = vector<double>(n, 1.0);
                vector_range<vector<double>> diag_σ = project(this->hessian->diag, ublas::range(n, 2 * n));
                diag_σ = 2.0 * σ2;
                // scale the whole thing by αβ
                this->hessian->diag *= αβ;

                // d²v/dα²
                this->hessian->diag(2 * n) =
                    0.5 * α *(-2 + 2 * β / Mβ - (2 * Mα + n - 4 * α) * psi1_α + β * Σ_μ2_σ2 - (2 * Mα + n - 2 * α) * α * psi2_α);

                // d²v/dαdβ
                this->hessian->αβ = 0.5 * αβ * (2 / Mβ + Σ_μ2_σ2);

                // d²v/dβ²
                this->hessian->diag(2 * n + 1) = this->hessian->αβ; // same value as the cross term

                // d²v/d{μ|σ}id{α|β}
                // place vector ranker.μ as the first half of the gh.h.μσαβ vector
                vector_range<vector<double>> αβ_μ = project(this->hessian->μσαβ, ublas::range(0, n));
                αβ_μ = μ;
                // place vector σ2 as the second half of the gh.h.μσαβ vector
                vector_range<vector<double>> αβ_σ = project(this->hessian->μσαβ, ublas::range(n, 2 * n));
                αβ_σ = σ2;
                // scale gh.h.μσαβ by αβ
                this->hessian->μσαβ *= αβ;
            }

            // Now we add observations
            // We loop over all non zero elements of the sparse observation matrix, obs.X

            for(auto it = obs.X.begin1() ; it != obs.X.end1(); ++it) {
                for(auto el = it.begin(); el != it.end(); ++el) {
                    const int i = el.index1();
                    const int j = el.index2();
                    if (i == j) {
                        continue; // this should not happen
                    }
                    const int count = *el;

                    const double σδ = sqrt(σ2(i) + σ2(j));
                    const double h = sqrt(16.0 / M_PI + 2 * σδ * σδ);
                    const double sqrt_π_h = sqrt(M_PI) * h;
                    const double μδ = μ(i) - μ(j);
                    const double μδ_over_h = μδ / h;
                    const double exp_minus_μδ_over_h_square = exp(-μδ_over_h * μδ_over_h);
                    const double gμδ = - 0.5 * (1 - erf(μδ_over_h));
                    const double gσδ = - exp_minus_μδ_over_h_square * σδ / sqrt_π_h;

                    v += count * (exp_minus_μδ_over_h_square * h / (2 * sqrt(M_PI)) + μδ * gμδ);

                    if (compute_gradient) {
                        // Compute gradient
                        // dv/dμi, dv/dμj
                        g_μ(i) += count * gμδ;
                        g_μ(j) -= count * gμδ;

                        // dv/dlnσi, dv/dlnσj
                        g_σ(i) -= count * (gσδ * σ2(i) / σδ);
                        g_σ(j) -= count * (gσδ * σ2(j) / σδ);
                    }

                    if (compute_hessian) {

                        //    hμμδ = - exp(-(μδ/h)**2) / (sqrt(π) * h)
                        //    hμσδ = 2 * μδ * σδ * exp(-(μδ/h)**2) / (sqrt(π) * h**3)
                        //    hσσδ = - exp(-(μδ/h)**2) * (256 + 4 * π * σδ**2 * (8 + μδ**2)) / (π**(5/2) * h**5)

                        const double hμμδ = - exp_minus_μδ_over_h_square / sqrt_π_h;
                        const double hμσδ = 2 * μδ * σδ * exp_minus_μδ_over_h_square / (sqrt(M_PI) * h * h * h);
                        double sqrt_π_h5 = sqrt_π_h * sqrt_π_h; sqrt_π_h5 *= sqrt_π_h5; sqrt_π_h5 *= sqrt_π_h;
                        const double hσσδ = - exp_minus_μδ_over_h_square * (256 + 4 * M_PI * σδ * σδ * (8 + μδ * μδ)) / sqrt_π_h5;

                        // diagonal hessian terms
                        this->hessian->diag(i) -= count * hμμδ;
                        this->hessian->diag(j) -= count * hμμδ;
                        this->hessian->diag(n+i) -= count * σ2(i) / σδ * (gσδ * σ2(j) / (σδ * σδ) + hσσδ * σ2(i) / σδ + gσδ);
                        this->hessian->diag(n+j) -= count * σ2(j) / σδ * (gσδ * σ2(i) / (σδ * σδ) + hσσδ * σ2(j) / σδ + gσδ);

                        int row = std::min(i, j);
                        int col = std::max(i, j);

                        // hμiμj, hσiσj
                        this->hessian->obs(row, col) += count * hμμδ;
                        this->hessian->obs(n + row, n + col) -= count * (σ2(i) * σ2(j) / (σδ * σδ) * (hσσδ - gσδ / σδ));

                        // hμiσi, hμjσi
                        double chσ = count * hμσδ * σ2(i) / σδ;
                        this->hessian->obs(i, n + i) -= chσ;
                        this->hessian->obs(j, n + i) += chσ;

                        // hμiσj, hμjσj
                        chσ = count * hμσδ * σ2(j) / σδ;
                        this->hessian->obs(i, n + j) -= chσ;
                        this->hessian->obs(j, n + j) += chσ;
                    }
                }
            }
        }
};
