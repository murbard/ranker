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

// boost optional
#include <boost/optional.hpp>

using namespace boost;
using namespace boost::numeric;
using namespace boost::numeric::ublas;
using namespace boost::math;

const double Mα = 1.2;
const double Mβ = 2.0;

// boost variate generator
typedef boost::mt19937 RNGType;
typedef boost::random::normal_distribution<> Normal;
typedef boost::random::variate_generator<RNGType&, Normal> NormalGen;

typedef boost::random::gamma_distribution<> Gamma;
typedef boost::random::variate_generator<RNGType&, Gamma> GammaGen;
RNGType rng;

GammaGen gammagen(rng, Gamma(Mα, 1.0/  Mβ));
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
        // mapped matrix X
        mapped_matrix<int> X;
        // optional coordinate matrix that matches X
        optional<coordinate_matrix<int> > Xcoord;

        Observations(int n) : X(n, n) {}
        Observations(int n, const Instance& instance, int count) : X(n,n) {
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
                }
            }
        }
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

        void computeCoordinateMatrix() {
            Xcoord = optional<coordinate_matrix<int>>(X);
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

        Ranker(int n) : n(n), params(2 * n + 2), hessian(), gradient(), val() {}
        Ranker(int n, const vector<double> &params) : n(n), params(params), hessian(), gradient(), val() {}

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

            // Now we add observations
            // We loop over all non zero elements of the sparse observation matrix, obs.X

            coordinate_matrix<double> hobs(2 * n, 2 * n, obs.X.nnz() * 6);

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
                        (*g_μ)(i) += count * gμδ;
                        (*g_μ)(j) -= count * gμδ;

                        // dv/dlnσi, dv/dlnσj
                        (*g_σ)(i) -= count * (gσδ * σ2(i) / σδ);
                        (*g_σ)(j) -= count * (gσδ * σ2(j) / σδ);
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
                        hessian->diag(i) -= count * hμμδ;
                        hessian->diag(j) -= count * hμμδ;
                        hessian->diag(n+i) -= count * σ2(i) / σδ * (gσδ * σ2(j) / (σδ * σδ) + hσσδ * σ2(i) / σδ + gσδ);
                        hessian->diag(n+j) -= count * σ2(j) / σδ * (gσδ * σ2(i) / (σδ * σδ) + hσσδ * σ2(j) / σδ + gσδ);

                        int row = std::min(i, j);
                        int col = std::max(i, j);

                        // hμiμj, hσiσj
                        hobs.append_element(row, col, count * hμμδ);
                        hobs.append_element(n + row, n + col, - count * (σ2(i) * σ2(j) / (σδ * σδ) * (hσσδ - gσδ / σδ)));

                        // hμiσi, hμjσi
                        double chσ = count * hμσδ * σ2(i) / σδ;
                        hobs.append_element(i, n + i, - chσ);
                        hobs.append_element(j, n + i, chσ);

                        // hμiσj, hμjσj
                        chσ = count * hμσδ * σ2(j) / σδ;
                        hobs.append_element(i, n + j, - chσ);
                        hobs.append_element(j, n + j, chσ);
                    }
                }
                if (compute_hessian)
                    hessian->obs = std::move(hobs);

            }
            if (!val)
                val = optional<double>(v);
            else
                *val = v;
        }

        void fit(const Observations& obs, const double tol = 1e-8, const int max_iter = 1e6) {
            int k = 0;
            while (k < max_iter) {
                evaluate(obs, true, true);
                // if the gradient is small, we're done
                if (norm_2(*gradient) < tol) {
                    break;
                }
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
                do {
                    // cg (hessian, gradient, dx,  1e-10, tol, 1000 * n );
                    // use the ublas cg solver with diagonal preconditioner
                    vector<double> save_params(2 * n + 2);
                    auto dx = std::move(hessian->cg(*gradient, λ, tol, 1000 * n));
                    other = std::move(Ranker(n, params - dx));
                    other.evaluate(obs);
                    if ((*other.val) < (*val)) {
                        break;
                    } else {
                        λ *= 10;
                        if (λ > 1e10)
                            throw std::runtime_error("giving up");
                    }
                } while (true);
                *this = std::move(other);
                std::cout << "iter: " << k << ", val:" << *val << ", λ: " << λ << std::endl;
                k++;
            }
        }
};

/// @brief
/// @param argc
/// @param argv
/// @return
int main(int argc, char ** argv) {

    const int n = 3;

    // initialize ublas vector to {0.1, -0.2, 0.3, log(1.1), log(2.2), log(3.3), log(1.5), log(3.0)}
    ublas::vector<double> params(2 * n + 2);
    params <<= 0.1, -0.2, 0.3, log(1.1), log(2.2), log(3.3), log(1.5), log(3.0);
    Ranker ranker(n, params);
    Observations obs(n);
    obs.X(0, 2) = 2.0;
    obs.X(1, 2) = 2.0;
    obs.X(2, 0) = 3.0;
    obs.X(1, 0) = 1.0;

    ranker.evaluate(obs, true, true);

    // print all parts of the gradient and hessian
    std::cout << ranker.print_gradient() << std::endl;
    std::cout << ranker.print_hessian() << std::endl;

    // fit
    ranker.fit(obs);




    ranker.evaluate(obs, true, true);
    std::cout << ranker.print_gradient() << std::endl;
    std::cout << ranker.print_hessian() << std::endl;
    std::cout << *ranker.val << std::endl;

    int m = 200;
    Instance inst = Instance::random(m);
    obs = Observations(m, inst, m * m* 10);
    ranker = Ranker(m);
    ranker.fit(obs);

    return 0;

}