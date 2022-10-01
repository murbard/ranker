from scipy.special import polygamma, erf
from np import exp, sqrt, pi as π

# alpha: α
# gamma : γ
# delta : δ
# Sigma: Σ
# pi: π


def gradients():

    # ∫ InvGamma(α, β, v) log( InvGamma(α', β', v)) dv
    dα += (1 + model.α) * polygamma(1, α) - β / model.b
    dβ += - α / model.β + 1 / β + model.α / β

    # ∫∫ InvGamma(α, β, v) Normal(μ, σ, z) log(Normal(0, √v)) dz dv

    dμ += - α * β * μ
    dσ += - α * β * σ
    dα += - β / 2 * (μ**2 +  σ**2) + 1/2 * polygamma(1, α)
    dβ += - α / 2 * (μ**2 +  σ**2) + 1/(2 * β)

    # Σ_ij - o[i,j] ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ)) dδ'
    # consider f(μi, μj, σi, σj) = ∫ Normal(μi - μj, √(σi²+σj²), δ) log(1 + e^(-δ))
    # let σ = √(σi²+σj²) and μ = μi - μj

    # dμ / dμi = 1
    # dμ / dμj = -1
    # dσ / dσi = σi / √(σi^2 + σj^2)
    # dσ / dσj = σj / √(σi^2 + σj^2)

    # df / dμ =  ∫ Normal(0, 1, δ') / (1 + e^(μ + σ δ')) dδ'
    # This can be approximated by the approximation to the logistic-normal integral
    # df / dσ =  ∫ Normal(0, 1, δ') δ' / (1 + e^(μ + σ δ')) dδ'


    for (i, j) in self.obs:

        σδ = sqrt(σ[i]**2 + σ[j]**2)
        μδ = μ[i] - μ[j]

        dσδ += - σδ * exp(-π μδ**2 / (16 + 2 * π * σδ**2)) / sqrt(16 + 2 * π * σδ**2)
        dμδ += - σδ /2 * (1 + erf( sqrt(π) * μδ / sqrt(2 + π * σδ**2 / 4)))

        dμ[i] += -dμδ
        dμ[j] += dμδ
        dσ[i] += dσδ * σ[i] / σδ
        dσ[j] += dσδ * σ[j] / σδ
