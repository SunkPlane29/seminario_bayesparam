using Turing, StatsPlots, Distributions, NonlinearSolve, KernelDensity

function create_posterior(prior_dist, x, β)
    return αi -> begin
        if !(insupport(prior_dist, αi))
            return 0.0
        end
        
        log_prior_density = logpdf(prior_dist, αi)
        
        log_likelihood_sum = 0.0
        for xi in x
            log_likelihood_sum += logpdf(Cauchy(xi, β), αi)
        end
        
        return exp(log_prior_density + log_likelihood_sum)
    end
end

begin
    prior = Uniform(-5.0, 5.0) 
    α = 1.0
    β = 2.0

    # n = [1, 2, 4, 8, 16, 32, 64, 128, 200]
    n = 1:200

    θdist = Uniform(-π/2, π/2)
    αrange = -6.0:0.01:6.0

    anim = @animate for ni in n
        θ_obs = rand(θdist, ni)
        x_obs = @. α + β * tan(θ_obs)

        posterior_pdf_func = create_posterior(prior, x_obs, β)
        unnormalized_posterior_values = [posterior_pdf_func(αi) for αi in αrange]

        integral_approximation = sum(unnormalized_posterior_values) * step(αrange)
        normalized_posterior_values = if integral_approximation > 0
            unnormalized_posterior_values ./ integral_approximation
        else
            unnormalized_posterior_values
        end

        plot(αrange, normalized_posterior_values, label="Posterior PDF", xlabel="Parameter (α)", 
            ylabel="Density", title="Posterior Distribution (n = $ni)")
        prior_pdf_values = [pdf(prior, αi) for αi in αrange]
        plot!(αrange, prior_pdf_values, label="Prior PDF", linestyle=:dash)
    end

    gif(anim, "belief_evolution.gif", fps=7)
end

begin
    prior = Uniform(-5.0, 5.0) 
    α = 1.0 
    β = 2.0 

    θdist = Uniform(-π/2, π/2)
    θ_obs = rand(θdist, 200) 
    
    x_obs = @. α + β * tan(θ_obs)

    αrange = -6.0:0.01:6.0

    posterior_pdf_func = create_posterior(prior, x_obs, β)
    unnormalized_posterior_values = [posterior_pdf_func(αi) for αi in αrange]
    
    integral_approximation = sum(unnormalized_posterior_values) * step(αrange)
    normalized_posterior_values = if integral_approximation > 0
        unnormalized_posterior_values ./ integral_approximation
    else
        unnormalized_posterior_values
    end

    println(αrange[argmax(normalized_posterior_values)])
    plot(αrange, normalized_posterior_values, label="Posterior PDF", xlabel="Parameter (α)", 
        ylabel="Density", title="Posterior Distribution", xlimit=(0.0, 2.0))
    
    prior_pdf_values = [pdf(prior, αi) for αi in αrange]
    plot!(αrange, prior_pdf_values, label="Prior PDF", linestyle=:dash)
    savefig("posterior_1.png")
end

@model function farol(x)
    α ~ Uniform(-5.0, 5.0)
    β ~ Uniform(0.0, 5.0) 

    x ~ Cauchy(α, β)
end

begin
    α = 1.0
    β = 2.0

    θdist = Uniform(-π/2, π/2)
    θ_obs = rand(θdist, 200) 

    x_obs = @. α + β * tan(θ_obs)

    model = farol(x_obs)

    # chain = sample(model, MH(), 100_000) 
    chain = sample(model, NUTS(), 10_000)
    describe(chain)
    plot(chain) 
    savefig("chain.png")
end

begin
    αvals = vec(chain[:α].data)
    βvals = vec(chain[:β].data)

    println(typeof(αvals))
    println(typeof(βvals))

    α_1σ = quantile(αvals, [0.16, 0.84])
    α_2σ = quantile(αvals, [0.025, 0.975])

    β_1σ = quantile(βvals, [0.16, 0.84])
    β_2σ = quantile(βvals, [0.025, 0.975])

    post = kde((αvals, βvals))

    imax = argmax(post.density)

    indα = imax[1]
    indβ = imax[2]

    α_map = post.x[indα]
    β_map = post.y[indβ]

    println("MAP α: ", α_map)
    println("MAP β: ", β_map)
    
    println("α 1σ: ", α_1σ)
    println("α 2σ: ", α_2σ)

    println("β 1σ: ", β_1σ)
    println("β 2σ: ", β_2σ)

    marginalkde(αvals, βvals, levels=3, clip=((-3.0, 3.0), (-3.0, 3.0)), 
        xlabel=raw"$\alpha$ [km]", ylabel=raw"$\beta$ [km]")
    savefig("posterior_2.png")
end