using Turing, StatsPlots, Distributions, NonlinearSolve

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

    plot(αrange, normalized_posterior_values, label="Posterior PDF", xlabel="Parameter (θ)", ylabel="Density", title="Posterior Distribution")
    
    prior_pdf_values = [pdf(prior, αi) for αi in αrange]
    plot!(αrange, prior_pdf_values, label="Prior PDF", linestyle=:dash)
end