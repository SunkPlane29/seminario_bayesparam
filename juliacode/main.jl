using Turing, StatsPlots, Distributions, NonlinearSolve

# This function returns another function, which calculates the unnormalized posterior PDF.
# The returned function takes one argument: 'theta_val' (the parameter value).
# It assumes the likelihood of theta_val given an observation xi_val is pdf(Cauchy(xi_val, β_likelihood_scale), theta_val).
function create_posterior_pdf_calculator(prior_dist, data_observed, β_likelihood_scale)
    return theta_val -> begin
        # Check if theta_val is within the support of the prior
        if !(insupport(prior_dist, theta_val))
            return 0.0
        end
        
        log_prior_density = logpdf(prior_dist, theta_val)
        
        log_likelihood_sum = 0.0
        for xi_val in data_observed
            # Likelihood of theta_val given xi_val: pdf(Cauchy(location=xi_val, scale=β_likelihood_scale), theta_val)
            log_likelihood_sum += logpdf(Cauchy(xi_val, β_likelihood_scale), theta_val)
        end
        
        # Return the unnormalized posterior density (prior * likelihood)
        return exp(log_prior_density + log_likelihood_sum)
    end
end

begin
    # n = [] # Original: n, unused in the snippet
    # posterior = [] # Original: posterior, unused in the snippet

    prior = Uniform(-5.0, 5.0) # Prior for the unknown parameter (let's call it theta_param)
    α = 1.0 # Parameter for data generation
    β_data_gen = 2.0 # Parameter for data generation (scale in Cauchy, or part of location)
    
    # This is the β used as the scale parameter in the Cauchy likelihood term.
    # It might be different from β_data_gen.
    β_likelihood_scale = 2.0 #km 

    θuniform = Uniform(-π/2, π/2)
    # Generate a single random angle for data generation
    # Note: rand(θuniform, 1) returns a 1-element vector.
    # For a single scalar, use rand(θuniform).
    θ_for_data = rand(θuniform, 10) # Results in a 1-element vector
    
    # Generate observed data x. Original code produces a 1-element vector.
    # x_observed = @. α + β_data_gen + tan(θ_for_data)
    # This means x_observed[1] is a draw from Cauchy(α + β_data_gen, 1.0)
    # Let's ensure x_observed is a collection (e.g., a vector) of observations.
    x_observed = @. α + β * tan(θ_for_data) # A single observation in a vector

    # Define the range of parameter values for plotting the posterior
    # This range should ideally cover where the posterior has significant mass.
    theta_param_range = -6.0:0.0001:6.0

    # Create the posterior PDF calculator function
    posterior_pdf_func = create_posterior_pdf_calculator(prior, x_observed, β_likelihood_scale)

    # Evaluate the unnormalized posterior PDF over the defined range
    unnormalized_posterior_values = [posterior_pdf_func(tp) for tp in theta_param_range]
    
    # Normalize the posterior PDF values for plotting (simple Riemann sum normalization)
    # This makes the area under the curve approximate to 1.
    integral_approximation = sum(unnormalized_posterior_values) * step(theta_param_range)
    normalized_posterior_values = if integral_approximation > 0
        unnormalized_posterior_values ./ integral_approximation
    else
        unnormalized_posterior_values # Avoid division by zero if all values are zero
    end

    # Plot the normalized posterior PDF
    plot(theta_param_range, normalized_posterior_values, label="Posterior PDF", xlabel="Parameter (θ)", ylabel="Density", title="Posterior Distribution")
    
    # Optionally, plot the prior PDF for comparison
    prior_pdf_values = [pdf(prior, tp) for tp in theta_param_range]
    plot!(theta_param_range, prior_pdf_values, label="Prior PDF", linestyle=:dash)
    
    # To display the plot
    # display(current()) # or savefig("posterior_plot.png")
end