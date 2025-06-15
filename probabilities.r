# Probability distribution functions:
# -> DISCRETE
#      - Probability mass function (PMF - d*)
#      - Cumulative distribution function (CDF - p*)
# -> CONTINUOUS
#      - Probability density function (PDF - d*)
#      - Cumulative distribution function (CDF - p*)

# =====================
# DISCRETE DISTRIBUTIONS
# =====================
cat("=== DISCRETE ===\n")

# Bernoulli 
# Single trial with success/failure (coin flip, pass/fail test, yes/no survey response)
# By definition Bernoulli has only two outputs (1=success, 0=failure)
# Interpretation:
#  - P(X = 1) = Probability of success on single trial
#  - P(X = 0) = Probability of failure on single trial
p <- 0.3     # Probability of success
cat("\n* Bernoulli *\n")
cat(sprintf("P(X=1): %.4f\n", dbinom(1, 1, p)))  # Exact point
cat(sprintf("P(X=0): %.4f\n", dbinom(0, 1, p)))

# ****************************************************************************************************************

# Binomial distribution
# Fixed number of independent Bernoulli trials (number of heads in 10 coin flips, defective items in batch)
n <- 10; p <- 0.4   # Number of trials, probability of success

# Interpretation:
#  - P(X = 3) = Probability of exactly 3 successes in 10 trials
#  - P(X ≤ 2) = Probability of 2 or fewer successes

cat("\n* Binomial *\n")

# Exact point probability
cat(sprintf("P(X=3): %.4f\n", dbinom(3, n, p)))  # Exactly 3 successes

# Range probabilities
cat(sprintf("P(3 < X ≤ 5): %.4f\n", pbinom(5, n, p) - pbinom(3, n, p)))  # 4 or 5 successes
cat(sprintf("P(2 ≤ X < 4): %.4f\n", pbinom(3, n, p) - pbinom(1, n, p)))   # 2 or 3 successes

# Tail probabilities
cat(sprintf("P(X > 4): %.4f\n", 1 - pbinom(4, n, p)))     # 5+ successes
cat(sprintf("P(X ≥ 6): %.4f\n", 1 - pbinom(5, n, p)))     # 6+ successes (strict)
cat(sprintf("P(X < 2): %.4f\n", pbinom(1, n, p)))         # 0 or 1 success
cat(sprintf("P(X ≤ 3): %.4f\n", pbinom(3, n, p)))         # 0-3 successes

# Special cases
cat(sprintf("P(X ≠ 4): %.4f\n", 1 - dbinom(4, n, p)))     # All except 4 successes
cat(sprintf("P(X ∈ {2,5,7}): %.4f\n", sum(dbinom(c(2,5,7), n, p))))  # Specific values

# ****************************************************************************************************************

# Geometric distribution
# Number of trials until first success (attempts until first sale, rolls until first six)
p <- 0.2  # Probability of success on each trial

cat("\n* Geometric *\n")

# Interpretations: 
#  - P(2 ≤ X ≤ 5) = Success occurs on trial 2, 3, 4, or 5
#  - P(X > 4) = First success takes 5+ trials

# Exact point probability (PMF)
cat(sprintf("P(X=5): %.4f\n", dgeom(4, p)))  # Exactly 5 trials needed for 1st success (R uses failures before success)

# Cumulative probabilities (CDF)
cat(sprintf("P(X ≤ 3): %.4f\n", pgeom(2, p)))  # Success within 1st 3 trials
cat(sprintf("P(X > 4): %.4f\n", 1 - pgeom(3, p)))  # Needs 5+ trials
cat(sprintf("P(X ≥ 3): %.4f\n", 1 - pgeom(1, p)))  # Alternative to P(X>2)

# Range probabilities 
cat(sprintf("P(2 ≤ X ≤ 5): %.4f\n", pgeom(4, p) - pgeom(0, p)))  # Success between 2nd-5th trial
cat(sprintf("P(3 < X < 7): %.4f\n", pgeom(5, p) - pgeom(2, p)))   # Strict range

# Special cases
cat(sprintf("P(X is odd): %.4f\n", sum(dgeom(seq(0, 98, 2), p))))  # Success on 1st, 3rd, 5th,... trial
cat(sprintf("P(X ∈ {1, 3, 5}): %.4f\n", sum(dgeom(c(0, 2, 4), p))))  # Specific trial numbers

# ****************************************************************************************************************

# Negative Binomial distribution
# Number of trials until r-th success (attempts until 5th success, time until r failures)
r <- 2; p <- 0.4   # Number of successes, probability of success

# Interpretation:
#  - P(X = 10) = Probability 3rd success occurs on 10th trial
#  - P(X ≤ 8) = Probability 3rd success occurs by 8th trial

cat("\n* Negative binomial *\n")

# Exact point probability (PMF)
cat(sprintf("P(X=5 failures): %.4f\n", dnbinom(5, r, p)))  # Exactly 5 failures before rth success

# Cumulative probabilities (CDF)
cat(sprintf("P(X ≤ 4): %.4f\n", pnbinom(4, r, p)))        # rth success within 4 failures
cat(sprintf("P(X > 6): %.4f\n", 1 - pnbinom(6, r, p)))    # Needs 7+ failures for rth success
cat(sprintf("P(X ≥ 5): %.4f\n", 1 - pnbinom(4, r, p)))    # Alternative to P(X>4)

# Range probabilities
cat(sprintf("P(2 ≤ X ≤ 7): %.4f\n", pnbinom(7, r, p) - pnbinom(1, r, p)))  # rth success between 2-7 failures
cat(sprintf("P(3 < X < 8): %.4f\n", pnbinom(7, r, p) - pnbinom(3, r, p)))   # Strict range

# Special cases
cat(sprintf("P(X is even): %.4f\n", sum(dnbinom(seq(0, 98, 2), r, p))))  # Even number of failures
cat(sprintf("P(X ∈ {1, 3, 5}): %.4f\n", sum(dnbinom(c(1, 3, 5), r, p))))   # Specific failure counts
cat(sprintf("P(total trials ≤ 6): %.4f\n", pnbinom(3, r, p)))               # r successes + ≤3 failures

# ****************************************************************************************************************

# Poisson 
# Count of rare events in fixed time/space (emails per hour, accidents per day, mutations per genome)
lambda <- 4.0  # Average rate of events (λ)

# Interpretations:
#  - P(X=2) = Exactly 2 calls arrive in an hour
#  - P(X ≥ 4) = 4 or more customers enter the store

cat("\n* Poisson *\n")

# Exact point probability (PMF)
cat(sprintf("P(X=2 events): %.4f\n", dpois(2, lambda)))  # Exactly 2 events occurring

# Cumulative probabilities (CDF)
cat(sprintf("P(X ≤ 3): %.4f\n", ppois(3, lambda)))        # 3 or fewer events
cat(sprintf("P(X > 5): %.4f\n", 1 - ppois(5, lambda)))    # 6+ events
cat(sprintf("P(X < 5): %.4f\n", ppois(4, lambda)))        # 4 or fewer events
cat(sprintf("P(X ≥ 4): %.4f\n", 1 - ppois(3, lambda)))    # Alternative to P(X>3)

# Range probabilities
cat(sprintf("P(2 ≤ X ≤ 5): %.4f\n", ppois(5, lambda) - ppois(1, lambda)))  # Between 2-5 events
cat(sprintf("P(3 < X < 7): %.4f\n", ppois(6, lambda) - ppois(3, lambda)))   # Strict range

# Special cases
cat(sprintf("P(X is even): %.4f\n", sum(dpois(seq(0, 48, 2), lambda))))  # Even number of events
cat(sprintf("P(X ∈ {1, 3, 5}): %.4f\n", sum(dpois(c(1, 3, 5), lambda))))  # Specific event counts
cat(sprintf("P(μ-σ ≤ X ≤ μ+σ): %.4f\n", ppois(lambda + sqrt(lambda), lambda) - ppois(lambda - sqrt(lambda) - 1, lambda)))  # Within 1 SD of mean

# ****************************************************************************************************************

# =====================
# CONTINUOUS DISTRIBUTIONS
# =====================
cat("\n=== CONTINUOUS ===\n")

# Uniform distribution
# Equal probability across a range (random number generation, modeling uncertainty with no prior information)
a <- 2.0; b <- 5.0  # Lower and upper bounds

# Interpretation:
#  - P(2.5 ≤ X ≤ 4.5) = Probability value falls between 2.5 and 4.5
#  - P(X > 4.2) = Probability value exceeds 4.2

cat("\n* Uniform *\n")

# Cumulative probabilities (CDF) - For continuous distributions, < and ≤ give same results
cat(sprintf("P(2.5 ≤ X ≤ 4.5): %.4f\n", punif(4.5, a, b) - punif(2.5, a, b)))
cat(sprintf("P(X ≤ 3.5): %.4f\n", punif(3.5, a, b)))
cat(sprintf("P(X > 4.2): %.4f\n", 1 - punif(4.2, a, b)))
cat(sprintf("P(X < 4.2): %.4f\n", punif(4.2, a, b)))

# ****************************************************************************************************************

# Exponential distribution
# Time between events, waiting times, failure rates (time until next customer arrival, component lifespan)
lambda <- 0.5  # Rate parameter (lambda)

# Interpretation:
#  - P(1.0 ≤ X ≤ 3.0) = Probability waiting time is between 1 and 3 units
#  - P(X > 1.5) = Probability waiting time exceeds 1.5 units

cat("\n* Exponential *\n")
cat(sprintf("P(1.0 ≤ X ≤ 3.0): %.4f\n", pexp(3.0, lambda) - pexp(1.0, lambda)))
cat(sprintf("P(X ≤ 2.0): %.4f\n", pexp(2.0, lambda)))
cat(sprintf("P(X > 1.5): %.4f\n", 1 - pexp(1.5, lambda)))

# ****************************************************************************************************************

# Normal distribution
# Natural phenomena with central tendency (heights, test scores, measurement errors, many real-world variables)
mu <- 0.0; sigma <- 1.0  # Mean and standard deviation

# Interpretation:
#  - P(-1.0 ≤ X ≤ 1.0) = Probability value falls within 1 standard deviation of mean
#  - P(X > 1.96) = Probability value exceeds 1.96 

cat("\n* Normal *\n")
cat(sprintf("P(-1.0 ≤ X ≤ 1.0): %.4f\n", pnorm(1.0, mu, sigma) - pnorm(-1.0, mu, sigma)))
cat(sprintf("P(X ≤ 0.5): %.4f\n", pnorm(0.5, mu, sigma)))
cat(sprintf("P(X > 1.96): %.4f\n", 1 - pnorm(1.96, mu, sigma)))

# ****************************************************************************************************************

# Beta distribution
# Proportions/percentages bounded between 0 and 1 (success rates, market share, probability estimates)
alpha <- 2.0; beta_param <- 3.0  # Shape parameters

# Interpretation:
#  - P(0.2 ≤ X ≤ 0.7) = Probability proportion falls between 20% and 70%
#  - P(X > 0.6) = Probability proportion exceeds 60%

cat("\n* Beta *\n")
cat(sprintf("P(0.2 ≤ X ≤ 0.7): %.4f\n", pbeta(0.7, alpha, beta_param) - pbeta(0.2, alpha, beta_param)))
cat(sprintf("P(X ≤ 0.5): %.4f\n", pbeta(0.5, alpha, beta_param)))
cat(sprintf("P(X > 0.6): %.4f\n", 1 - pbeta(0.6, alpha, beta_param)))

# ****************************************************************************************************************

# Gamma distribution
# Positive continuous values, often waiting times for multiple events (time for k events to occur, rainfall amounts)
shape <- 2.0; scale <- 1.5  # Shape and scale parameters

# Interpretation:
#  - P(1.0 ≤ X ≤ 4.0) = Probability event time is between 1 and 4 units
#  - P(X > 3.0) = Probability event time exceeds 3 units

cat("\n* Gamma *\n")
cat(sprintf("P(1.0 ≤ X ≤ 4.0): %.4f\n", pgamma(4.0, shape, scale=1/scale) - pgamma(1.0, shape, scale=1/scale)))
cat(sprintf("P(X ≤ 2.5): %.4f\n", pgamma(2.5, shape, scale=1/scale)))
cat(sprintf("P(X > 3.0): %.4f\n", 1 - pgamma(3.0, shape, scale=1/scale)))

# ****************************************************************************************************************

# Student's t distribution
# Small sample hypothesis testing, confidence intervals when population variance unknown (t-tests, robust alternative to normal)
df <- 5  # Degrees of freedom

# Interpretation:
#  - P(-1.5 ≤ X ≤ 1.5) = Probability standardized value falls between -1.5 and 1.5
#  - P(X > 2.0) = Probability standardized value exceeds 2.0

cat("\n* Student's t *\n")
cat(sprintf("P(-1.5 ≤ X ≤ 1.5): %.4f\n", pt(1.5, df) - pt(-1.5, df)))
cat(sprintf("P(X ≤ 1.0): %.4f\n", pt(1.0, df)))
cat(sprintf("P(X > 2.0): %.4f\n", 1 - pt(2.0, df)))