import numpy as np
from scipy import stats

# Probability distribution functions (*nothing to do with the acronym PDF below):
# -> DISCRETE
#      - Probability mass function (PMF - f(x))
#      - Cumulative distribution function (CDF - F(X))
# -> CONTINOUS
#      - Probability density function (PDF - f(x))
#      - Cumulative distribution function (CDF - F(X))
#
# Really good video explaining the concept:
#   https://www.youtube.com/watch?v=YXLVjCKVP7U in fact the dude has a great collection of videos on probability distributions
#   that I extremely recommend you to watch: https://www.zstatistics.com/videos#/distributions 

# =====================
# DISCRETE DISTRIBUTIONS
# =====================
print("=== DISCRETE ===")

# Bernoulli 
# Single trial with success/failure (coin flip, pass/fail test, yes/no survey response)
# By definition Bernoully has only two outputs (1=success, 0=failure)
# Interpretation:
#  - P(X = 1) = Probability of success on single trial
#  - P(X = 0) = Probability of failure on single trial
p = 0.3     # Probability of success
print("\n* Bernoulli *")
print(f"P(X=1): {stats.bernoulli.pmf(1, p):.4f}")  # Exact point
print(f"P(X=0): {stats.bernoulli.pmf(0, p):.4f}")

# ****************************************************************************************************************

# Binomial distribution
# Fixed number of independent Bernoulli trials (number of heads in 10 coin flips, defective items in batch)
n, p = 10, 0.4   # Number of trials, probability of success

# Interpretation:
#  - P(X = 3) = Probability of exactly 3 successes in 10 trials
#  - P(X ≤ 2) = Probability of 2 or fewer successes

print("\n* Binomial *")

# Exact point probability
print(f"P(X=3): {stats.binom.pmf(3, n, p):.4f}")  # Exactly 3 successes

# Range probabilities
print(f"P(3 < X ≤ 5): {stats.binom.cdf(5, n, p) - stats.binom.cdf(3, n, p):.4f}")  # 4 or 5 successes
print(f"P(2 ≤ X < 4): {stats.binom.cdf(3, n, p) - stats.binom.cdf(1, n, p):.4f}")   # 2 or 3 successes

# Tail probabilities
print(f"P(X > 4): {1 - stats.binom.cdf(4, n, p):.4f}")     # 5+ successes
print(f"P(X ≥ 6): {1 - stats.binom.cdf(5, n, p):.4f}")     # 6+ successes (strict)
print(f"P(X < 2): {stats.binom.cdf(1, n, p):.4f}")         # 0 or 1 success
print(f"P(X ≤ 3): {stats.binom.cdf(3, n, p):.4f}")         # 0-3 successes

# Special cases
print(f"P(X ≠ 4): {1 - stats.binom.pmf(4, n, p):.4f}")     # All except 4 successes
print(f"P(X ∈ {{2,5,7}}): {sum(stats.binom.pmf([2,5,7], n, p)):.4f}")  # Specific values

# ****************************************************************************************************************

# Geometric distribution
# Number of trials until first success (attempts until first sale, rolls until first six)
p = 0.2  # Probability of success on each trial

print("\n* Geometric *")

# Interpretations: 
#  - P(2 ≤ X ≤ 5) = Success occurs on trial 2, 3, 4, or 5
#  - P(X > 4) = First success takes 5+ trials

# Exact point probability (PMF)
print(f"P(X=5): {stats.geom.pmf(5, p):.4f}")  # Exactly 5 trials needed for 1st success

# Cumulative probabilities (CDF)
print(f"P(X ≤ 3): {stats.geom.cdf(3, p):.4f}")  # Success within 1st 3 trials
print(f"P(X > 4): {1 - stats.geom.cdf(4, p):.4f}")  # Needs 5+ trials
print(f"P(X ≥ 3): {1 - stats.geom.cdf(2, p):.4f}")  # Alternative to P(X>2)

# Range probabilities 
print(f"P(2 ≤ X ≤ 5): {stats.geom.cdf(5, p) - stats.geom.cdf(1, p):.4f}")  # Success between 2nd-5th trial
print(f"P(3 < X < 7): {stats.geom.cdf(6, p) - stats.geom.cdf(3, p):.4f}")   # Strict range

# Special cases
print(f"P(X is odd): {sum(stats.geom.pmf(range(1, 100, 2), p)):.4f}")  # Success on 1st, 3rd, 5th,... trial
print(f"P(X ∈ {{1, 3, 5}}): {sum(stats.geom.pmf([1, 3, 5], p)):.4f}")  # Specific trial numbers

# ****************************************************************************************************************

# Negative Binomial distribution
# Number of trials until r-th success (attempts until 5th success, time until r failures)
r, p = 2, 0.4   # Number of successes, probability of success

# Interpretation:
#  - P(X = 10) = Probability 3rd success occurs on 10th trial
#  - P(X ≤ 8) = Probability 3rd success occurs by 8th trial

print("\n* Negative binominal *")

# Exact point probability (PMF)
print(f"P(X=5 failures): {stats.nbinom.pmf(5, r, p):.4f}")  # Exactly 5 failures before 3rd success

# Cumulative probabilities (CDF)
print(f"P(X ≤ 4): {stats.nbinom.cdf(4, r, p):.4f}")        # 3rd success within 4 failures
print(f"P(X > 6): {1 - stats.nbinom.cdf(6, r, p):.4f}")    # Needs 7+ failures for 3rd success
print(f"P(X ≥ 5): {1 - stats.nbinom.cdf(4, r, p):.4f}")    # Alternative to P(X>4)

# Range probabilities
print(f"P(2 ≤ X ≤ 7): {stats.nbinom.cdf(7, r, p) - stats.nbinom.cdf(1, r, p):.4f}")  # 3rd success between 2-7 failures
print(f"P(3 < X < 8): {stats.nbinom.cdf(7, r, p) - stats.nbinom.cdf(3, r, p):.4f}")   # Strict range

# Special cases
print(f"P(X is even): {sum(stats.nbinom.pmf(range(0, 100, 2), r, p)):.4f}")  # Even number of failures
print(f"P(X ∈ {{1, 3, 5}}): {sum(stats.nbinom.pmf([1, 3, 5], r, p)):.4f}")   # Specific failure counts
print(f"P(total trials ≤ 6): {stats.nbinom.cdf(3, r, p):.4f}")               # r=3 successes + ≤3 failures


# ****************************************************************************************************************

# 5. Poisson 
# Count of rare events in fixed time/space (emails per hour, accidents per day, mutations per genome)
lam = 4.0  # Average rate of events (λ)

# Interpretations:
#  - P(X=2) = Exactly 2 calls arrive in an hour
#  - P(X ≥ 4) = 4 or more customers enter the store

print("\n* Poisson *")

# Exact point probability (PMF)
print(f"P(X=2 events): {stats.poisson.pmf(2, lam):.4f}")  # Exactly 2 events occurring

# Cumulative probabilities (CDF)
print(f"P(X ≤ 3): {stats.poisson.cdf(3, lam):.4f}")        # 3 or fewer events
print(f"P(X > 5): {1 - stats.poisson.cdf(5, lam):.4f}")    # 6+ events
print(f"P(X ≥ 4): {1 - stats.poisson.cdf(3, lam):.4f}")    # Alternative to P(X>3)

# Range probabilities
print(f"P(2 ≤ X ≤ 5): {stats.poisson.cdf(5, lam) - stats.poisson.cdf(1, lam):.4f}")  # Between 2-5 events
print(f"P(3 < X < 7): {stats.poisson.cdf(6, lam) - stats.poisson.cdf(3, lam):.4f}")   # Strict range

# Special cases
print(f"P(X is even): {sum(stats.poisson.pmf(range(0, 50, 2), lam)):.4f}")  # Even number of events
print(f"P(X ∈ {{1, 3, 5}}): {sum(stats.poisson.pmf([1, 3, 5], lam)):.4f}")  # Specific event counts
print(f"P(μ-σ ≤ X ≤ μ+σ): {stats.poisson.cdf(lam + np.sqrt(lam), lam) - stats.poisson.cdf(lam - np.sqrt(lam) - 1, lam):.4f}")  # Within 1 SD of mean


# ****************************************************************************************************************

# =====================
# CONTINUOUS DISTRIBUTIONS
# =====================
print("\n=== CONTINUOUS ===")

# Uniform distribution
# Equal probability across a range (random number generation, modeling uncertainty with no prior information)
a, b = 2.0, 5.0  # Lower and upper bounds

# Interpretation:
#  - P(2.5 ≤ X ≤ 4.5) = Probability value falls between 2.5 and 4.5
#  - P(X > 4.2) = Probability value exceeds 4.2

print("\n* Uniform *")

# Cumulative probabilities (CDF) - For continuous distributions, < and ≤ give same results
print(f"P(2.5 ≤ X ≤ 4.5): {stats.uniform.cdf(4.5, a, b-a) - stats.uniform.cdf(2.5, a, b-a):.4f}")
print(f"P(X ≤ 3.5): {stats.uniform.cdf(3.5, a, b-a):.4f}")
print(f"P(X > 4.2): {1 - stats.uniform.cdf(4.2, a, b-a):.4f}")

# ****************************************************************************************************************

# Exponential distribution
# Time between events, waiting times, failure rates (time until next customer arrival, component lifespan)
lam = 0.5  # Rate parameter (lambda)

# Interpretation:
#  - P(1.0 ≤ X ≤ 3.0) = Probability waiting time is between 1 and 3 units
#  - P(X > 1.5) = Probability waiting time exceeds 1.5 units

print("\n* Exponential *")
print(f"P(1.0 ≤ X ≤ 3.0): {stats.expon.cdf(3.0, scale=1/lam) - stats.expon.cdf(1.0, scale=1/lam):.4f}")
print(f"P(X ≤ 2.0): {stats.expon.cdf(2.0, scale=1/lam):.4f}")
print(f"P(X > 1.5): {1 - stats.expon.cdf(1.5, scale=1/lam):.4f}")

# ****************************************************************************************************************

# Normal distribution
# Natural phenomena with central tendency (heights, test scores, measurement errors, many real-world variables)
mu, sigma = 0.0, 1.0  # Mean and standard deviation

# Interpretation:
#  - P(-1.0 ≤ X ≤ 1.0) = Probability value falls within 1 standard deviation of mean
#  - P(X > 1.96) = Probability value exceeds 1.96 

print("\n* Normal *")
print(f"P(-1.0 ≤ X ≤ 1.0): {stats.norm.cdf(1.0, mu, sigma) - stats.norm.cdf(-1.0, mu, sigma):.4f}")
print(f"P(X ≤ 0.5): {stats.norm.cdf(0.5, mu, sigma):.4f}")
print(f"P(X > 1.96): {1 - stats.norm.cdf(1.96, mu, sigma):.4f}")

# ****************************************************************************************************************

# Beta distribution
# Proportions/percentages bounded between 0 and 1 (success rates, market share, probability estimates)
alpha, beta_param = 2.0, 3.0  # Shape parameters

# Interpretation:
#  - P(0.2 ≤ X ≤ 0.7) = Probability proportion falls between 20% and 70%
#  - P(X > 0.6) = Probability proportion exceeds 60%

print("\n* Beta *")
print(f"P(0.2 ≤ X ≤ 0.7): {stats.beta.cdf(0.7, alpha, beta_param) - stats.beta.cdf(0.2, alpha, beta_param):.4f}")
print(f"P(X ≤ 0.5): {stats.beta.cdf(0.5, alpha, beta_param):.4f}")
print(f"P(X > 0.6): {1 - stats.beta.cdf(0.6, alpha, beta_param):.4f}")

# ****************************************************************************************************************

# Gamma distribution
# Positive continuous values, often waiting times for multiple events (time for k events to occur, rainfall amounts)
shape, scale = 2.0, 1.5  # Shape and scale parameters

# Interpretation:
#  - P(1.0 ≤ X ≤ 4.0) = Probability event time is between 1 and 4 units
#  - P(X > 3.0) = Probability event time exceeds 3 units

print("\n* Gamma *")
print(f"P(1.0 ≤ X ≤ 4.0): {stats.gamma.cdf(4.0, shape, scale=scale) - stats.gamma.cdf(1.0, shape, scale=scale):.4f}")
print(f"P(X ≤ 2.5): {stats.gamma.cdf(2.5, shape, scale=scale):.4f}")
print(f"P(X > 3.0): {1 - stats.gamma.cdf(3.0, shape, scale=scale):.4f}")

# ****************************************************************************************************************

# Student's t distribution
# Small sample hypothesis testing, confidence intervals when population variance unknown (t-tests, robust alternative to normal)RetryClaude can make mistakes. Please double-check responses.
df = 5  # Degrees of freedom

# Interpretation:
#  - P(-1.5 ≤ X ≤ 1.5) = Probability standardized value falls between -1.5 and 1.5
#  - P(X > 2.0) = Probability standardized value exceeds 2.0

print("\n* Student's t *")
print(f"P(-1.5 ≤ X ≤ 1.5): {stats.t.cdf(1.5, df) - stats.t.cdf(-1.5, df):.4f}")
print(f"P(X ≤ 1.0): {stats.t.cdf(1.0, df):.4f}")
print(f"P(X > 2.0): {1 - stats.t.cdf(2.0, df):.4f}")