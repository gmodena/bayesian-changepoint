// Built with stan 2.11
data {
    int<lower=1> N;
    real D[N]; 
}

parameters {
    real mu1;
    real mu2;

    real<lower=0> sigma1;
    real<lower=0> sigma2;
}

// Marginalize out tau and
// calculate log_p(D | mu1, sd1, mu2, sd2)
// TODO: we can make this linear via dynamic programming
transformed parameters {
      vector[N] log_p;
      real mu;
      real sigma;
      log_p = rep_vector(-log(N), N);
      for (tau in 1:N)
        for (i in 1:N) {
          mu = i < tau ? mu1 : mu2;
          sigma = i < tau ? sigma1 : sigma2;
          log_p[tau] = log_p[tau] + normal_lpdf(D[i] | mu, sigma);
      }
}

    
model {
    mu1 ~ normal(0, 10);
    mu2 ~ normal(0, 10);
    
    // scale parameters need to be > 0;
    // we constrained sigma1, sigma2 to be positive
    // so that stan interprets the following as half-normal priors
    sigma1 ~ normal(0, 10);
    sigma2 ~ normal(0, 10);
    
    target += log_sum_exp(log_p);
} 

//Draw the discrete parameter tau. This is highly inefficient
generated quantities {
    int<lower=1,upper=N> tau;
    tau = categorical_rng(softmax(log_p));
}
