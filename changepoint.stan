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
// Quadratic time solution
//transformed parameters {
//      vector[N] log_p;
//      real mu;
//      real sigma;
//      log_p = rep_vector(-log(N), N);
//      for (tau in 1:N)
//        for (i in 1:N) {
//          mu = i < tau ? mu1 : mu2;
//          sigma = i < tau ? sigma1 : sigma2;
//          log_p[tau] = log_p[tau] + normal_lpdf(D[i] | mu, sigma);
//      }
//}


// Linear time solution, as suggested by @idontgetoutmuch
// https://github.com/gmodena/bayesian-changepoint/issues/1
transformed parameters {
  vector[N] log_p;
  {
    vector[N+1] log_p_e;
    vector[N+1] log_p_l;
    log_p_e[1] = 0;
    log_p_l[1] = 0;
    for (i in 1:N) {
      log_p_e[i + 1] = log_p_e[i] + normal_lpdf(D[i] | mu1, sigma1);
      log_p_l[i + 1] = log_p_l[i] + normal_lpdf(D[i] | mu2, sigma2);
    }
    log_p = rep_vector(-log(N) + log_p_l[N + 1], N) + head(log_p_e, N) - head(log_p_l, N);
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
