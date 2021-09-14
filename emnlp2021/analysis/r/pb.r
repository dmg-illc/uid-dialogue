library(lmerTest)
library(lme4)

# Load Maptask measurements
pb <- read.csv('~/code/erp-paper/emnlp2021/dialogue-data/held-out/photobook-gpt2-finetuned.csv')
chains <- read.csv('~/code/erp-paper/emnlp2021/dialogue-data/held-out/photobook-chains-gpt2-finetuned.csv')


# Name variables
pb$logh <- log(pb$xu_h)
pb$logpdial <- log(pb$position_in_dialogue)
pb$logpround <- log(pb$position_in_round)

chains$logh <- log(chains$xu_h)
chains$logpchain <- log(chains$position_in_chain)


# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logpdial + (1 + logpdial | dialogue_id), pb)
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logpdial + (1 + logpdial | dialogue_id)
   # Data: pb

# REML criterion at convergence: 45408.9

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -4.2759 -0.6193 -0.0146  0.6911  4.3517 

# Random effects:
 # Groups      Name        Variance Std.Dev. Corr 
 # dialogue_id (Intercept) 0.030681 0.17516       
             # logpdial    0.001419 0.03767  -0.61
 # Residual                0.141826 0.37660       
# Number of obs: 49013, groups:  dialogue_id, 750

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)  -0.122147   0.009001 751.683355  -13.57   <2e-16 ***
# logpdial      0.031286   0.002351 729.402766   13.31   <2e-16 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
         # (Intr)
# logpdial -0.801
# optimizer (nloptwrap) convergence code: 0 (OK)
# Model failed to converge with max|grad| = 0.00994282 (tol = 0.002, component 1)



# --------------- Position in round ---------------
m <- lmer(logh ~ 1 + logpround + (1 + logpround | dialogue_id), pb)
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logpround + (1 + logpround | dialogue_id)
   # Data: pb

# REML criterion at convergence: 45713

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -4.3060 -0.6118 -0.0119  0.6906  4.3311 

# Random effects:
 # Groups      Name        Variance Std.Dev. Corr 
 # dialogue_id (Intercept) 0.022918 0.15139       
             # logpround   0.001198 0.03462  -0.36
 # Residual                0.142994 0.37815       
# Number of obs: 49013, groups:  dialogue_id, 750

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)   
# (Intercept)  -0.009907   0.006988 757.040027  -1.418  0.15665   
# logpround    -0.007347   0.002576 739.946257  -2.853  0.00445 **
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
          # (Intr)
# logpround -0.627



# --------------- Position in chain ---------------
m <- lmer(logh ~ 1 + logpchain + (1 + logpchain | dialogue_id), chains)
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logpchain + (1 + logpchain | dialogue_id)
   # Data: chains

# REML criterion at convergence: 22821.6

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -5.2524 -0.6392  0.0043  0.6566  4.1079 

# Random effects:
 # Groups      Name        Variance Std.Dev. Corr 
 # dialogue_id (Intercept) 0.022071 0.14856       
             # logpchain   0.002213 0.04704  -0.03
 # Residual                0.083824 0.28952       
# Number of obs: 49945, groups:  dialogue_id, 2453

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept) -5.919e-02  3.624e-03  2.446e+03 -16.331   <2e-16 ***
# logpchain    1.270e-02  2.748e-03  2.395e+03   4.622    4e-06 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
          # (Intr)
# logpchain -0.413

