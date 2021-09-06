library(lmerTest)
library(lme4)

# Load Maptask measurements
mt <- read.csv('~/code/erp-paper/emnlp2021/dialogue-data/held-out/maptask-gpt2-finetuned.csv')

# Keep only information-transmission acts
acts <- list('instruct', 'explain', 'check', 'query_yn', 'reply_w', 'clarify')
mt <- mt[mt$move_type %in% acts,]

# Name variables
mt$logh <- log(mt$xu_h)
mt$logp <- log(mt$position_in_dialogue)
mt$logt <- log(mt$position_in_transaction)

# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logp + (1 + logp | dialogue_id), mt)
summary(m)
# REML criterion at convergence: 867.4

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.4652 -0.6549 -0.0042  0.6290  3.5346 

# Random effects:
 # Groups      Name        Variance Std.Dev. Corr 
 # dialogue_id (Intercept) 0.020235 0.14225       
             # logp        0.000298 0.01726  -0.91
 # Residual                0.071941 0.26822       
# Number of obs: 3708, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)
# (Intercept)  0.037460   0.031175 34.732108   1.202    0.238
# logp        -0.001551   0.005496 28.623613  -0.282    0.780

# Correlation of Fixed Effects:
     # (Intr)
# logp -0.908

# --------------- Position in transaction ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt | dialogue_id), mt)
summary(m)
# REML criterion at convergence: 848.6

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.5062 -0.6536 -0.0089  0.6272  3.6009 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 7.070e-03 0.084081      
             # logt        5.614e-05 0.007493 -0.49
 # Residual                7.169e-02 0.267756      
# Number of obs: 3708, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)    
# (Intercept) -0.014191   0.016813 36.933189  -0.844    0.404    
# logt         0.024150   0.004879 27.064134   4.950 3.46e-05 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.594
# optimizer (nloptwrap) convergence code: 0 (OK)
# Model failed to converge with max|grad| = 0.00490771 (tol = 0.002, component 1)

# --------------- Position in transaction [follower] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|dialogue_id), mt[mt$speaker == 'f',])
summary(m)
# REML criterion at convergence: 492

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.2706 -0.6469  0.0023  0.6292  3.4291 

# Random effects:
 # Groups      Name        Variance Std.Dev. Corr 
 # dialogue_id (Intercept) 0.019289 0.13888       
             # logt        0.002407 0.04906  -0.80
 # Residual                0.081521 0.28552       
# Number of obs: 1257, groups:  dialogue_id, 38

# Fixed effects:
            # Estimate Std. Error       df t value Pr(>|t|)  
# (Intercept)  0.08353    0.03301 26.68540   2.531   0.0176 *
# logt        -0.01334    0.01351 21.43071  -0.987   0.3345  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.859

# --------------- Position in transaction [giver] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|dialogue_id), mt[mt$speaker == 'g',])
summary(m)
# REML criterion at convergence: 327.9

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.5692 -0.6400 -0.0190  0.6266  3.8016 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 6.770e-03 0.082279      
             # logt        4.844e-06 0.002201 -0.71
 # Residual                6.459e-02 0.254141      
# Number of obs: 2451, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)    
# (Intercept) -0.039502   0.017047 39.275281  -2.317   0.0258 *  
# logt         0.034867   0.005296 38.021472   6.583 9.04e-08 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.571
# optimizer (nloptwrap) convergence code: 0 (OK)
# Model failed to converge with max|grad| = 0.0142069 (tol = 0.002, component 1)


