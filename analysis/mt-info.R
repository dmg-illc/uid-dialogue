library(lmerTest)
library(lme4)

# Load Maptask measurements
mt <- read.csv('~/code/erp-paper/emnlp2021/dialogue-data/held-out/maptask-gpt2-finetuned.csv')

# Keep only information-transmission acts
acts <- list('instruct', 'explain', 'check', 'query_yn', 'reply_w', 'clarify', 'query_w', 'reply_yn')
mt <- mt[mt$move_type %in% acts,]

# Name variables
mt$logh <- log(mt$xu_h)
mt$logp <- log(mt$position_in_dialogue)
mt$logt <- log(mt$position_in_transaction)

# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logp + (1 + logp | dialogue_id), mt)
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logp + (1 + logp | dialogue_id)
   # Data: mt

# REML criterion at convergence: 916.5

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.4834 -0.6487 -0.0060  0.6248  3.5282 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 0.0226402 0.15047       
             # logp        0.0003937 0.01984  -0.93
 # Residual                0.0719902 0.26831       
# Number of obs: 3921, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)
# (Intercept) 3.372e-02  3.196e-02 3.568e+01   1.055    0.298
# logp        3.032e-04  5.659e-03 3.025e+01   0.054    0.958

# Correlation of Fixed Effects:
     # (Intr)
# logp -0.919


# --------------- Position in transaction ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt | dialogue_id), mt)
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logt + (1 + logt | dialogue_id)
   # Data: mt

# REML criterion at convergence: 899.5

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.5227 -0.6547 -0.0087  0.6219  3.5943 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 6.363e-03 0.079768      
             # logt        9.297e-05 0.009642 -0.31
 # Residual                7.180e-02 0.267957      
# Number of obs: 3921, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)    
# (Intercept) -0.009274   0.016145 36.333064  -0.574    0.569    
# logt         0.023757   0.004927 26.614648   4.822 5.09e-05 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.573


# --------------- Position in transaction [giver] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|dialogue_id), mt[mt$speaker == 'g',])
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logt + (1 + logt | dialogue_id)
   # Data: mt[mt$speaker == "g", ]

# REML criterion at convergence: 345.3

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.6171 -0.6480 -0.0169  0.6304  3.7951 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 6.724e-03 0.081998      
             # logt        3.575e-06 0.001891 -1.00
 # Residual                6.488e-02 0.254714      
# Number of obs: 2515, groups:  dialogue_id, 38

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   -0.03782    0.01697   42.90955  -2.229   0.0311 *  
# logt           0.03464    0.00525 2418.14179   6.598 5.09e-11 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.579


# --------------- Position in transaction [follower] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|dialogue_id), mt[mt$speaker == 'f',])
summary(m)
# Linear mixed model fit by REML. t-tests use Satterthwaite's method [
# lmerModLmerTest]
# Formula: logh ~ 1 + logt + (1 + logt | dialogue_id)
   # Data: mt[mt$speaker == "f", ]

# REML criterion at convergence: 519

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.3479 -0.6444  0.0028  0.6237  3.4625 

# Random effects:
 # Groups      Name        Variance Std.Dev. Corr 
 # dialogue_id (Intercept) 0.017171 0.1310        
             # logt        0.003025 0.0550   -0.79
 # Residual                0.079910 0.2827        
# Number of obs: 1406, groups:  dialogue_id, 38

# Fixed effects:
            # Estimate Std. Error       df t value Pr(>|t|)   
# (Intercept)  0.09036    0.03103 24.44255   2.912  0.00755 **
# logt        -0.01301    0.01379 22.43622  -0.943  0.35567   
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.859
