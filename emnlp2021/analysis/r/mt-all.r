library(lmerTest)
library(lme4)

# Load Maptask measurements
mt <- read.csv('~/code/erp-paper/emnlp2021/dialogue-data/held-out/maptask-gpt2-finetuned.csv')

# Name variables
mt$logh <- log(mt$xu_h)
mt$logp <- log(mt$position_in_dialogue)
mt$logt <- log(mt$position_in_transaction)

# --------------- Position in dialogue ---------------
m <- lmer(logh ~ 1 + logp + (1 + logp | dialogue_id), mt)
summary(m)
# REML criterion at convergence: 3581.8

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.1671 -0.5860  0.0049  0.6334  3.6501 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 1.308e-02 0.114357      
             # logp        6.325e-05 0.007953 -1.00
 # Residual                9.028e-02 0.300464      
# Number of obs: 8002, groups:  dialogue_id, 38

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)  
# (Intercept)  7.584e-04  2.403e-02  4.339e+01   0.032    0.975  
# logp        -6.965e-03  3.666e-03  1.891e+02  -1.900    0.059 .
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logp -0.850
# optimizer (nloptwrap) convergence code: 0 (OK)
# boundary (singular) fit: see ?isSingular


# --------------- Position in transaction ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt | dialogue_id), mt)
summary(m)
# REML criterion at convergence: 3591.9

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.0897 -0.5836 -0.0033  0.6284  3.6560 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 6.246e-03 0.079033      
             # logt        1.134e-05 0.003368 -0.07
 # Residual                9.040e-02 0.300669      
# Number of obs: 8002, groups:  dialogue_id, 38

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)  
# (Intercept) -0.0295247  0.0149621 35.7615075  -1.973   0.0562 .
# logt        -0.0003339  0.0036877 24.3411560  -0.091   0.9286  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.456


# --------------- Position in transaction [follower] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|dialogue_id), mt[mt$speaker == 'f',])
summary(m)
# REML criterion at convergence: 1894

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -2.6811 -0.6085 -0.0081  0.6259  3.6811 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr 
 # dialogue_id (Intercept) 0.0118278 0.10876       
             # logt        0.0002234 0.01495  -0.46
 # Residual                0.0985658 0.31395       
# Number of obs: 3443, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)  
# (Intercept) -0.050101   0.023020 30.237543  -2.176   0.0375 *
# logt        -0.005976   0.007141 21.292784  -0.837   0.4120  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.673


# --------------- Position in transaction [giver] ---------------
m <- lmer(logh ~ 1 + logt + (1 + logt|dialogue_id), mt[mt$speaker == 'g',])
summary(m)
# REML criterion at convergence: 1563.1

# Scaled residuals: 
    # Min      1Q  Median      3Q     Max 
# -3.1908 -0.5835  0.0107  0.6077  3.6835 

# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr
 # dialogue_id (Intercept) 5.444e-03 0.073785     
             # logt        8.187e-05 0.009048 0.20
 # Residual                8.066e-02 0.284004     
# Number of obs: 4559, groups:  dialogue_id, 38

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)  
# (Intercept) -0.021988   0.014855 35.441968  -1.480   0.1477  
# logt         0.009192   0.004588 22.667386   2.004   0.0572 .
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Correlation of Fixed Effects:
     # (Intr)
# logt -0.427


