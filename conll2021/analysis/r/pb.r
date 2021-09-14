library(lme4)
library(lmerTest)

pb <- read.csv('/Volumes/Disk1/erp/results_conll/mi/pb_gpt2-ft.csv')

# Log transform variables
pb$logh <- log(pb$normalised_h)
pb$loghdial <- log(pb$normalised_h_doc)

pb$logp <- log(pb$position_in_dialogue)

pb$loglen <- log(pb$length)


# ================ Decontextualised information content ================

# -------------- Dialogue --------------
m <- lmer(logh ~ 1 + logp + loglen + (1 + logp + loglen | dialogue_id), pb)
summary(m)
# Random effects:
 # Groups      Name        Variance Std.Dev. Corr       
 # dialogue_id (Intercept) 0.033312 0.18251             
             # logp        0.001774 0.04212  -0.58      
             # loglen      0.003190 0.05648  -0.27 -0.07
 # Residual                0.113535 0.33695             
# Number of obs: 40917, groups:  dialogue_id, 750

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   1.786217   0.009751 665.951420  183.18   <2e-16 ***
# logp          0.040724   0.002449 738.314723   16.63   <2e-16 ***
# loglen       -0.180505   0.002839 671.286923  -63.59   <2e-16 ***


# ================ Contextualised information content ================

# -------------- Dialogue --------------
m <- lmer(loghdial ~ 1 + logp + loglen + (1 + logp + loglen | dialogue_id), pb)
summary(m)
# Random effects:
 # Groups      Name        Variance Std.Dev. Corr       
 # dialogue_id (Intercept) 0.035930 0.18955             
             # logp        0.001512 0.03889  -0.53      
             # loglen      0.004259 0.06526  -0.50  0.13
 # Residual                0.136562 0.36954             
# Number of obs: 40917, groups:  dialogue_id, 750

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   1.985685   0.010440 680.523183 190.191  < 2e-16 ***
# logp         -0.016255   0.002528 748.259196  -6.431 2.26e-10 ***
# loglen       -0.250080   0.003204 675.304442 -78.063  < 2e-16 ***
 

# ==================== Mutual information ====================

# -------------- Dialogue --------------
m <- lmer(mi_doc ~ 1 + logp + loglen + (1 + logp + loglen | dialogue_id), pb)
summary(m)
# Random effects:
 # Groups      Name        Variance Std.Dev. Corr       
 # dialogue_id (Intercept) 0.31267  0.5592              
             # logp        0.01794  0.1339   -0.63      
             # loglen      0.03976  0.1994   -0.96  0.38
 # Residual                0.71604  0.8462              
# Number of obs: 40917, groups:  dialogue_id, 750

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)  -1.088505   0.027226 787.804964  -39.98   <2e-16 ***
# logp          0.279352   0.006852 845.350902   40.77   <2e-16 ***
# loglen        0.354808   0.008780 741.913858   40.41   <2e-16 ***
