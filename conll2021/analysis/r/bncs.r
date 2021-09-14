library(lme4)
library(lmerTest)

bncs <- read.csv('/Volumes/Disk1/erp/results_conll/mi/bnc_spoken_gpt2-ft.csv')

# Log transform variables
bncs$logh <- log(bncs$normalised_h)
bncs$loghdial<- log(bncs$normalised_h_doc)
bncs$logp <- log(bncs$position)
bncs$loglen <- log(bncs$length)


# ================ Decontextualised information content ================

# -------------- Dialogue --------------
m <- lmer(logh ~ 1 + logp + loglen + (1 + logp + loglen | dialogue_id), bncs)
summary(m)
# Random effects:
 # Groups      Name        Variance  Std.Dev. Corr       
 # dialogue_id (Intercept) 0.0206180 0.14359             
             # logp        0.0007347 0.02710  -0.66      
             # loglen      0.0014627 0.03825  -0.67 -0.06
 # Residual                0.0824793 0.28719             
# Number of obs: 13638, groups:  dialogue_id, 187

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)  1.813e+00  1.468e-02  1.869e+02 123.445   <2e-16 ***
# logp        -5.296e-04  3.369e-03  1.849e+02  -0.157    0.875    
# loglen      -8.019e-02  3.518e-03  1.752e+02 -22.790   <2e-16 ***


# ================ Contextualised information content ================

# -------------- Dialogue --------------
m <- lmer(loghdial ~ 1 + logp + loglen + (1 + logp + loglen| dialogue_id), bncs)
summary(m)
# Random effects:
 # Groups      Name        Variance Std.Dev. Corr       
 # dialogue_id (Intercept) 0.058185 0.24122             
             # logp        0.003593 0.05994  -0.78      
             # loglen      0.004173 0.06460  -0.92  0.47
 # Residual                0.242158 0.49210             
# Number of obs: 13634, groups:  dialogue_id, 187

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   1.729034   0.024917 196.882824  69.393  < 2e-16 ***
# logp         -0.028877   0.006404 211.115715  -4.509 1.08e-05 ***
# loglen       -0.051150   0.005973 194.828520  -8.563 3.30e-15 ***


# ==================== Mutual information ====================

# -------------- Document --------------
m <- lmer(mi_doc ~ 1 + logp + loglen + (1 + logp + loglen | dialogue_id), bncs)
summary(m)
# Random effects:
 # Groups      Name        Variance Std.Dev. Corr       
 # dialogue_id (Intercept) 0.123449 0.35135             
             # logp        0.005650 0.07517  -0.64      
             # loglen      0.007534 0.08680  -0.92  0.29
 # Residual                1.331982 1.15412             
# Number of obs: 13634, groups:  dialogue_id, 187

# Fixed effects:
             # Estimate Std. Error        df t value Pr(>|t|)    
# (Intercept)   0.44581    0.04851 215.91284   9.189  < 2e-16 ***
# logp          0.06310    0.01224 270.56400   5.157 4.86e-07 ***
# loglen       -0.10421    0.01058 203.65398  -9.853  < 2e-16 ***