library(lme4)
library(lmerTest)

ptb <- read.csv('/Volumes/Disk1/erp/results_conll/mi/ptb_gpt2-ft.csv')

# Log transform variables
ptb$logh <- log(ptb$normalised_h)
ptb$loghdoc<- log(ptb$normalised_h_doc)

ptb$logpdoc <- log(ptb$position_in_doc)
ptb$logppar <- log(ptb$position_in_par)

ptb$loglen <- log(ptb$length)

# ================ Decontextualised information content ================

# -------------- Document --------------
m <- lmer(logh ~ 1 + logpdoc + loglen + (1 + logpdoc + loglen | doc_id), ptb)
summary(m)
# Random effects:
 # Groups   Name        Variance Std.Dev. Corr       
 # doc_id   (Intercept) 0.110478 0.33238             
          # logpdoc     0.001883 0.04340  -1.00      
          # loglen      0.005845 0.07645  -0.84  0.84
 # Residual             0.034605 0.18602             
# Number of obs: 6459, groups:  doc_id, 400

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   1.965761   0.024923 264.687129  78.873  < 2e-16 ***
# logpdoc       0.029391   0.003755 172.766329   7.827  4.8e-13 ***
# loglen       -0.124912   0.006357 274.144897 -19.651  < 2e-16 ***


# ================ Contextualised information content ================

# -------------- Document --------------
m <- lmer(loghdoc ~ 1 + logpdoc + loglen + (1 + logpdoc + loglen | doc_id), ptb)
summary(m)
# Random effects:
 # Groups   Name        Variance Std.Dev. Corr       
 # doc_id   (Intercept) 0.102346 0.31992             
          # logpdoc     0.001353 0.03678  -1.00      
          # loglen      0.005847 0.07646  -0.78  0.78
 # Residual             0.041595 0.20395             
# Number of obs: 6457, groups:  doc_id, 400

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   1.878079   0.025527 282.526885  73.571   <2e-16 ***
# logpdoc       0.002278   0.003757 148.692845   0.607    0.545    
# loglen       -0.107451   0.006674 269.137923 -16.101   <2e-16 ***


# ==================== Mutual information ====================

# -------------- Document --------------
m <- lmer(mi_doc ~ 1 + logpdoc + loglen + (1 + logpdoc + loglen | doc_id), ptb)
summary(m)
# Random effects:
 # Groups   Name        Variance Std.Dev. Corr       
 # doc_id   (Intercept) 0.344002 0.58652             
          # logpdoc     0.003311 0.05754   0.25      
          # loglen      0.026838 0.16382  -1.00 -0.22
 # Residual             0.157509 0.39687             
# Number of obs: 6457, groups:  doc_id, 400

# Fixed effects:
              # Estimate Std. Error         df t value Pr(>|t|)    
# (Intercept)   0.711482   0.047984 308.248059   14.83   <2e-16 ***
# logpdoc       0.120768   0.007067 385.884699   17.09   <2e-16 ***
# loglen       -0.172742   0.013434 310.981157  -12.86   <2e-16 ***
