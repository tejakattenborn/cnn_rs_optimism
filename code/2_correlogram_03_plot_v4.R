

############## Description

# plot the correlograms of predictor and response variables within and across orthoimages
# Manuscript: Kattenborn et al. > Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks
# teja dot kattenborn at uni uni minus leipzig dot de


############## Code


require(lattice)
require(ncf)
require(roll)
require(zoo)

wdir = "PATH"
setwd(wdir)

range01 <- function(x){(x-mean(x))/(max(x[1:100])-mean(x))}


load(file = "correlog_output_copred_v2.RData")
load(file = "correlog_output_coresp_resp_v2.RData")

cotest$correlation = range01(cotest$correlation)
cotest_rmean = roll_mean(cotest$correlation[200:length(cotest$correlation)], width = 10000) #, min_obs = 1000)
cotest_rsd = roll_sd(cotest$correlation[200:length(cotest$correlation)], width = 1000, min_obs = 1, center = TRUE)

coresp$correlation = range01(coresp$correlation)
coresp_rmean = roll_mean(coresp$correlation[200:length(coresp$correlation)], width = 10000) #, min_obs = 1000)
coresp_rsd = roll_sd(coresp$correlation[200:length(coresp$correlation)], width = 1000, min_obs = 1, center = TRUE)

col_respl = rgb(0, 100, 255, max = 255, alpha = 200)
col_respp = rgb(0, 100, 255, max = 255, alpha = 50)
col_testl = rgb(180, 0, 40, max = 255, alpha = 200)
col_testp = rgb(180, 0, 40, max = 255, alpha = 50)


pdf("correlogram_pred_resp_in_one_v3.pdf", width=8, height=4.5)

par(mfrow=c(1,2))
plot(cotest$correlation[1:100], type="l", lwd = 4, col = col_testl,
     main = "a) SAC within orthoimages",
     xlab = "distance [m]", ylab = "rel. multivariate correlation",
     ylim=c(-0.7,1))
grid(lty=1)
lines(cotest$correlation[1:100], lwd = 4, col = col_testl)
lines(coresp$correlation[1:100], lwd = 2, col=col_respl,
      ylim = c(-0.7,1))

legend("bottomleft", c("image tiles (predictors)", "species cover (response)") , lwd = 4, col=c(col_testl, col_respl), bty = "n")

dftest <- data.frame(x =1:length(cotest_rmean),
                     F =cotest_rmean,
                     L =cotest_rmean-cotest_rsd,
                     U =cotest_rmean+cotest_rsd)
plot(dftest$x, dftest$F, ylim = c(-0.7,1), type = "l", yaxt = "n", ylab="", col=col_testl,
     main = "b) SAC between orthoimages",
     xlim = c(11000,70500))
polygon(c(dftest$x,rev(dftest$x)),c(dftest$L,rev(dftest$U)),col = col_testp, border = FALSE)

grid(lty=1)
dfresp <- data.frame(x =1:length(coresp_rmean),
                     F =coresp_rmean,
                     L =coresp_rmean-coresp_rsd,
                     U =coresp_rmean+coresp_rsd)
polygon(c(dfresp$x,rev(dfresp$x)),c(dfresp$L,rev(dfresp$U)),col = col_respp, border = FALSE)
lines(dfresp$x, dfresp$F, ylim = c(-0.7,1), type = "l", col=col_respl, lwd = 2,
      xlim = c(11000,70500))
lines(dftest$x, dftest$F, lwd = 4, col=col_testl)
dev.off()
