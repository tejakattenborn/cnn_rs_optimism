require(lattice)
require(ncf)

wdir = "D:/data_ortho_confobi_sac"
setwd(wdir)

meta_files = list.files("1_results_256_200lv_250epoch_latent/",full.names = T, pattern = "CFB", recursive = T)

rm(meta_all)
for (i in 1:length(meta_files)){
  if(exists("meta_all")==F){
    meta_all = read.csv(meta_files[i])
    meta_all$X.1 = i
  }else{
    meta_all_tmp = read.csv(meta_files[i])
    meta_all_tmp$X.1 = i
    meta_all = rbind(meta_all,meta_all_tmp)
  }
}

meta_all = as.data.frame(meta_all)
meta_all$xmin = rowMeans(cbind(meta_all$xmax, meta_all$xmin))
meta_all$ymin = rowMeans(cbind(meta_all$ymax, meta_all$ymin))
meta_all = meta_all[, !(colnames(meta_all) %in% c("xmax","ymax", "X"))]
colnames(meta_all)[1:3] = c("plot_id", "x", "y")


#write.csv(meta_all, "1_results_latentvariables/meta_all.csv", row.names = FALSE)



### subsetting / settings
dtest = meta_all[sample(1:nrow(meta_all), 20000),]
#dtest = meta_all[meta_all$plot_id==c(plot),]  # plots where different varilog estimators deviate strongly: 1, 12 29, 30


########################################################
### Correlogram

# test spline.correlog?
par(mfrow = c(3,1))

cotest <- correlog(x = dtest$x, y = dtest$y, z = dtest[,4:ncol(dtest)], increment=1, resamp=10) # check autocorrleation, increment ~ bin size

#save(cotest, file = "correlog_output_copred_v2.RData")
load(file = "correlog_output_copred_v2.RData")

plot(cotest$mean.of.class, cotest$correlation, type = "p", ylim = c(-0.1,0.15), # plot correlogram
     ylab="Moran Similarity", xlab="averaged distance class", main="a) Corrlog: all bins", pch=20)
abline(h=0, col="red")
#lines(cotest$mean.of.class,cotest$p-0.2, col="grey")

plot(cotest$mean.of.class[cotest$n>5000], cotest$correlation[cotest$n>5000], type = "p", ylim = c(-0.1,0.15), # plot correlogram
     ylab="Moran Similarity", xlab="averaged distance class", main="b) Corrlog: bins with No. points > 5000", pch=20)
abline(h=0, col="red")

plot(cotest$mean.of.class[cotest$n>5000], cotest$correlation[cotest$n>5000], type = "p", ylim = c(-0.1,0.15), # plot correlogram
     ylab="Moran Similarity", xlab="averaged distance class", main="c) Corrlog: bins with No. points > 5000, log scale", log="x", pch=20)
abline(h=0, col="red")


# lines(cotest$mean.of.class, colMeans(cotest_allm, na.rm = T), lwd=2, col="red")
# lines(cotest$mean.of.class, apply(cotest_allm, 2, quantile, probs = 0.25, na.rm = T), lwd=2, col="orange")
# lines(cotest$mean.of.class, apply(cotest_allm, 2, quantile, probs = 0.75, na.rm = T), lwd=2, col="orange")

# similarity vs distance
mean(cotest$correlation[cotest$mean.of.class>150])
mean(cotest$correlation[cotest$mean.of.class<150])
mean(cotest$correlation[cotest$mean.of.class<20])
mean(cotest$correlation[cotest$mean.of.class<10])

# compare plateaus (20-150 vs 150 to Inf)
ttest = t.test(cotest$correlation[cotest$mean.of.class>20 & cotest$mean.of.class<150], cotest$correlation[cotest$mean.of.class>150])
ttest$p.value
ttest$statistic



# for two plot figure


pdf("fig_sac_rgb_v1.pdf", width=8, height=5)
par(mfrow = c(1,2))
par(mar=c(5.1,4.1,4.1,0.1)) # bottom, left, top and right

plot(cotest$mean.of.class[cotest$mean.of.class < 100], cotest$correlation[cotest$mean.of.class < 100], type = "p", log="",
     ylim = c(-0.1,0.11), # plot correlogram
     ylab="Multivariate correlation", xlab="averaged distance class [m]", bty="n")
#grid()
abline(h=mean(cotest$correlation[cotest$mean.of.class<150 & cotest$mean.of.class>20]), col="red", lwd=2, lty=2)
#abline(h=mean(cotest$correlation[cotest$mean.of.class>150]), col="blue", lwd=2, lty=2)
legend("bottomleft", legend = c("mean > 20 m & < 120 m", "mean > 150 m"),
       lty = c(2,2), col = c("red", "blue"), lwd=c(2,2),
       bty="n")

par(mar=c(5.1,0.1,4.1,0.1)) 
plot(cotest$mean.of.class[cotest$n>5000 & cotest$mean.of.class>150], cotest$correlation[cotest$n>5000 & cotest$mean.of.class>150],type = "p",
     ylim = c(-0.1,0.11), # plot correlogram
     ylab="", xlab="averaged distance class", bty="n", yaxt = "n")
#abline(h=mean(cotest$correlation[cotest$mean.of.class<150 & cotest$mean.of.class>20]), col="red", lwd=2, lty=2)
abline(h=mean(cotest$correlation[cotest$mean.of.class>150]), col="blue", lwd=2, lty=2)
#grid()
dev.off()
#abline(h=0, col="red")
#lines(cotest$mean.of.class,cotest$p-0.2, col="grey")


#devtools::install_github("LKremer/ggpointdensity")
library(ggpointdensity)
library(ggplot2)
library(viridis)
require(ggpubr)

cotest_df = data.frame(dist = cotest$mean.of.class, cor = cotest$correlation)
cotest_df = cotest_df[cotest$n > 4000,]

p1 = ggplot(data = cotest_df, mapping = aes(x = dist, y = cor)) +
  geom_pointdensity(adjust = 1) +
  xlim(0, 100) +
  ylim(-0.1, 0.1) +
  xlab("distance [m]") + 
  ylab("Multivariate correlation") +
  ggtitle(label = "SAC within UAV flights") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_color_viridis(begin=0, end=0)

p2 = ggplot(data = cotest_df, mapping = aes(x = dist, y = cor)) +
  geom_pointdensity(adjust = 1.0) +
  #xlim(150, max(cotest_df$dist)) +
  xlim(150, 60000) +
  ylim(-0.1, 0.1) +
  theme_minimal() +
  xlab("distance [m]") + 
  ylab("") +
  #ylab("Multivariate correlation") + 
  ggtitle(label = "SAC accross UAV flights") +
  scale_color_viridis() +
  #geom_hline(yintercept = mean(cotest_df$cor[cotest_df$dist > 150])) + 
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        plot.title = element_text(hjust = 0.5)
        #, legend.position = "none"
  )

p_both = ggarrange(p1, p2,
                   #labels = c("SAC", "noSAC"),
                   ncol = 2, nrow = 1,
                   widths = c(0.7,1))

p_both

pdf("fig_sac_ggplot.pdf",
    width=7, height=2.8)
p_both
dev.off()


# coefficent of var https://stackoverflow.com/questions/13195442/moving-variance-in-r


par(mfrow = c(1,2))
fit3 = spline.correlog(x = dtest$x, y = dtest$y, z = dtest[,4:ncol(dtest)], resamp=10, np=1000)
plot(fit3, ylim=c(0-0.05, 0.1), xlim=c(0,100))
plot(fit3, ylim=c(0-0.05, 0.1))



########################################################
### Variogram ###


data_geoR = as.geodata(dtest, coords.col = 2:3, data.col = 4:ncol(dtest))
# calc semi-variance
dtmp = seq(0,100,1) # define distance intervals (bis jetzt teste ich v.a. < 150, eignetlich will ich auch zeigen, dass SAC zwischen Plots geringer ist als in Plots)
bin <- variog(data_geoR, uvec = dtmp, max.dist = max(dtmp), estimator.type="classical")
env <-variog.mc.env(data_geoR, obj.variog = bin)



plot(bin, envelope = env, pch=21, bg="grey", xlab="Distance (m)", ylab="Semivariance", main = "Semivarigoram on selected variable", xlim=c(0,100))














#require(geoR)
data_geoR = as.geodata(dtest, coords.col = 2:3, data.col = 4:ncol(dtest))
# calc semi-variance
dtmp = seq(0,100,1) # define distance intervals (bis jetzt teste ich v.a. < 150, eignetlich will ich auch zeigen, dass SAC zwischen Plots geringer ist als in Plots)
bin <- variog(data_geoR, uvec = dtmp, max.dist = max(dtmp), estimator.type="classical")

plot(bin$u, rowSums(bin$v), pch=21, bg="grey", xlab="Distance (m)", ylab="Semivariance",
     main = "Variog estimator 'classical", xlim=c(0,max(dtmp)))
#, ylim=c(min(bin$v), max(bin$v)))
grid()

bin <- variog(data_geoR, uvec = dtmp, max.dist = max(dtmp), estimator.type="modulus")
# plot 1) latent variable in space and 2) semi-variance
plot(bin$u, rowSums(bin$v), pch=21, bg="grey", xlab="Distance (m)", ylab="Semivariance",
     main = "Variog estimator 'Hawkins and Cressie'", xlim=c(0,max(dtmp)))
grid()
#, ylim=c(min(bin$v), max(bin$v)))












dtest = meta_all[meta_all$plot_id==c(plot),]
# multivariate
cotest <- correlog(x = dtest$x, y = dtest$y, z = dtest[, c(4:ncol(dtest))], increment=1, resamp=10) # check autocorrleation, increment ~ bin size
plot(cotest$mean.of.class, cotest$correlation, type = "o", ylim = c(-0.5,0.5), # plot correlogram
     ylab="Moran Similarity", xlab="averaged distance class")
lines(cotest$mean.of.class,cotest$p-0.3)
abline(h=0, col="red")
grid()
plot = plot+1

#non centered
cotestnc <- correlog.nc(x = dtest$x, y = dtest$y, z = dtest[, c(4:ncol(dtest))], increment=1, resamp=10) 
plot(cotestnc$mean.of.class, cotestnc$correlation, type = "o", ylim = c(-1,1), # plot correlogram
     ylab="Moran Similarity", xlab="averaged distance class")
lines(cotestnc$mean.of.class,cotestnc$p-0.7)


lv = 4
cotest <- correlog(x = dtest$x, y = dtest$y, z = dtest[,lv], increment=0.02, resamp=1)
cotest_all = list()
for(i in 5:(ncol(meta_all))){
  cotest_all[[i-4]] <- correlog(x = dtest$x, y = dtest$y, z = dtest[,i], increment=0.02, resamp=1)$correlation
  print(i)
}
cotest_allm = do.call(rbind, cotest_all)

plot(cotest$mean.of.class, cotest$correlation, type = "l", ylim = c(-1,1), # plot correlogram
     ylab="Moran Similarity", xlab="averaged distance class")

lines(cotest$mean.of.class, colMeans(cotest_allm, na.rm = T), lwd=2, col="red")
lines(cotest$mean.of.class, apply(cotest_allm, 2, quantile, probs = 0.25, na.rm = T), lwd=2, col="orange")
lines(cotest$mean.of.class, apply(cotest_allm, 2, quantile, probs = 0.75, na.rm = T), lwd=2, col="orange")
abline(h=0, lty =2)













