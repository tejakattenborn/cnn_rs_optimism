

############## Description

# Create statistics and figures of the optimism (inflated accuracy by spatially dependent data)
# Manuscript: Kattenborn et al. > Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks
# teja dot kattenborn at uni uni minus leipzig dot de


############## Code


require(ggplot2)
require(ggpubr)
library(tidyverse)


setwd("PATH")
no_plots = 47 # number of available orthoimage flights


all_runs_f1 = list.files("SAC_eval_fold1", recursive = F, full.names = T)
all_runs_f2 = list.files("SAC_eval_fold2", recursive = F, full.names = T)
all_runs_f3 = list.files("SAC_eval_fold3", recursive = F, full.names = T)
all_runs_f4 = list.files("SAC_eval_fold4", recursive = F, full.names = T)
all_runs_f5 = list.files("SAC_eval_fold5", recursive = F, full.names = T)


results_sac_f1 = matrix(NA, ncol=12, nrow=no_plots)
results_nosac_f1 = matrix(NA, ncol=12, nrow=no_plots)
results_sac_f2 = matrix(NA, ncol=12, nrow=no_plots)
results_nosac_f2 = matrix(NA, ncol=12, nrow=no_plots)
results_sac_f3 = matrix(NA, ncol=12, nrow=no_plots)
results_nosac_f3 = matrix(NA, ncol=12, nrow=no_plots)
results_sac_f4 = matrix(NA, ncol=12, nrow=no_plots)
results_nosac_f4 = matrix(NA, ncol=12, nrow=no_plots)
results_sac_f5 = matrix(NA, ncol=12, nrow=no_plots)
results_nosac_f5 = matrix(NA, ncol=12, nrow=no_plots)


for(iter in 1:length(all_runs_f1)){
  load(all_runs_f1[[iter]])
  results_sac_f1[,iter] = rowMeans(accuracies_testsac$F1_site_species, na.rm = TRUE)
  results_nosac_f1[,iter] = rowMeans(accuracies_testnosac$F1_site_species, na.rm = TRUE)
}
for(iter in 1:length(all_runs_f2)){
  load(all_runs_f2[[iter]])
  results_sac_f2[,iter] = rowMeans(accuracies_testsac$F1_site_species, na.rm = TRUE)
  results_nosac_f2[,iter] = rowMeans(accuracies_testnosac$F1_site_species, na.rm = TRUE)
}
for(iter in 1:length(all_runs_f3)){
  load(all_runs_f3[[iter]])
  results_sac_f3[,iter] = rowMeans(accuracies_testsac$F1_site_species, na.rm = TRUE)
  results_nosac_f3[,iter] = rowMeans(accuracies_testnosac$F1_site_species, na.rm = TRUE)
}
for(iter in 1:length(all_runs_f4)){
  load(all_runs_f4[[iter]])
  results_sac_f4[,iter] = rowMeans(accuracies_testsac$F1_site_species, na.rm = TRUE)
  results_nosac_f4[,iter] = rowMeans(accuracies_testnosac$F1_site_species, na.rm = TRUE)
}
for(iter in 1:length(all_runs_f5)){
  load(all_runs_f5[[iter]])
  results_sac_f5[,iter] = rowMeans(accuracies_testsac$F1_site_species, na.rm = TRUE)
  results_nosac_f5[,iter] = rowMeans(accuracies_testnosac$F1_site_species, na.rm = TRUE)
}

results_sac = list(results_sac_f1, results_sac_f2, results_sac_f3,results_sac_f4, results_sac_f5)
results_nosac = list(results_nosac_f1, results_nosac_f2, results_nosac_f3, results_nosac_f4, results_nosac_f5)
results_sac = apply(simplify2array(results_sac), 1:2, mean, na.rm=TRUE)
results_nosac = apply(simplify2array(results_nosac), 1:2, mean, na.rm=TRUE)



results_sac_m = data.frame(matrix(NA, nrow = no_plots*12, ncol=3))
colnames(results_sac_m) = c("score", "no_train", "aug")
results_sac_m$score =  as.vector(results_sac)
results_sac_m$no_train = c(rep("n = 10", 4*no_plots), rep("n = 25", 4*no_plots), rep("n = 40", 4*no_plots))
results_sac_m$aug = rep(c(rep("rad", no_plots), rep("geo", no_plots), rep("both", no_plots), rep("none", no_plots)), 3)

results_nosac_m = data.frame(matrix(NA, nrow = no_plots*12, ncol=3))
colnames(results_nosac_m) = c("score", "no_train", "aug")
results_nosac_m$score =  as.vector(results_nosac)
results_nosac_m$no_train = c(rep("n = 10", 4*no_plots), rep("n = 25", 4*no_plots), rep("n = 40", 4*no_plots))
results_nosac_m$aug = rep(c(rep("rad", no_plots), rep("geo", no_plots), rep("both", no_plots), rep("none", no_plots)), 3)



results_sac_m$aug <- factor(results_sac_m$aug,
                            levels = c('none','rad', 'geo', 'both'),ordered = TRUE)

results_nosac_m$aug <- factor(results_nosac_m$aug,
                              levels = c('none','rad', 'geo', 'both'),ordered = TRUE)



colors = c("#666666", "#5b9bd5", "#7030a0", "#ffd966")

p_sac = results_sac_m[complete.cases(results_sac_m),] %>%
  ggplot(aes(x=factor(aug),y=score, fill=aug)) +
  geom_boxplot() +
  geom_jitter(width=0.1,alpha=0.2) +
  xlab("") + 
  #ylab(expression("F"[1]*"-score")) + 
  ylab(expression("F"[1])) + 
  scale_fill_manual(values = colors) +
  facet_wrap(~no_train,ncol = 4) +
  ylim(0.2,1) +
  ggtitle(label = "non-independent test data") +
  guides(fill=guide_legend(title="data augmentation:")) + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom",
        plot.title = element_text(hjust = 0.5))


p_nosac = results_nosac_m[complete.cases(results_nosac_m),] %>%
  ggplot(aes(x=factor(aug),y=score, fill=aug)) +
  geom_boxplot() +
  geom_jitter(width=0.1,alpha=0.2) +
  xlab("") + 
  scale_fill_manual(values = colors) +
  facet_wrap(~no_train,ncol = 4) +
  ylim(0.2,1) +
  ggtitle(label = "independent test data") +
  guides(fill=guide_legend(title="data augmentation:")) + 
  
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position="bottom",
        plot.title = element_text(hjust = 0.5)
        #, legend.position = "none"
  )

p_both = ggarrange(p_sac, p_nosac,
                   #labels = c("SAC", "noSAC"),
                   ncol = 2, nrow = 1,
                   widths = c(1.1,1))
p_both
pdf("fig_performance_bias_v2.pdf", width=7, height=4)
p_both
dev.off()



# test if optimism is significantly different for augmentation, i.e. if the mean of augmentation is significantly lower/greater than when using none.

# mean across augmentation schemes for val-sac
n_10_sac_aug =  results_sac_m[results_sac_m$no_train == "n = 10" & results_sac_m$aug != "none", 1]
n_25_sac_aug =  results_sac_m[results_sac_m$no_train == "n = 25" & results_sac_m$aug != "none", 1]
n_40_sac_aug =  results_sac_m[results_sac_m$no_train == "n = 40" & results_sac_m$aug != "none", 1]

# mean across augmentation schemes for val-sac
n_10_nosac_aug =  results_nosac_m[results_nosac_m$no_train == "n = 10" & results_sac_m$aug != "none", 1]
n_25_nosac_aug =  results_nosac_m[results_nosac_m$no_train == "n = 25" & results_sac_m$aug != "none", 1]
n_40_nosac_aug =  results_nosac_m[results_nosac_m$no_train == "n = 40" & results_sac_m$aug != "none", 1]

# mean across augmentation schemes for val-sac
n_10_sac_noaug =  results_sac_m[results_sac_m$no_train == "n = 10" & results_sac_m$aug == "none", 1]
n_25_sac_noaug =  results_sac_m[results_sac_m$no_train == "n = 25" & results_sac_m$aug == "none", 1]
n_40_sac_noaug =  results_sac_m[results_sac_m$no_train == "n = 40" & results_sac_m$aug == "none", 1]

# mean across augmentation schemes for val-sac
n_10_nosac_noaug =  results_nosac_m[results_nosac_m$no_train == "n = 10" & results_sac_m$aug == "none", 1]
n_25_nosac_noaug =  results_nosac_m[results_nosac_m$no_train == "n = 25" & results_sac_m$aug == "none", 1]
n_40_nosac_noaug =  results_nosac_m[results_nosac_m$no_train == "n = 40" & results_sac_m$aug == "none", 1]


t.test(n_10_sac_aug - n_10_nosac_aug, n_10_sac_noaug - n_10_nosac_noaug,
       paired = FALSE, alternative = "two.sided", na.action = "na.omit")
t.test(n_25_sac_aug - n_25_nosac_aug, n_10_sac_noaug - n_25_nosac_noaug,
       paired = FALSE, alternative = "two.sided", na.action = "na.omit")
t.test(n_40_sac_aug - n_40_nosac_aug, n_10_sac_noaug - n_40_nosac_noaug,
       paired = FALSE, alternative = "two.sided", na.action = "na.omit")



# test if F-scores are significantly lower for the augmentation method, i.e. if the mean of augmentation='none' is significantly greater.

mean(results_nosac_m$score[results_nosac_m$aug == "geo"], na.rm=TRUE)
mean(results_nosac_m$score[results_nosac_m$aug == "both"], na.rm=TRUE)
mean(results_nosac_m$score[results_nosac_m$aug == "rad"], na.rm=TRUE)
mean(results_nosac_m$score[results_nosac_m$aug == "none"], na.rm=TRUE)

t.test(results_nosac_m$score[results_nosac_m$aug == "none"], results_nosac_m$score[results_nosac_m$aug == "geo"],
       paired = TRUE, alternative = "greater", na.action = "na.omit")
t.test(results_nosac_m$score[results_nosac_m$aug == "none"], results_nosac_m$score[results_nosac_m$aug == "rad"],
       paired = TRUE, alternative = "greater", na.action = "na.omit")
t.test(results_nosac_m$score[results_nosac_m$aug == "none"], results_nosac_m$score[results_nosac_m$aug == "both"],
       paired = TRUE, alternative = "greater", na.action = "na.omit")

mean(results_nosac_m$score[results_nosac_m$aug == "geo"], na.rm=TRUE)
mean(results_nosac_m$score[results_nosac_m$aug == "none"], na.rm=TRUE)

t.test(results_nosac_m$score[results_nosac_m$aug == "geo"], results_nosac_m$score[results_nosac_m$aug == "none"],
       paired = TRUE, alternative = "less", na.action = "na.omit")
t.test(results_nosac_m$score[results_nosac_m$aug == "rad"], results_nosac_m$score[results_nosac_m$aug == "none"],
       paired = TRUE, alternative = "less", na.action = "na.omit")
t.test(results_nosac_m$score[results_nosac_m$aug == "both"], results_nosac_m$score[results_nosac_m$aug == "none"],
       paired = TRUE, alternative = "less", na.action = "na.omit")



# test if F-scores are significantly lower for independent validation, i.e. if the mean is significantly greater.
t.test(results_sac_m$score, results_nosac_m$score,
       paired = TRUE, alternative = "greater", na.action = "na.omit")


# mean across augmentation schemes for val-sac
n_10_mean_sac =  mean(results_sac_m[results_sac_m$no_train == "n = 10", 1], na.rm = TRUE)
n_25_mean_sac =  mean(results_sac_m[results_sac_m$no_train == "n = 25", 1], na.rm = TRUE)
n_40_mean_sac =  mean(results_sac_m[results_sac_m$no_train == "n = 40", 1], na.rm = TRUE)

# mean across augmentation schemes for val-sac
n_10_mean_nosac =  mean(results_nosac_m[results_nosac_m$no_train == "n = 10", 1], na.rm = TRUE)
n_25_mean_nosac =  mean(results_nosac_m[results_nosac_m$no_train == "n = 25", 1], na.rm = TRUE)
n_40_mean_nosac =  mean(results_nosac_m[results_nosac_m$no_train == "n = 40", 1], na.rm = TRUE)

# optimism in f-score units
n_10_mean_sac - n_10_mean_nosac
n_25_mean_sac - n_25_mean_nosac
n_40_mean_sac - n_40_mean_nosac
n10 = n_10_mean_sac - n_10_mean_nosac
n25 = n_25_mean_sac - n_25_mean_nosac
n40 = n_40_mean_sac - n_40_mean_nosac

# optimism in f-score %
(n_10_mean_sac - n_10_mean_nosac) / n_10_mean_sac * 100
(n_25_mean_sac - n_25_mean_nosac) / n_25_mean_sac * 100
(n_40_mean_sac - n_40_mean_nosac) / n_40_mean_sac * 100
n10perc = (n_10_mean_sac - n_10_mean_nosac) / n_10_mean_sac * 100
n25perc = (n_25_mean_sac - n_25_mean_nosac) / n_25_mean_sac * 100
n40perc = (n_40_mean_sac - n_40_mean_nosac) / n_40_mean_sac * 100

plot(c(10,25,40), c(n10,n25,n40))
plot(c(10,25,40), c(n10perc,n25perc,n40perc))




