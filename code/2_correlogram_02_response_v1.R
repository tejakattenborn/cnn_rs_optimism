

require(raster)
require(foreach)
require(doParallel)
require(lattice)
require(ncf)

library(ggpointdensity)
library(ggplot2)
library(viridis)
require(ggpubr)

no_cores = 7

setwd("D:/data_ortho_confobi_sac/")



#############################################
# 1 RESPONSE
#############################################

dirs = list.dirs("input_msk", recursive = F, full.names = F)

cl<-makeCluster(no_cores)
registerDoParallel(cl)


vals_all <- foreach(dir = 1:length(dirs)) %dopar% {
  require(raster)
  #vals_all = list()
  #for(dir in 1:length(dirs)){
  
  masks = list.files(paste0("input_msk/",dirs[dir],"/256"), full.names = T)
  meta = read.csv(paste0("input_img_jpeg/",dirs[dir],"/256/metadataXYpos.csv"))
  
  vals = matrix(0, nrow = dim(meta)[1], ncol = 18)
  colnames(vals) = c("x", "y", 1:16)
  
  vals[,1] = (meta$xmin + meta$xmax)/2
  vals[,2] = (meta$ymin + meta$ymax)/2
  
  #for(tiles in 1:10){
  for(tiles in 1:dim(meta)[1]){
    
    msk = raster(masks[tiles])
    #image(msk)
    
    msk_vals = getValues(msk)
    msk_vaks_perc = table(msk_vals)/length(msk_vals)
    vals[tiles,as.numeric(names(msk_vaks_perc))+2] = msk_vaks_perc
    
  }
  
  return(vals)
  #vals_all = vals
  
  #als_all[[dir]] = vals
}

stopCluster(cl)

vals_all_mrg = do.call(rbind, vals_all)
#write.csv(vals_all_mrg, "5_calc_semivariogram_ref_tile_cover.csv")

vals_all_mrg = read.csv("5_calc_semivariogram_ref_tile_cover.csv")
vals_all_mrg = vals_all_mrg[,1:14] # remove empty colunms)
vals_all_mrg = as.data.frame(vals_all_mrg)

###### calc correlogram

dresp = vals_all_mrg[sample(1:nrow(vals_all_mrg), 20000),]
#dtest = meta_all[meta_all$plot_id==c(plot),]  # plots where different varilog estimators deviate strongly: 1, 12 29, 30

coresp <- correlog(x = dresp$x, y = dresp$y, z = dresp[,3:ncol(dresp)], increment=1, resamp=10) # check autocorrleation, increment ~ bin size
#save(coresp, file = "correlog_output_coresp_resp_v2.RData")
load(file = "correlog_output_coresp_resp_v2.RData")

coresp_df = data.frame(dist = coresp$mean.of.class, cor = coresp$correlation)
coresp_df = coresp_df[coresp$n > 4000,]

#range01 <- function(x){(x-min(x))/(max(x)-min(x))}
#range01 <- function(x){(x-mean(x))/(max(x)-mean(x))}



coresp_df$cor = range01(coresp_df$cor)

p1 = ggplot(data = coresp_df, mapping = aes(x = dist, y = cor)) +
  geom_pointdensity(adjust = 1) +
  xlim(0, 100) +
  ylim(-0.3, 1.0) +
  xlab("distance [m]") + 
  ylab("Multivariate correlation") +
  geom_hline(yintercept = mean(coresp_df$cor[coresp_df$dist < 150])) + 
  ggtitle(label = "SAC within UAV flights") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_color_viridis(begin=0, end=0)

p2 = ggplot(data = coresp_df, mapping = aes(x = dist, y = cor)) +
  geom_pointdensity(adjust = 1.0) +
  #xlim(150, max(copred_df$dist)) +
  xlim(150, 60000) +
  ylim(-0.3, 1.0) +
  theme_minimal() +
  xlab("distance [m]") + 
  ylab("") +
  #ylab("Multivariate correlation") + 
  ggtitle(label = "SAC accross UAV flights") +
  scale_color_viridis() +
  geom_hline(yintercept = mean(coresp_df$cor[coresp_df$dist > 150])) + 
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

pdf("fig_sac_ggplot_resp.pdf",
    width=7, height=2.8)
p_both
dev.off()




#############################################
# 2 PREDICTOR
#############################################

load(file = "correlog_output_copred_v2.RData")
copred = cotest
copred_df = data.frame(dist = copred$mean.of.class, cor = copred$correlation)
copred_df = copred_df[copred$n > 4000,]

#range01 <- function(x){(x-min(x))/(max(x)-min(x))}

copred_df$cor = range01(copred_df$cor)


p1 = ggplot(data = copred_df, mapping = aes(x = dist, y = cor)) +
  geom_pointdensity(adjust = 1) +
  xlim(0, 100) +
  ylim(-0.3, 1.0) +
  xlab("distance [m]") + 
  ylab("Multivariate correlation") +
  geom_hline(yintercept = mean(copred_df$cor[copred_df$dist < 150])) + 
  ggtitle(label = "SAC within UAV flights") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5)) +
  scale_color_viridis(begin=0, end=0)

p2 = ggplot(data = copred_df, mapping = aes(x = dist, y = cor)) +
  geom_pointdensity(adjust = 1.0) +
  #xlim(150, max(copred_df$dist)) +
  xlim(150, 60000) +
  ylim(-0.3, 1.0) +
  theme_minimal() +
  xlab("distance [m]") + 
  ylab("") +
  #ylab("Multivariate correlation") + 
  ggtitle(label = "SAC accross UAV flights") +
  scale_color_viridis() +
  geom_hline(yintercept = mean(copred_df$cor[copred_df$dist > 150])) + 
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

pdf("fig_sac_ggplot_pred.pdf",
    width=7, height=2.8)
p_both
dev.off()
