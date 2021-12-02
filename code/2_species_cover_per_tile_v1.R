



# script to derive canopy cover per species per tile

library(raster)
library(foreach)

setwd("D:/data_ortho_confobi_sac/")
root_dir = "input_msk"
plot_dir = list.dirs(root_dir, recursive = F)



cover_all = list()

cl <- parallel::makeCluster(6)
doParallel::registerDoParallel(cl)

cover_all = foreach(plot = 1:length(plot_dir)) %dopar% {
  #for(plot in 1:length(plot_dir)){#
  library(raster)
  
  msks = list.files(plot_dir[plot], full.names = T, pattern = "png", recursive = T)
  cover = matrix(NA, nrow = length(msks), ncol=15)
  colnames(cover) = c("plot_id", (1:ncol(cover)-1)[-1])
  cover[,1] = plot # add plot ID to table
  
  for(tile in 1:nrow(cover)){
    
    msk = raster(msks[tile])
    msk_vals = table(getValues(msk))
    cover[tile,(as.numeric(names(msk_vals))+1)] = as.numeric(msk_vals)
  }
  
  return(cover)
  #print(paste0("plot-id ", plot)); flush.console()
}

parallel::stopCluster(cl)

save(cover_all, file="speciescover_all_plots.RData")
cover_allm = do.call(rbind, cover_all)[,1:13]/256*256 # normalize to cover [%*0.01] and remove unecessary col






#### load latent variables
meta_files = list.files("1_results_latentvariables/",full.names = T, pattern = "CFB", recursive = T)
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


#check dims
dim(meta_all)
dim(cover_allm)

# identify tiles that are mainly covered by species X and derive their latent variable space (sd, mean)
min_frac = 0.7
sel_dom = function(spec){which(cover_allm[,spec]>min_frac)}

means = matrix(NA, nrow = dim(cover_allm)[2]-1, ncol=dim(meta_all)[2]-3)
sds = matrix(NA, nrow = dim(cover_allm)[2]-1, ncol=dim(meta_all)[2]-3)

for(species in 2:(dim(cover_allm)[2]-1)){
  for(lv in 4:ncol(meta_all)){
    means[species-1, lv-3] = mean(meta_all[sel_dom(species),lv])
    sds[species-1, lv-3] = sd(meta_all[sel_dom(species),lv])
  }
}

plot(means[1,], type="l", ylim=c(0.05,0.08), xlim=c(0,50)) # only 50 first latent variables
for(species in 2:nrow(means)){
  lines(means[species,], col=species)
}



library(keras)
library(tensorflow)
library(tfdatasets)
# library(dplyr)
# library(ggplot2)
#library(glue)
# require(data.table)
#require(abind) # binding RGB arrays
require(countcolors)

optimizer <- tf$keras$optimizers$Adam(1e-4)
lv = 200
batch_size = 10
n_img = 100 # image per species to simulate

checkpoint <-  tf$train$Checkpoint(optimizer = optimizer,
                                   encoder = encoder,
                                   decoder = decoder)
status= checkpoint$restore(tf$train$latest_checkpoint("D:/data_ortho_confobi_sac/1_results_256_200latent/checkpoints_cvae")) # checkpoint file can also be modified to point not to the latest but another epoch
status$assert_existing_objects_matched()




lv_spec = function(species){rnorm(lv, mean = means[species,] ,sd=sds[species,])}

for(img_no in 1:n_img*dim(cover_allm)[2]){
  
  
  pred = as.array(decoder(matrix(lv_spec(species), nrow=1)))
  writeJPEG(pred, target = paste0(species, "_", img_no), quality = 0.7, bg = "white", color.space)
  
    if (img_no %% 100 == 0) {
    species = species + 1
  }
  
}



# most frequent species (globally)
# tiles with cover > 70 of frequent species --> latent space (sd and mean) of species-specific tiles)