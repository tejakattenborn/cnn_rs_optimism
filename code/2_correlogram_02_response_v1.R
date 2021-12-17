

############## Description

# calculate correlogram for the response variables, that is the species cover derived per image tile
# Manuscript: Kattenborn et al. > Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks
# teja dot kattenborn at uni uni minus leipzig dot de


############## Code


require(raster)
require(foreach)
require(doParallel)
require(lattice)
require(ncf)

no_cores = 16


setwd("PATH")

dirs = list.dirs("input_msk", recursive = F, full.names = F)

cl<-makeCluster(no_cores)
registerDoParallel(cl)

vals_all <- foreach(dir = 1:length(dirs)) %dopar% {
  require(raster)

  masks = list.files(paste0("input_msk/",dirs[dir],"/256"), full.names = T)
  meta = read.csv(paste0("input_img_jpeg/",dirs[dir],"/256/metadataXYpos.csv"))
  
  vals = matrix(0, nrow = dim(meta)[1], ncol = 18)
  colnames(vals) = c("x", "y", 1:16)
  
  vals[,1] = (meta$xmin + meta$xmax)/2
  vals[,2] = (meta$ymin + meta$ymax)/2
  
  for(tiles in 1:dim(meta)[1]){
        msk = raster(masks[tiles])
    msk_vals = getValues(msk)
    msk_vaks_perc = table(msk_vals)/length(msk_vals)
    vals[tiles,as.numeric(names(msk_vaks_perc))+2] = msk_vaks_perc
  }
  return(vals)
}

stopCluster(cl)

vals_all_mrg = do.call(rbind, vals_all)
vals_all_mrg = vals_all_mrg[,1:14] # remove empty colunms)
vals_all_mrg = as.data.frame(vals_all_mrg)

###### calc correlogram

dresp = vals_all_mrg[sample(1:nrow(vals_all_mrg), 20000),]

coresp <- correlog(x = dresp$x, y = dresp$y, z = dresp[,3:ncol(dresp)], increment=1, resamp=10) # check autocorrleation, increment ~ bin size
save(coresp, file = "correlog_output_coresp_resp_v2.RData")
