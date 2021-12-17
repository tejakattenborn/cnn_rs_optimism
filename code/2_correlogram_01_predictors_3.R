

############## Description

# calculate correlogram for the predictor variables based on the latent space (derived from the variational autoencoder)
# Manuscript: Kattenborn et al. > Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks
# teja dot kattenborn at uni uni minus leipzig dot de


############## Code


require(lattice)
require(ncf)

wdir = "PATH"
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

### subsetting / settings
dtest = meta_all[sample(1:nrow(meta_all), 20000),]

########################################################
### Correlogram

cotest <- correlog(x = dtest$x, y = dtest$y, z = dtest[,4:ncol(dtest)], increment=1, resamp=10) # check autocorrleation, increment ~ bin size
save(cotest, file = "correlog_output_copred_v2.RData")
