

############## Description

# Calculate the distance among the individual sites.
# Manuscript: Kattenborn et al. > Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks
# teja dot kattenborn at uni uni minus leipzig dot de


############## Code


require(rgdal)

setwd("PATH")

AOI = readOGR(dsn = getwd(), layer = "ConFoBio_plots")
AOI = AOI[!AOI$plot_id==0,] # remove hainich data

dist_matrix = dist(cbind(AOI$x, AOI$y))
mean(dist_matrix)/1000 # mean in km
