
require(rgdal)

setwd("D:/googledrive/paper/paper_cnn_spatial_autocorrelation/plots_statistics")



AOI = readOGR(dsn = getwd(), layer = "ConFoBio_plots")

AOI = AOI[!AOI$plot_id==0,] # remove hainich data


dist_matrix = dist(cbind(AOI$x, AOI$y))
mean(dist_matrix)/1000 # mean in km
