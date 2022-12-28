**Initial set up/Exploratory Data Analysis**

The goal in this phase was to ingest the csv containing the training data and begin developing the features that will enable an effective predictive model build. To perform the analysis and model testing required to select the best approach to the problem, several packages were used. We build the bulk of our analysis using the package Caret. 

```{r warning=FALSE}
setwd("C:/Users/maria/OneDrive/Desktop/UVA Data Science/DS6030_Stat_learning/Rapid Response Project")
library(ggplot2)
library(dplyr)
library(tidyverse)
library(caret)
library(ROCR)
library(knitr)
library(doParallel)
library(ggpubr)
library(GGally)
```

The dataset used for this analysis includes four variables:
-	Class: a categorical variable defining 5 categories of land classification whether that be vegetation, soil, rooftop, various non-tarp and blue-tarp. 
-	Red: numerical variable containing the value of red pixels for a given land classification category; 
-	Green: numerical variable containing the value of green pixels for a given land classification category;
-	Blue:  numerical variable containing the value of blue pixels for a given land classification category. 
To this four variables, a new binary variable was added, BlueTarp, to determine whether the pixels is “BlueTarp” or not. 


```{r}
no_cores <- 8
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)

data<- read.table("HaitiPixels.csv", sep=",", header=TRUE)
data$BlueTarp<-ifelse(data$Class == "Blue Tarp","Yes","No")
data$BlueTarp<-factor(data$BlueTarp, levels = c("No", "Yes"))
```

As shown in the density plot below (Fig.1) each land classification category has a distinct spectral signature when it comes to its RGB spectrum. For example, we notice that objects classified as vegetation present high concentrations of pixels in the blue category, and an equal number of pixels in the green and the red category. The soil category on the other hand, show a sharp pick in the red pixels. For the purpose of this project, we will be focusing only on the Blue Tarp category since we are interested in building a model able to differentiate blue tarp from non-blue tarp objects. From the Blue Tarp pixel density plot we notice that the blue pixel has values above 200 with the most recurring value being 255. Concerning the red pixel, we notice that most of the pixel values are comprised between ~120 and ~200. 

```{r message=FALSE, warning=FALSE, error=FALSE, fig.height = 5, fig.width = 9, fig.cap="Figure 1. Density plot for each land classification category."}
vegetation<-data%>%filter(Class=="Vegetation")%>%
  ggplot(aes(x=Green)) +
  geom_density(color='green')+
  geom_density(aes(x=Red,color='red'),show.legend =TRUE)+
  geom_density(aes(x=Blue),color='blue')+ 
  scale_color_manual(values = c(green = "green",red = "red", blue= "blue"),labels=c(green = "Green pixels",red = "Red pixels",blue = "Blue pixels"))+
  labs(x="Vegetation",title="Vegetation pixel density plot")+
  scale_x_continuous(limits = c(0, 255))

soil<-data%>%filter(Class=="Soil")%>%
  ggplot(aes(x=Green)) +
  geom_density(color='green')+
  geom_density(aes(x=Red,color='red'),show.legend =TRUE)+
  geom_density(aes(x=Blue),color='blue')+ 
  scale_color_manual(values = c(green = "green",red = "red", blue= "blue"),labels=c(green = "Green pixels",red = "Red pixels",blue = "Blue pixels"))+
  labs(x="Soil",title="Soil pixel density plot")+
  scale_x_continuous(limits = c(0, 255))

rooftop<-data%>%filter(Class=="Rooftop")%>%
  ggplot(aes(x=Green)) +
  geom_density(color='green')+
  geom_density(aes(x=Red,color='red'),show.legend =TRUE)+
  geom_density(aes(x=Blue),color='blue')+ 
  scale_color_manual(values = c(green = "green",red = "red", blue= "blue"),labels=c(green = "Green pixels",red = "Red pixels",blue = "Blue pixels"))+
  labs(x="Rooftop",title="Rooftop pixel density plot")+
  scale_x_continuous(limits = c(0, 255))

bluetarp<-data%>%filter(Class=="Blue Tarp")%>%
  ggplot(aes(x=Green)) +
  geom_density(color='green')+
  geom_density(aes(x=Red,color='red'),show.legend =TRUE)+
  geom_density(aes(x=Blue),color='blue')+ 
  scale_color_manual(values = c(green = "green",red = "red", blue= "blue"),labels=c(green = "Green pixels",red = "Red pixels",blue = "Blue pixels"))+
  labs(x="Blue Tarp",title="Blue Tarp pixel density plot")+
  scale_x_continuous(limits = c(0, 255))

nontarp<-data%>%filter(Class=="Various Non-Tarp")%>%
  ggplot(aes(x=Green)) +
  geom_density(color='green')+
  geom_density(aes(x=Red,color='red'),show.legend =TRUE)+
  geom_density(aes(x=Blue),color='blue')+ 
  scale_color_manual(values = c(green = "green",red = "red", blue= "blue"),labels=c(green = "Green pixels",red = "Red pixels",blue = "Blue pixels"))+
  labs(x="Various Non-Tarp",title="Various Non-Tarp pixel density plot")+
  scale_x_continuous(limits = c(0, 255))

ggarrange(vegetation,soil,rooftop,nontarp, bluetarp,common.legend = TRUE, legend = "right")
```

The figure below (Fig.2) provides an overview of the relationship between the different pixel colors for the Blue Tarp category. As observed before and as confirmed by the scatterplot, the blue and the green pixels seems to be representative of this land classification category, and as observed in the scatterplot, these two-color categories seem to increase concurrently. We can hypothesize the presence of a possible interaction between the two variables, but we are not going to explore that possibility in the context of this analysis. 

```{r cache = TRUE, fig.height = 5, fig.width = 9, fig.cap="Figure 2. Different plots showing the possible interaction between different colors."}
ggpairs(data[2:5],aes(color=BlueTarp,alpha = 0.5))
```

In general, as the RGB values trended to the higher range, a cleaner split between the "blue tarps" and the other classes (rooftop, soil, etc.) becomes clearer. The low end of the RGB values in the data points, however, have considerably more overlap suggesting that linear models and otherwise less flexible choices may not prevail for our data (Fig. 3). The above plots show that it is unlikely to find a Blue Tarp pixel with a high Red value, but not as unlikely for a pixel with a high Green value. A possible reason for this is that Green and Blue are more visually related than Red and Blue.

```{r fig.cap="Figure 3. Scatterplot of pixel values divided by category."}
library(plotly)
color_3d <-plot_ly(data, x=~Red, y=~Green, z=~Blue,
   color=~Class, colors = c('#0E74EB', '#F0CD62', '#F09E7B', '#B1C1E3', '#54AD21'),size=1)
color_3d<-color_3d%>%add_markers()
color_3d
```
