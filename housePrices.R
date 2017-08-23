library(DMwR)
#data(algae)
data1 <- read.csv("D:\\Kaggle\\train.csv", header=TRUE) #algae[-c(62, 199), ] # the 2 incomplete samples
#clean.algae <- knnImputation(algae, k=10) # lm() does not handle NAs!
la1 <- lm(data1$SalePrice ~ data1$OverallQual  + data1$GrLivArea + data1$GarageArea + data1$GarageCars + data1$MoSold, data = data1)

la1
summary(la1)

final.la1 <- step(la1)
testData <- read.csv("D:\\Kaggle\\test.csv", header=TRUE) 

preds <- predict(final.la1,testData)

length(preds)
X = append(preds, "Values", after = 0)#length(x))
ids = testData$Id
X = append(ids, "IDs", after = 0)#length(x))

for i in 0:1469
  if preds[i] < 0
    preds[i] = 100000

predictions = cbind(testData$Id, data.frame(preds))
predictions = cbind(X, data.frame(preds))
predictions
write.csv(predictions, 'D:\\Kaggle\\predictions.csv', row.names = F)
#data <- read.csv("D:\\Kaggle\\train.csv", header=TRUE)
#data2 <- data.frame(SalePrice = data$SalePrice , OverallQual = data$OverallQual, data$GrLivArea)
#data2 = subset(data2, data2$LotArea < 100000)

#plot(data2$OverallQual, data2$SalePrice)

#scatter3D(x = data2$OverallQual, y = data2$GrLivArea , z = data2$SalePrice)
