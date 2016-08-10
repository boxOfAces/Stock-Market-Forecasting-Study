library("quantmod")
library("rpart")
library("rpart.plot")
library("randomForest")
library("e1071")

startDate = as.Date("2016-01-01")
endDate = as.Date("2016-01-22")
getSymbols("INFY.NS", src = "yahoo", from = startDate, to = endDate)
RSI3<-RSI(Op(INFY.NS), n= 3)
EMA5<-EMA(Op(INFY.NS),n=5)
EMAcross<- Op(INFY.NS)-EMA5

MACD<-MACD(Op(INFY.NS),fast = 12, slow = 26, signal = 9)
#Calculate a MACD with standard parameters
MACDsignal<-MACD[,2]
#Grab just the signal line to use as our indicator. 

SMI<-SMI(Op(INFY.NS),n=13,slow=25,fast=2,signal=9)
#Stochastic Oscillator with standard parameters
SMI<-SMI[,1]
#Grab just the oscillator to use as our indicator

PriceChange<- Cl(INFY.NS) - Op(INFY.NS)
#Calculate the difference between the close price and open price
Class<-ifelse(PriceChange>0,"UP","DOWN")
#Create a binary classification variable, the variable we are trying to predict. 

#calculate proportion of UP days
class_bool = ifelse(Class == "UP",1,0)
prop_up = rollapply(class_bool, 5, mean)

DataSet<-data.frame(prop_up,RSI3,EMAcross,MACDsignal,SMI,Class)
#Create our data set
colnames(DataSet)<-c("PropUP", "RSI3","EMAcross","MACDsignal","Stochastic","Class")
#Name the columns
DataSet<-DataSet[-c(1:33),]
#Get rid of the data where the indicators are being calculated 

TrainingSet<-DataSet[1:1004,]
#Use 2/3 of the data to build the tree
TestSet<-DataSet[1004:1480,]
#And leave out 1/3 data to test our strategy 

DecisionTree<-rpart(Class~PropUP+RSI3+EMAcross+MACDsignal+Stochastic,data=TrainingSet, cp=.001)
#Specifying the indicators to we want to use to predict the class and 
#controlling the growth of the tree by setting the minimum amount of information gained (cp) needed to justify a split.

prp(DecisionTree,type=2,extra=8)
#Nice plotting tool with a couple parameters to make it look good. If you want to play around with the visualization yourself, here is a great resource. 

printcp(DecisionTree)
#shows the minimal cp for each trees of each size.
plotcp(DecisionTree,upper="splits")
#plots the average geometric mean for trees of each size.

PrunedDecisionTree<-prune(DecisionTree,cp=0.0079)
#I am selecting the complexity parameter (cp) that has the lowest cross-validated error (xerror) 

prp(PrunedDecisionTree, type=2, extra=8)

#random forests
#create the forest
randomForestModel<-randomForest(Class~PropUP+RSI3+EMAcross+MACDsignal+Stochastic,data=TrainingSet, importance=TRUE, ntree = 1000)

# View the forest results.
print(randomForestModel) 

# Importance of each predictor.
print(importance(randomForestModel,type = 2)) 
varImpPlot(randomForestModel,
           sort = T,
           main="Variable Importance",
           n.var=5)

#svm
svmModel <- svm(Class~PropUP+RSI3+EMAcross+MACDsignal+Stochastic,data = TrainingSet, gamma = 0.15, cost = 4)
tuneResult <- tune(svm, Class~PropUP+RSI3+EMAcross+MACDsignal+Stochastic,data=TrainingSet,
                   ranges = list(gamma = seq(0.05,0.3,0.01), cost = 2^(0:5))
)
print(tuneResult)
# best performance: MSE = 0.2848812, gamma 0.15 cost 4
# Draw the tuning graph
plot(tuneResult)

#evaluation
#decision tree
confusion = table(predict(PrunedDecisionTree,TestSet,type="class"),TestSet[,6],dnn=list('predicted','actual'))

#random forest
confusion = table(predict(randomForestModel,TestSet,type="class"),TestSet[,6],dnn=list('predicted','actual'))

#SVM
confusion = table(predict(svmModel,TestSet,type="class"),TestSet[,6],dnn=list('predicted','actual'))

sprintf("Precision: %f", confusion[1,1]/sum(confusion[1,1:2]))
sprintf("Recall: %f", confusion[1,1]/sum(confusion[1:2,1])) 
sprintf("Accuracy: %f", (confusion[1,1]+confusion[2,2])/sum(confusion[,])) 
