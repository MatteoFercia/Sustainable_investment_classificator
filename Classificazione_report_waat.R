### Obiettivi ###
### Analisi e sviluppo classificatore ###
#
# Creazione dataset
# Design classificatore (valutazione di almeno 2/3 algoritmi di classificazione)
# Evaluation, precision, recall e f1-score
### Clustering analysis ###
#
# Clustering dei paesi in funzione degli investimenti in green economy e social media 
# Valutazione di almeno 2/3 algoritmi di clustering

library(readr)
library(ggplot2)
library(ggrepel)
library(tidyverse)
library(corrplot)
library(caret)
dataset_finale <- read_csv("~/PycharmProjects/00192_00194_grosso_fercia/dataset_finale.csv", na= 'None')
View(dataset_finale)
summary(dataset_finale)
str(dataset_finale)

ds1 <- dataset_finale
View(ds1)
summary(ds1)

attach(ds1)
ds1 <- dataset_finale %>% mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))  ## Eliminazione degli NA
ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`, c('Negative', 'Positive'), c('0', '1')) ## Refactoring della response value

################################################################################
## ANALISI ESPLORATIVA
################################################################################
pairs(ds1[-c(1,17)])
## Mostra possibile relazioni tra tutte le variabili escluse Country e Risultato investimento; è possibile notare come ad una prima
## occhiata le uniche relazioni interessanti riguardino quelle tra il sentiment dei tweet ed il totale, anche se questa variabile è 
## semplicemente la somma delle tre e di conseguenza la forte relazione tra esse è il risultato di questa procedura. Con il corrplot
## si cercherà di analizzare in maggiore dettaglio la situazione.

corrplot(cor(ds1[-c(1,17,14:16)]), type = "upper", tl.pos = "td",
         method = "pie", tl.cex = 0.5, tl.col = 'black', diag = FALSE ) 
## Dal corrplot è possibile osservare una forte correlazione positiva tra alcune variabili: tralasciando quella tra il sentiment dei tweet
## e il numer totale di tweet, in quanto quest'ultimo è una combinazione dei tre, è possibile osservare altre interessanti correlazioni:
## considerando il Totale della produzione di energia, notiamo una forte correlazione positiva con la Popolazione del paese (0.92),
## con il consumo di elettricità (1) e con la quota di investimenti in green (0.86). Sulla base di queste relazioni è possibile concludere
## che i paesi ad investire maggiormente in green energy siano quelli con una popolazione abbastanza ampia e conseguentemente abbiano un ampio
## fabbisogno energetico.
## Considerando invece la quota di Investimenti in Green possiamo notare una serie di relazioni positive tra cui quella con il Tot dell'energia
## prodotta, spiegata poco fa, e anche quella scontata con il Tot di Renewable Energy prodotta dal paese (0.99), ma anche Electricity Consumption(0.85),
## Population (0.75), e Fossil CO2 Emissions (0.84). Queste relazioni ci portano a concludere che un paese investirà maggiormente in green quando
## il suo fabbisogno di energia dipendente a sua volta dalla popolazione sarà elevato, e le relative emissioni di CO2 saranno elevate.

gg1<-ggplot(ds1, aes(x=`RE_%_of_total`, y=ds1$`Fossil CO2 Emissions`, color = Country)) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('Emissioni in base alla percentuale di RE sul totale della produzione di energia') + ylab('CO2 Emissions') + xlab('%RE Prodotta sul totale')
gg1 
## Dal grafico è possibile osservare come ad eccezione degli Stati Uniti che rappresenta una quota di emissioni elevatissima rispetto
## alla produzione di Renewable energy, gli altri paesi abbiano una quantità di emissioni molto più contenuta, diversificandosi rispetto
## alla % di RE prodotta.

gg2<-ggplot(ds1, aes(x=Total_GWh_prod, y=`Fossil CO2 Emissions`, color = `Population`)) + 
  scale_x_log10() +
  geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  scale_colour_gradient(low = 'green', high = 'purple') + 
  ggtitle('Emissioni in base al totale della produzione di energia') +
  ylab('CO2 Emissions') + xlab('Totale Energia Prodotta(log scale)')
gg2 
## Il grafico mostra come la quota di emissioni prodotta sia superiore nei paesi con una popolazione elevata e che di conseguenza
## necesita di un fabbisogno energetico superiore, riprendendo sostanzialmente il trend osservato nel grafico precedente: gli Stati
## Uniti sono il paese più grande e pertanto presenteranno una forte quota di emissioni relativa al suo enorme fabbisogno, mentre gli altri
## paesi presentano un rapporto più contenuto.

gg3<-ggplot(ds1, aes(x=ds1$`Investimenti Green`, y=ds1$`Electricity Consumption`, color = Country)) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('Investimenti in base al consumo elettrico') + ylab('Consumo elettrico') + xlab('Investimenti in green')
gg3
## Grafico che consente di osservare i risultati trovati in precedenza da un punto di vista diverso:
## è possibile osservare come i paesi che sono caratterizzati da una quota superiore di investimenti in Renewable Energy
## sono quelli con un consumo di energia elettrica più alto, tra cui Canada e Stati Uniti, che rappresentano anche
## due tra i paesi con una popolazione elevata.

gg4<-ggplot(ds1, aes(x=ds1$`Avg Consumption per capita`, y=ds1$`Investimenti Green`, size =Population, col =Total_RE_Gwh)) + 
  geom_point() + geom_label_repel(aes(label = Country),
                                  box.padding   = 0.35, 
                                  point.padding = 0.5,
                                  segment.color = 'grey50') + scale_colour_gradient(low = 'green', high = 'red') +
  ggtitle('Investimenti in base al consumo medio percapita') + ylab('Investimenti Green') + xlab('Avg consumption percapita')
gg4 
## Dal grafico è possibile notare la quota di investimenti in base al consumo percapita e dimensione del paese per popolazione;
## tralasciando i commenti su Canada e Stati Uniti, che sono gli stessi per i grafici precedenti, dal plot attuale possiamo 
## osservare alcune situazioni interessanti: innanzitutto possiamo notare come, ad esclusione di Canada e Stati Uniti, gli altri paesi
## caratterizzati da una discreta quota di investimenti in RE siano perlopiù paesi situati nel continente Europeo, ad eccezione del Giappone,
## e tra questi compaiano anche paesi come Svezia e Norvegia che non sono caratterizzati da un'elevata popolazione. L'altra informazione
## interessante riguarda l'elevato consumo di energia percapita dell'Islanda, il qual risulata più alto addirittura di quello degli Stati Uniti.

gg5<-ggplot(ds1, aes(x=`Avg Consumption per capita`, y=Fossil_CO2_percapita, size = Population, col = `RE_%_of_total`)) + 
  geom_point() + geom_text(aes(label=Country),hjust=-0.15, vjust=0) + scale_colour_gradient(low = 'blue', high = 'yellow') + 
  ggtitle('Emissioni in base al consumo percapita') + ylab('CO2 Emissions percapita') + xlab('Avg Consumption percapita')
gg5 
## Rispetto al grafico precedente, mettendo in relazione le CO2 Emissions percapita con il cosnumo medio possiamo notare come altri paesi
## rispetto a Canada e Stati Uniti, specialmente di dimensioni ridotte, come Australia, South Korea, Lussemburgo ed Estonia, siano caratterizzati da 
## un dato di emissioni abbastanza elevato, soprattutto rispetto a paesi come il Giappone e la Germania, fortemente industrializzati.

################################################################################
## REGRESSIONE LINEARE
################################################################################
lm.full_modell=lm(ds1$`Investimenti Green`~., data = ds1[-c(1,8)], na.action = na.omit) 
summary(lm.full_modell)
str(lm.full_modell)
preds = predict(lm.full_modell, ds1[-c(1,8)])
mean( (preds - ds1$`Investimenti Green`)^2 ) ## MSE =  0.002047968
## Dal summary possiamo notare come utilizzando la maggior parte dei predittori, solo il totale della RE prodotta
## sia significativo. Il modello tuttavia overfitta i dati e pertanto l'R2 pari a 1 è il risultato dell'overfitting; di conseguenza
## sarebbe meglio procedere servendosi di altre variabili.

lm1=lm(ds1$`Investimenti Green`~ ds1$`Electricity Consumption` + ds1$Population + ds1$`Fossil CO2 Emissions` + ds1$Total_RE_Gwh, data = ds1, na.action = na.omit) 
summary(lm1)
str(lm1)
preds1 = predict(lm1, ds1[-c(1,8)])
mean( (preds1 - ds1$`Investimenti Green`)^2 ) ## MSE =  0.2369797
## Servendosi di un modello formato dalla variabile Tot_RE_Gwh, che rappresnta la variabile maggiormente significativa, e aggiungendo
## anche la Popolazione e le Emissioni CO2 possiamo notare come il modello consenta di ottenere risultati simili a quelli del modello precedente, con un leggero calo
## nell'R2 e un leggero aumento nel Mean Squared Error

lm2=lm(ds1$`Investimenti Green`~ ds1$Population , data = ds1, na.action = na.omit) 
summary(lm2)
str(lm2)
preds2 = predict(lm2, ds1[-c(1,8)])
mean( (preds2 - ds1$`Investimenti Green`)^2 ) ## MSE =  12.64388
## Facendo la regressione tra Population e Investimenti in green è possibile notare come non soltanto il predittore ora è significativo
## ma addirittura che cambia di segno, implicando che all'aumentare della popolazione in un paese, la quota di investimenti saràa superiore.
## Dall'R2 possiamo vedere come solo Population spiega il 70% circa della variabilità nella nostra response.

lm3=lm(ds1$`Investimenti Green`~ ds1$`Fossil CO2 Emissions`, data = ds1, na.action = na.omit) 
summary(lm3)
str(lm3)
preds3 = predict(lm3, ds1[-c(1,8)])
mean( (preds3 - ds1$`Investimenti Green`)^2 ) ## MSE =  10.85385
## Anche in questo caso, considerando solamente il predittore Fossil CO2 EMissions, possiamo vedere come i risultati siano molto diversi:
## ora il predittore risulta significativo, spiega il 72% della variabilità associata alla nostra response value.

lm4=lm(ds1$`Investimenti Green`~ ds1$`Electricity Consumption` ,data = ds1, na.action = na.omit) 
summary(lm4)
str(lm4)
preds4 = predict(lm4, ds1[-c(1,8)])
mean( (preds4 - ds1$`Investimenti Green`)^2 ) ## MSE =  9.942661
## Anche per questo predittore, possiamo vedere come i risultati siano molto diversi:
## ora il predittore risulta significativo, spiega il 73% della variabilità associata alla nostra response value.


lm5=lm(ds1$`Investimenti Green`~ ds1$Population + ds1$`Electricity Consumption` + ds1$`Fossil CO2 Emissions`, data = ds1, na.action = na.omit) 
summary(lm5)
str(lm5)
preds5 = predict(lm5, ds1[-c(1,8)])
mean( (preds5 - ds1$`Investimenti Green`)^2 ) ## MSE =  9.240915
## Eseguendo una regressione tra queste tre variabili, possiamo vedere come si ottengono dei buoni risultati, simili a quelli ottenuti in precedenza:
## le tre varibili spiegano il 75% circa della variabilità associata al fenomeno, e la considerazione più importante riguarda il fatto
## che escludendo da questa regressione, cosi' come dalle regresioni precedenti la variabile Tot_RE_Gwh i risultati siano più significativi
## da un punto di vista statistico

gg_lm <- ggplot(ds1, aes(x = Population + `Electricity Consumption` + `Fossil CO2 Emissions`, y = `Investimenti Green`)) + geom_point(color="Red", size=2) + 
  geom_smooth(method = lm, color="Blue", size=2)
gg_lm

################################################################################
## K-FOLD CROSS VALIDATION
################################################################################
set.seed(123)
k = 10 
folds = sample(1:k, nrow(ds1[-c(1, 8)]), replace=TRUE)
tmp = cbind(ds1, folds)
cv.errors = vector()
for(j in 1:k) {
  cv.fit = lm(`Investimenti Green`~ Population + `Electricity Consumption` + `Fossil CO2 Emissions`, 
                  data = ds1[folds != j, ]) # tutti i dati tranne il chunk j
  pred = predict(cv.fit, ds1[folds==j,]) # predizione proprio su j che era stato lasciato fuori
  cv.errors[j] = mean( (ds1$`Investimenti Green`[folds==j] - pred)^2 ) # calcolo e registro l'errore
}

mean.cv.errors = mean(cv.errors)
mean.cv.errors ## MSE = 50.84339
## Calcolando il test MSE con la k-fold cross validation si ottiene un valore di 50.84, maggiore rispetto a quello ottenuto
## con le sole osservazioni training pari a 9.24.

################################################################################
## CLASSIFICAZIONE:
################################################################################
## REGRESSIONE LOGISTICA
################################################################################
#attach(ds1)
#ds1 <- na.omit(ds1) ## Eliminazione degli NA
#ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`, c('Negative', 'Positive'), c('0', '1')) ## Refactoring della response value
#View(ds1)
#summary(ds1)

glm.fit = glm(`Risultato Investimento` ~ .-Country-Total-Positive-Negative-Neutral, data = ds1, family = binomial, na.action = na.omit)
summary(glm.fit) 

glm.fit2 = glm(`Risultato Investimento` ~ Population + `Investimenti Green` + Total_GWh_prod + Total_RE_Gwh, data = ds1, family = binomial, na.action = na.omit)
summary(glm.fit2)

glm.fit3 = glm(`Risultato Investimento` ~ `Investimenti Green`, data = ds1, family = binomial, na.action = na.omit)
summary(glm.fit3)

glm.probs <- predict(glm.fit2, type = "response")
prob <- round(glm.probs, digits = 4) ##  per approssimare i risultati ottenuti a 4 cifre dopo la virgola
#ds1 <- cbind(ds1, "expected investment"=prob) ## codice per aggiungere la colonna con le expeted prob al dataset, non necessaria
glm.pred <- rep("Negative",37)
glm.pred[glm.probs > .5] = "Positive"
table(glm.pred, `Risultato Investimento`)
mean(glm.pred==`Risultato Investimento`) ## Accuracy=1
mean(glm.pred!=`Risultato Investimento`) ## ErrorRate=0

################################################################################
## REGRESSIONE LOGISTICA CON ALGORITMO ALTERNATIVO (K-FOLD CV)
################################################################################
library(caret)
## Define training control

train_control <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

## Train the model on training set
model <- train(`Risultato Investimento` ~ .,
               data = ds1[,-c(1,8)],
               trControl = train_control,
               method = "glm",
               family=binomial())

model ## Accuracy=0.7833333
1-0.7833333 ## Cross-Validation Error = 0.2166667

## Eseguendo la classificazione con la Regressione Logistica si ottengono dei discreti risultati nel test set
## in quanto il cross-validation error ottenuto pari a 0.217, mentre l'Accuracy ottenuta nei dati training pari a 1
## dipende ancora una volta dalla mancanza di un numero elevato di osservazioni rispetto ai predittori, risultando in
## overfitting

################################################################################
## PRECISION, RECALL, F1 REGLOG
################################################################################
library(caret)
ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`,c('0', '1'), c('Negative', 'Positive'),) ## Refactoring della response value

y_glm <- ds1$`Risultato Investimento`
predictions_glm <- as.factor(glm.pred)

precision_glm <- posPredValue(predictions_glm, y_glm, positive="Positive") # 1
recall_glm <- sensitivity(predictions_glm, y_glm, positive="Positive") # 1

F1_glm <- (2 * precision_glm * recall_glm) / (precision_glm + recall_glm) # 1
## Calcolando Precision Recall e F1 score usando la Reg.Log otteniamo punteggi pari a 1, 1 e 1

################################################################################
## KNN
################################################################################
library(class)
ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`, c('Negative', 'Positive'), c('0', '1')) ## Refactoring della response value

set.seed(5)
train=sample(nrow(ds1), nrow(ds1)*0.66)
test=-train
ds_train <- ds1[train,]
ds_test <- ds1[test,]
head(ds_train)
head(ds_test)
view(ds_test)
view(ds_train)

XTrain = ds1[,-c(1,8,17,19)] # seleziona i predittori
YTrain = ds1$`Risultato Investimento` # seleziona la variabile qualitativa

XTest = ds_test[,-c(1,8,17,19)]
YTest = ds_test$`Risultato Investimento`

p.YTrain = knn(XTrain, XTrain, YTrain, k=1)
table(p.YTrain, ds1$`Risultato Investimento`)
mean(p.YTrain!=ds1$`Risultato Investimento`) 
mean(p.YTrain==ds1$`Risultato Investimento`)
p.YTest = knn(XTrain, XTest, YTrain, k=1)
mean(p.YTest!=ds1[test,]$`Risultato Investimento`) 
mean(p.YTest==ds1[test,]$`Risultato Investimento`) 

p.YTrain1 = knn(XTrain, XTrain, YTrain, k=5)
table(p.YTrain1, ds1$`Risultato Investimento`)
mean(p.YTrain1!=ds1$`Risultato Investimento`) ## ErrorRate = 0.3243243
mean(p.YTrain1==ds1$`Risultato Investimento`) ## Accuracy = 0.6756757
p.YTest1 = knn(XTrain, XTest, YTrain, k=5)
mean(p.YTest1!=ds1[test,]$`Risultato Investimento`) ## ErrorRate = 0.3076923
mean(p.YTest1==ds1[test,]$`Risultato Investimento`) ## Accuracy = 0.6923077

p.YTrain2 = knn(XTrain, XTrain, YTrain, k=15)
table(p.YTrain2, ds1$`Risultato Investimento`)
mean(p.YTrain2!=ds1$`Risultato Investimento`) ## ErrorRate = 0.3513514
mean(p.YTrain2==ds1$`Risultato Investimento`) ## Accuracy = 0.6486486

p.YTest2 = knn(XTrain, XTest, YTrain, k=15)
mean(p.YTest2!=ds1[test,]$`Risultato Investimento`) ## ErrorRate = 0.4615385
mean(p.YTest2==ds1[test,]$`Risultato Investimento`) ## Accuracy = 0.5384615

p.YTrain3 = knn(XTrain, XTrain, YTrain, k=12)
table(ds1$`Risultato Investimento`,p.YTrain3)
mean(p.YTrain3!=ds1$`Risultato Investimento`) ## ErrorRate = 0.3513514
mean(p.YTrain3==ds1$`Risultato Investimento`) ## Accuracy = 0.6486486

p.YTest3 = knn(XTrain, XTest, YTrain, k=12)
table(ds1[test,]$`Risultato Investimento`,p.YTest3)
mean(p.YTest3!=ds1[test,]$`Risultato Investimento`) ## ErrorRate = 0.1538462
mean(p.YTest3==ds1[test,]$`Risultato Investimento`) ## Accuracy = 0.8461538

## Utilizzando K=1 il risultato ottenuto è caratterizzato da alta variance e basso bias,
## pertanto sarà caratterizzato da una scarsa performance predittiva; di conseguenza si procede 
## all'utilizzo di valori di K più elevati, ad esempio K=12, che consentono di ottenere dei risultati più robusti
## da un punto di vista statistico, con un trascurabile costo in termini di aumento del bias.

################################################################################
## KNN: ALGORITMO ALTERNATIVO (CV)
################################################################################
library(caret)
trControl <- trainControl(method  = "cv", number  = 5, savePredictions = TRUE)
fit <- train(`Risultato Investimento` ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:15),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = ds1[,-c(1,8,19)])

fit ## 0.6750000
## Dal fit possiamo vedere come eseguendo la knn con K=12 il nostro classificatore
## presenta la prediction accuracy più elevata.

################################################################################
## PRECISION, RECALL, F1 KNN
################################################################################
## CON LIBRERIA CARET
#library(caret)
y <- as.factor(ds1$`Risultato Investimento`)
predictions <- as.factor(p.YTrain3)

precision <- posPredValue(predictions, y, positive="Positive") # 0.4545455
precision

recall <- sensitivity(predictions, y, positive="Positive") # 0.4166667
recall

F1 <- (2 * precision * recall) / (precision + recall) # 0.4347826
F1
## Calcolando Precision Recall e F1 score usando KNN otteniamo punteggi pari a 0.3333333, 0.08333333 e 0.1333333

## ALTERNATIVA CON LA LIBRERIA ROCR
library(ROCR)

y <- ds1$`Risultato Investimento`
predictions <- p.YTrain3
pred <- prediction(as.numeric(predictions), y);
pred

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec");
plot (RP.perf);

# ROC curve
ROC.perf <- performance(pred, "tpr", "fpr");
plot (ROC.perf);

# F1 curve
ROC.perf <- performance(pred,"f");
plot (ROC.perf);

# ROC area under the curve
auc.tmp <- performance(pred,"auc");
auc <- as.numeric(auc.tmp@y.values)
auc # 0.5261905

################################################################################
## RANDOM FOREST
################################################################################
library(randomForest)
## Refactoring della response value
ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`, c('Negative', 'Positive'), c('0', '1')) 

## Check sulla response value per l'algoritmo random forest
class(ds1$`Risultato Investimento`)
table(ds1$`Risultato Investimento`)

set.seed(36)
train <- sample(1:nrow(ds1), .66 * nrow(ds1))
ds1_train <- ds1[train, ]
ds1_test  <- ds1[-train, ]
View(ds1_train)

rf=randomForest(ds1_train$`Risultato Investimento`~., data=ds1_train[,-c(4,9,11:16,19)],
                mtry=13,importance=TRUE,ntree=500, na.action = na.omit)
rf_preds = predict(rf,newdata=ds1_test)
rf_preds
table(rf_preds, ds1_test$`Risultato Investimento`)
8/13 ## Accuracy = 0.6153846
1-(8/13) ## ErrorRate = 0.3846154
importance(rf)
varImpPlot(rf, main = 'Variable Importance Plot')

rf1=randomForest(ds1_train$`Risultato Investimento`~., data=ds1_train[,-c(4,9,11:16,19)],
                mtry=7,importance=TRUE,ntree=500, na.action = na.omit)
rf_preds1 = predict(rf1,newdata=ds1_test)
rf_preds1
table(rf_preds1, ds1_test$`Risultato Investimento`)
8/13 ## Accuracy = 0.6153846
1-(8/13) ## ErrorRate = 0.3846154
importance(rf1)
varImpPlot(rf1, main = 'Variable Importance Plot')

rf2=randomForest(ds1_train$`Risultato Investimento`~., data=ds1_train[,-c(4,9,11:16,19)],
                 mtry=4,importance=TRUE,ntree=500, na.action = na.omit)
summary(rf2)
rf_preds2 = predict(rf2,newdata=ds1_test)
rf_preds2
table(rf_preds2, ds1_test$`Risultato Investimento`)
8/13 ## Accuracy = 0.6153846
1-(8/13) ## ErrorRate = 0.3846154
importance(rf2)
varImpPlot(rf2, main = 'Variable Importance Plot')

################################################################################
## PRECISION, RECALL, F1 RANDOM FOREST
################################################################################
## CON LIBRERIA CARET
library(caret)
y_rf <- ds1_test$`Risultato Investimento`
predictions_rf <- rf_preds1

precision_rf <- posPredValue(predictions_rf, y_rf, positive="1") #  1
precision_rf

recall_rf <- sensitivity(predictions_rf, y_rf, positive="1") # 0.1666667

F1_rf <- (2 * precision_rf * recall_rf) / (precision_rf + recall_rf) # 0.2857143
## Calcolando Precision Recall e F1 score usando KNN otteniamo punteggi pari a 1, 0.1666667 e 0.2857143

################################################################################
## SUPPORT VECTOR MACHINE
################################################################################
library(e1071)
## Codifica della variabile con -1 e 1 per il funzionamento di SVM, N.B: diversa da quella per gli altri classificatori.
ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`, c('0', '1'), c('-1', '1')) 

## Divisione del Ds in training e test
set.seed(5)
train_svm=sample(nrow(ds1), nrow(ds1)*0.66)
test_svm=-train_svm
ds_svm_train <- ds1[train_svm,-c(1,2,8)]
ds_svm_test <- ds1[test_svm,-c(1,2,8)]
head(ds_svm_train)
head(ds_svm_test)
view(ds_svm_test)
view(ds_svm_train)

svmfit = svm(`Risultato Investimento`~., data=ds_svm_train, kernel="linear", cost=10,
             scale=TRUE) #svm, kernel linear usa un svm lineare, cost punti da usare per i margini, scale per scalare i dati in caso di dati con misure diverse
names(svmfit)
svmfit$index
summary(svmfit)

table(ds_svm_train$`Risultato Investimento`, predict(svmfit,ds_svm_train))
table(ds_svm_test$`Risultato Investimento`, predict(svmfit, ds_svm_test))

g_svm <- ggplot(ds_svm_train, aes(x=`Total_RE_Gwh`, y=`Electricity Consumption`, col=`Risultato Investimento`))+geom_point()
g_svm

pred_test1 <-predict(svmfit,ds_svm_test)
y_svm <- ds_svm_test$`Risultato Investimento`
predictions_svm <- pred_test1

mean(pred_test1==ds_svm_test$`Risultato Investimento`) #accuracy 0.8461538
precision_svm <- posPredValue(predictions_svm, y_svm, positive="1") # 0.6666667
recall_svm <- sensitivity(predictions_svm, y_svm, positive="1") # 0.6666667
F1_svm <- (2 * precision_svm * recall_svm) / (precision_svm + recall_svm) #0.6666667


## Build model – linear kernel and C-classification (soft margin) with default cost (C=1)
svm_model2 <- svm(`Risultato Investimento`~ ., 
                 data=ds_svm_train, 
                 method="C-classification", 
                 kernel="linear",
                 scale = TRUE)

## Training set predictions
pred_train2 <-predict(svm_model2,ds_svm_train)
pred_train2
table("truth"=ds_svm_train$`Risultato Investimento`, "pred"=pred_train2)
mean(pred_train2==ds_svm_train$`Risultato Investimento`)#[1] 1
mean(pred_train2!=ds_svm_train$`Risultato Investimento`)#[1] 0 tasso errata class train

## Test set predictions
pred_test2 <-predict(svm_model2,ds_svm_test)
table("truth"=ds_svm_test$`Risultato Investimento`, "pred"=pred_test2)
mean(pred_test2==ds_svm_test$`Risultato Investimento`) #[1] 0.8181818
mean(pred_test2!=ds_svm_test$`Risultato Investimento`) #[1] 0.1818182 tasso errata classificazione test

################################################################################
## PRECISION, RECALL, F1 SVM
################################################################################
## CON LIBRERIA CARET
#library(caret)
y_svm <- ds_svm_test$`Risultato Investimento`
predictions_svm <- pred_test2

precision_svm <- posPredValue(predictions_svm, y_svm, positive="1") # 0.6666667
recall_svm <- sensitivity(predictions_svm, y_svm, positive="1") # 0.6666667

F1_svm <- (2 * precision_svm * recall_svm) / (precision_svm + recall_svm) # 0.6666667
## Calcolando Precision Recall e F1 score usando SVM otteniamo punteggi pari a 0.6666667, 0.6666667 e 0.6666667

################################################################################
## SUPPORT VECTOR MACHINE CON LOOCV
################################################################################
#library(e1071) ## Caricata nella procedura precedente
## Codifica della variabile con -1 e 1 per il funzionamento di SVM
#ds1$`Risultato Investimento` <- factor(ds1$`Risultato Investimento`, c('0', '1'), c('-1', '1')) ## già codificata prima
result_svm <- 0
for (i in 1:length(ds1)){
  fit_svm = svm(`Risultato Investimento` ~ ., data=ds1[-i,-c(1,2,8)], type='C-classification', kernel='linear', scale = TRUE)
  pred_svm = predict(fit_svm, ds1[i,-c(1,2,8),drop=F])
  result_svm <- result_svm + table(true=ds1[i,]$`Risultato Investimento`, pred_svm=pred_svm);
}
classAgreement(result_svm)
classAgreement(result_svm)$diag #[1] 0.8947368 media dei corretti classificati per i diversi fold
1 - (classAgreement(result_svm)$diag) #[1] 0.1052632 tasso errata classificazione

## Utilizzando entrambe le procedure, si ottiene un tasso di errata classificazione circa di 0.1052632, quindi
## usando SVM il classificatore funziona con discreti risultati in termini di prediction accuracy.


detach(ds1)
