function [confMat,TPR,TNR,PPV,NPV,ACC,F1,MCC] = ConfusionMatrixExtended(cate,validationPredictions)
%ConfusionMatrixExtended retorna:
% confMat,TPR,TNR,PPV,NPV,ACC,F1,MCC
% ver la notacion en:
% https://en.wikipedia.org/wiki/Confusion_matrix
%
%confusionchart(train{:,1},validationPredictions)
[confMat,order] = confusionmat(cate,validationPredictions); %#ok<ASGLU>

%%% recall
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
end
Recall=sum(recall)/size(confMat,1);

%%% precision
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
Precision=sum(precision)/size(confMat,1);


%%% specificity
for i =1:size(confMat,1)
    aaa=confMat;
    aaa(i,:)=0;
    aaa(:,i)=0;
    TN=sum(sum(aaa));
    FP=sum(confMat(:,i))-confMat(i,i);
    specificity(i)=(TN)/(TN+FP);
end
Specificity=sum(specificity)/size(confMat,1);

%%%negative predictive value 
for i =1:size(confMat,1)
    aaa=confMat;
    aaa(i,:)=0;
    aaa(:,i)=0;
    TN=sum(sum(aaa));
    FN=sum(confMat(i,:))-confMat(i,i);
    negativePredValue(i)=(TN)/(TN+FN);
end
NegativePredValue=sum(negativePredValue)/size(confMat,1);




PPV=Precision;
TPR=Recall;
TNR=Specificity;
NPV=NegativePredValue;

FDR=1-PPV;
FNR=1-TPR;
FPR=1-TNR;
FOR=1-NPV;


%%% mean F1
F1=Precision*Recall/(Precision+Recall);
%%% mean Accuracy
ACC=(sum(diag(confMat)))/sum(sum(confMat));
%%% mean Matthews correlation coefficient
MCC=sqrt(PPV*TPR*TNR*NPV)-sqrt(FDR*FNR*FPR*FOR);
end



