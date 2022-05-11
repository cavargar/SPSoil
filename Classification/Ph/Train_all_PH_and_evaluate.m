clc;
close;
clear;
disp('descargando datos de Ph');
load("AllDataph.mat");
train=AllDataPh;
trainingData=train;
cate=train{:,1};
fin=8;
fname="logfile.csv";
fh = fopen(fname, 'w+');
while fh == -1
    pause(1);
    fh = fopen(fname, 'w+');
end
fprintf(fh, "i;j;k;l;m;n;TPR;TNR;PPV;NPV;ACC;F1;MCC;confMat\n");
fclose(fh);

%%
tic
parfor i=1:fin
    for j=1:fin
         for k=1:fin
             for l=1:fin
                 for m=1:fin
                    for n=1:fin
                        costm=[0 i j; k 0 l; m n 0];
                        [trainedClassifier, validationAccuracy,validationPredictions] = trainClassifierLinearCost(trainingData,costm);
                        [confMat,TPR,TNR,PPV,NPV,ACC,F1,MCC] = ConfusionMatrixExtended(cate,validationPredictions);
                        fh = fopen(fname, 'a+');
                        while fh == -1
                            pause(1);
                            fh = fopen(fname, 'a+');
                        end
                        CM=confMat(:);
                        A = [i,j,k,l,m,n,TPR,TNR,PPV,NPV,ACC,F1,MCC,CM(1),CM(2),CM(3),CM(4),CM(5),CM(6),CM(7),CM(8),CM(9)];
                        fprintf(fh, "%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f\n",A);
                        fclose(fh);
                    end
                 end
             end
         end
    end
end
toc


