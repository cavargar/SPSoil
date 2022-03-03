clc;
close;
clear;
disp('descargando datos de Na');
load("AllDataNa.mat")
train=Allminmaxd1d2fftfeatureNa;
trainingData=train;
cate=train{:,1};
fin=150;
fname="logfile.csv";
fh = fopen(fname, 'w+');
while fh == -1
    pause(1);
    fh = fopen(fname, 'w+');
end
fprintf(fh, "i;j;TPR;TNR;PPV;NPV;ACC;F1;MCC;confMat\n");
fclose(fh);

%%
tic
parfor i=1:fin
    for j=1:fin
        costm=[0 i; j 0];
        [trainedClassifier, validationAccuracy,validationPredictions] = trainClassifierCubicCost(trainingData,costm);
        [confMat,TPR,TNR,PPV,NPV,ACC,F1,MCC] = ConfusionMatrixExtended(cate,validationPredictions);
        fh = fopen(fname, 'a+');
        while fh == -1
            pause(1);
            fh = fopen(fname, 'a+');
        end
        CM=confMat(:);
        A = [i,j,TPR,TNR,PPV,NPV,ACC,F1,MCC,CM(1),CM(2),CM(3),CM(4)];
        fprintf(fh, "%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f;%3.8f\n",A);
        fclose(fh);
    end
end
toc


