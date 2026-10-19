function summary = main(mode, datasetNames, runNum, resultDir)
%MAIN Run QELCS with the Case A settings.
% main                    Run the full experiment.
% main('quick', 'Colon')   Check the pipeline on Colon.
% main('paper', 'Colon')   Run the formal experiment on Colon.

if nargin < 1 || isempty(mode), mode = 'paper'; end
opts = parameters(mode);
if nargin < 2 || isempty(datasetNames)
    datasetNames = ["11_Tumors", "9_Tumors", "Adenocarcinoma", "ALLAML", ...
        "arcene", "Brain_Tumor1", "BreastCancer1", "BreastCancer2", ...
        "CLL-SUB-111", "Colon", ...
        "GLI_85", "Leukemia", "Leukemia1", "Leukemia2", "Lung_Cancer", ...
        "lung", "Lymphoma", "MLL_3", "nci9", "Prostate_GE", "SMK-CAN-187"];
    if strcmp(opts.mode, 'quick'), datasetNames = "Colon"; end
end
if nargin < 3 || isempty(runNum), runNum = opts.runNum; end
validateattributes(runNum, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
opts.runNum = runNum;
datasetNames = string(datasetNames);
datasetNames = datasetNames(:);
for i = 1:numel(datasetNames)
    if ~endsWith(datasetNames(i), '.mat')
        datasetNames(i) = datasetNames(i) + ".mat";
    end
end

projectDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(projectDir, 'DataSet');
if nargin < 4 || isempty(resultDir)
    resultDir = fullfile(projectDir, 'results', opts.mode);
end
if ~isfolder(resultDir), mkdir(resultDir); end
summary = table;

for datasetIndex = 1:numel(datasetNames)
    datasetName = datasetNames(datasetIndex);
    [feat, label] = load_dataset(fullfile(dataDir, datasetName));
    numClasses = numel(unique(label));
    fprintf('\n%s | N=%d | T=%d | runs=%d | k=%d | thresholds=%.2f/%.2f\n', ...
        datasetName, opts.N, opts.T, runNum, opts.k, ...
        opts.thresholdAccuracyStart, opts.thresholdSparseStart);

    testAcc = zeros(runNum, 1);
    trainCV = zeros(runNum, 1);
    gbest = zeros(runNum, 1);
    featureCount = zeros(runNum, 1);
    elapsedSeconds = zeros(runNum, 1);
    runSeeds = zeros(runNum, 1);
    foldTestAccuracy = cell(runNum, 1);
    foldTrainCV = cell(runNum, 1);
    foldGbest = cell(runNum, 1);
    foldFeatures = cell(runNum, 1);
    selectedFeatures = cell(runNum, opts.outerK);
    convergenceCurves = cell(runNum, opts.outerK);
    outerTestIndices = cell(runNum, opts.outerK);
    innerTrainIndices = cell(runNum, opts.outerK);
    innerValIndices = cell(runNum, opts.outerK);
    oofPredictions = zeros(size(feat, 1), runNum);

    for runIndex = 1:runNum
        % Keep the original RNG order: generate ALL folds before optimization.
        runSeeds(runIndex) = opts.seed + runIndex;
        rng(runSeeds(runIndex), 'twister');
        outerCV = cvpartition(label, 'KFold', opts.outerK);
        for outerFold = 1:opts.outerK
            trainIndex = training(outerCV, outerFold);
            innerCV = cvpartition(label(trainIndex), 'KFold', opts.innerK);
            cvTrain = cell(opts.innerK, 1);
            cvVal = cell(opts.innerK, 1);
            for innerFold = 1:opts.innerK
                cvTrain{innerFold} = training(innerCV, innerFold);
                cvVal{innerFold} = test(innerCV, innerFold);
            end
            innerTrainIndices{runIndex, outerFold} = cvTrain;
            innerValIndices{runIndex, outerFold} = cvVal;
        end

        oofPrediction = nan(size(label));
        assigned = false(size(label));
        foldAcc = zeros(opts.outerK, 1);
        foldCV = zeros(opts.outerK, 1);
        foldFitness = zeros(opts.outerK, 1);
        foldCount = zeros(opts.outerK, 1);
        runTime = 0;
        for outerFold = 1:opts.outerK
            fprintf('  Run %d/%d | fold %d/%d\n', ...
                runIndex, runNum, outerFold, opts.outerK);
            trainIndex = training(outerCV, outerFold);
            testIndex = test(outerCV, outerFold);
            xTrain = feat(trainIndex, :);
            yTrain = label(trainIndex);
            xTest = feat(testIndex, :);
            yTest = label(testIndex);
            algorithmOptions = opts;
            algorithmOptions.cvTrain = innerTrainIndices{runIndex, outerFold};
            algorithmOptions.cvVal = innerValIndices{runIndex, outerFold};

            timer = tic;
            model = gQELCS(xTrain, yTrain, algorithmOptions);
            prediction = jknn(xTrain(:, model.sf), yTrain, xTest(:, model.sf), opts.k);
            foldCV(outerFold) = subset_cv_accuracy( ...
                xTrain, yTrain, model.sf, algorithmOptions);
            runTime = runTime + toc(timer);

            oofPrediction(testIndex) = prediction;
            assigned(testIndex) = true;
            foldAcc(outerFold) = mean(prediction == yTest);
            foldFitness(outerFold) = model.gb;
            foldCount(outerFold) = model.nf;
            selectedFeatures{runIndex, outerFold} = model.sf;
            convergenceCurves{runIndex, outerFold} = model.c;
            outerTestIndices{runIndex, outerFold} = find(testIndex);
        end
        assert(all(assigned), 'Not all samples received an outer-fold prediction.');

        % Pool outer predictions; do not average unequal-sized test folds.
        testAcc(runIndex) = mean(oofPrediction == label);
        trainCV(runIndex) = mean(foldCV);
        gbest(runIndex) = mean(foldFitness);
        featureCount(runIndex) = mean(foldCount);
        elapsedSeconds(runIndex) = runTime;
        foldTestAccuracy{runIndex} = foldAcc;
        foldTrainCV{runIndex} = foldCV;
        foldGbest{runIndex} = foldFitness;
        foldFeatures{runIndex} = foldCount;
        oofPredictions(:, runIndex) = oofPrediction;
        fprintf('  TestAcc %.6f | TrainCV %.6f | Gbest %.6f | F %.2f | %.2fs\n', ...
            testAcc(runIndex), trainCV(runIndex), gbest(runIndex), ...
            featureCount(runIndex), runTime);
    end

    testClasses = repmat(compose('%d/%d', numClasses, numClasses), runNum, 1);
    runTable = table((1:runNum)', testAcc, trainCV, gbest, featureCount, ...
        repmat(opts.outerK, runNum, 1), repmat(opts.innerK, runNum, 1), ...
        testClasses, elapsedSeconds, runSeeds, ...
        'VariableNames', {'Run', 'TestAcc', 'TrainCV', 'Gbest', 'F', ...
        'OuterK', 'InnerK', 'TestClasses', 'TimeSeconds', 'SplitSeed'});
    datasetResult = struct;
    datasetResult.dataset = datasetName;
    datasetResult.algorithm = 'QELCS';
    datasetResult.protocol = 'Repeated nested CV: outer 10-fold, inner 5-fold';
    datasetResult.options = opts;
    datasetResult.runTable = runTable;
    datasetResult.foldTestAccuracy = foldTestAccuracy;
    datasetResult.foldTrainCV = foldTrainCV;
    datasetResult.foldGbest = foldGbest;
    datasetResult.foldFeatures = foldFeatures;
    datasetResult.selectedFeatures = selectedFeatures;
    datasetResult.convergenceCurves = convergenceCurves;
    datasetResult.outerTestIndices = outerTestIndices;
    datasetResult.innerTrainIndices = innerTrainIndices;
    datasetResult.innerValIndices = innerValIndices;
    datasetResult.runSeeds = runSeeds;
    datasetResult.oofPredictions = oofPredictions;
    datasetResult.seedScheme = 'rng(baseSeed + runIndex, twister) once per run';
    baseName = erase(datasetName, '.mat');
    writetable(runTable, fullfile(resultDir, baseName + "_runs.csv"));
    save(fullfile(resultDir, baseName + "_result.mat"), 'datasetResult');

    row = table(datasetName, mean(testAcc), std(testAcc), mean(trainCV), ...
        std(trainCV), mean(gbest), std(gbest), mean(featureCount), ...
        std(featureCount), mean(elapsedSeconds), ...
        'VariableNames', {'Dataset', 'TestAccMean', 'TestAccStd', ...
        'TrainCVMean', 'TrainCVStd', 'GbestMean', 'GbestStd', ...
        'FeatureMean', 'FeatureStd', 'TimeMeanSeconds'});
    if isempty(summary), summary = row; else, summary = [summary; row]; end %#ok<AGROW>
    writetable(summary, fullfile(resultDir, 'summary.csv'));
    save(fullfile(resultDir, 'summary.mat'), 'summary', 'opts', 'datasetNames', 'runNum');
end
fprintf('\nResults saved to: %s\n', resultDir);
end


function accuracy = subset_cv_accuracy(feat, label, selected, opts)
correct = 0;
total = 0;
for fold = 1:numel(opts.cvTrain)
    tr = opts.cvTrain{fold};
    va = opts.cvVal{fold};
    prediction = jknn(feat(tr, selected), label(tr), feat(va, selected), opts.k);
    correct = correct + nnz(prediction == label(va));
    total = total + nnz(va);
end
accuracy = correct / total;
end


function [feat, label] = load_dataset(path)
if ~isfile(path), error('qelcs:DatasetMissing', 'Dataset not found: %s', path); end
data = load(path);
if ~isfield(data, 'feat') || ~isfield(data, 'label')
    error('qelcs:DatasetVariables', '%s must contain feat and label.', path);
end
feat = double(data.feat);
[~, ~, label] = unique(data.label(:));
label = double(label);
if isempty(feat) || size(feat, 1) ~= numel(label)
    error('qelcs:DatasetSize', 'Invalid data dimensions in %s.', path);
end
if any(~isfinite(feat), 'all') || any(ismissing(data.label(:)))
    error('qelcs:NonFiniteData', 'Invalid features or labels in %s.', path);
end
end
