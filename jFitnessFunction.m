function [cost, classError, featureRatio, foldError] = jFitnessFunction(feat, label, mask, opts)
mask = logical(mask(:)');
numSelected = nnz(mask);
featureRatio = numSelected / numel(mask);
if numSelected == 0
    classError = 1;
    cost = 1;
    foldError = ones(numel(opts.cvTrain), 1);
    return;
end

numFolds = numel(opts.cvTrain);
foldError = zeros(numFolds, 1);
foldSize = zeros(numFolds, 1);
for f = 1:numFolds
    tr = opts.cvTrain{f};
    va = opts.cvVal{f};
    pred = jknn( ...
        feat(tr, mask), label(tr), feat(va, mask), opts.k);
    yValid = label(va);
    foldSize(f) = numel(yValid);
    foldError(f) = mean(pred ~= yValid);
end
classError = sum(foldSize .* foldError) / sum(foldSize);
cost = 0.90 * classError + 0.10 * featureRatio;
end
