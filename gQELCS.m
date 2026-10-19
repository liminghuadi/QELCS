function result = gQELCS(feat, label, opts)

feat = double(feat);
label = label(:);
[n, d] = size(feat);
if n ~= numel(label)
    error('qelcs:SizeMismatch', 'Rows of feat must equal the number of labels.');
end
nParticles = opts.N;
maxIter = opts.T;

% Stable dual-score filtering and progressive initialization.
[suScores, classScores] = ...
    stable_filter_scores(feat, label, opts);
filterScores = (1 - opts.classBalanceWeight) * normalize01(suScores(:))' + ...
    opts.classBalanceWeight * normalize01(classScores(:))';
[~, rankIndex] = sort(filterScores + eps * rand(1, d), 'descend');
candidateCount = max(1, ceil(opts.suCandidateRatio * d));
candidateRankIndex = rankIndex(1:candidateCount);
candidateMask = false(1, d);
candidateMask(candidateRankIndex) = true;
[position, velocity] = piecewise_initialization( ...
    candidateRankIndex, nParticles, d, opts);

pbestPosition = position;
pbestMask = false(nParticles, d);
pbestFitness = inf(nParticles, 1);
pbestError = ones(nParticles, 1);
pbestRatio = ones(nParticles, 1);
numFolds = numel(opts.cvTrain);
pbestFoldError = ones(nParticles, numFolds);
gbestFitness = inf;
gbestMask = false(1, d);
gbestError = 1;
gbestRatio = 1;

% Neutral decoding establishes the initial particle roles.
for i = 1:nParticles
    mask = position(i, :) > 0.5;
    mask = mask & candidateMask;
    mask = guarantee_nonempty(mask, position(i, :), candidateMask, rankIndex);
    [fit, err, ratio, foldError] = ...
        jFitnessFunction(feat, label, mask, opts);
    pbestMask(i, :) = mask;
    pbestFitness(i) = fit;
    pbestError(i) = err;
    pbestRatio(i) = ratio;
    pbestFoldError(i, :) = foldError(:)';
    if fit < gbestFitness
        gbestFitness = fit;
        gbestMask = mask;
        gbestError = err;
        gbestRatio = ratio;
    end
end

archive = empty_archive(d, numFolds);
archive = update_robust_archive(archive, pbestMask, pbestFitness, ...
    pbestError, pbestRatio, pbestFoldError, opts);

curve = zeros(1, maxIter);
for t = 1:maxIter
    role = role_partition(pbestError, pbestRatio);
    progress = t / maxIter;
    [accuracyLeader, sparseLeader] = robust_archive_leaders(archive, progress, opts);

    remaining = (1 - progress) ^ opts.thresholdDecayExponent;
    thresholdAccuracy = opts.thresholdConverged - ...
        (opts.thresholdConverged - opts.thresholdAccuracyStart) * remaining;
    thresholdSparse = opts.thresholdConverged + ...
        (opts.thresholdSparseStart - opts.thresholdConverged) * remaining;
    inertia = opts.wMax - (opts.wMax - opts.wMin) * progress;

    % Dense, accurate particles follow the sparse leader; sparse particles
    % follow the accuracy leader. The thresholds converge to 0.5.
    for i = 1:nParticles
        if role(i) == 1
            leader = sparseLeader;
            threshold = thresholdSparse;
        else
            leader = accuracyLeader;
            threshold = thresholdAccuracy;
        end

        r1 = rand(1, d);
        r2 = rand(1, d);
        velocity(i, :) = inertia * velocity(i, :) + ...
            opts.c1 * r1 .* (pbestPosition(i, :) - position(i, :)) + ...
            opts.c2 * r2 .* (leader - position(i, :));
        velocity(i, :) = max(-opts.vMax, min(opts.vMax, velocity(i, :)));
        position(i, :) = max(0, min(1, position(i, :) + velocity(i, :)));
        position(i, ~candidateMask) = 0;
        velocity(i, ~candidateMask) = 0;

        mask = (position(i, :) > threshold) & candidateMask;
        mask = guarantee_nonempty(mask, position(i, :), candidateMask, rankIndex);
        [fit, err, ratio, foldError] = ...
            jFitnessFunction(feat, label, mask, opts);

        % No auxiliary criterion changes pbest or gbest.
        if fit < pbestFitness(i)
            pbestPosition(i, :) = position(i, :);
            pbestMask(i, :) = mask;
            pbestFitness(i) = fit;
            pbestError(i) = err;
            pbestRatio(i) = ratio;
            pbestFoldError(i, :) = foldError(:)';
        end
        if fit < gbestFitness
            gbestFitness = fit;
            gbestMask = mask;
            gbestError = err;
            gbestRatio = ratio;
        end

    end

    % Reuse evaluated subsets; archive guidance adds no fitness evaluations.
    archive = update_robust_archive(archive, pbestMask, pbestFitness, ...
        pbestError, pbestRatio, pbestFoldError, opts);
    curve(t) = gbestFitness;
    if opts.verbose
        fprintf('Iter %3d/%3d | fitness %.6f | features %d\n', ...
            t, maxIter, gbestFitness, nnz(gbestMask));
    end
end

result.gb = gbestFitness;
result.error = gbestError;
result.ratio = gbestRatio;
result.mask = gbestMask;
result.sf = find(gbestMask);
result.nf = nnz(gbestMask);
result.c = curve;
end


function [position, velocity] = piecewise_initialization(rankIndex, nParticles, d, opts)
position = zeros(nParticles, d);
velocity = zeros(nParticles, d);
candidateCount = numel(rankIndex);
coreLength = max(1, ceil(opts.initialCoreRatio * candidateCount));
segmentStep = max(1, round(candidateCount / nParticles));
for i = 1:nParticles
    partitionLength = min(candidateCount, max(coreLength, segmentStep * i));
    coreFeatures = rankIndex(1:coreLength);
    position(i, coreFeatures) = 0.6 + 0.4 * rand(1, coreLength);
    if partitionLength > coreLength
        remaining = rankIndex(coreLength + 1:partitionLength);
        position(i, remaining) = rand(1, numel(remaining));
    end
end
end


function role = role_partition(errors, ratios)
roleScore = normalize01(ratios) - normalize01(errors);
[~, order] = sort(roleScore, 'descend');
half = floor(numel(errors) / 2);
role = zeros(numel(errors), 1);
role(order(1:half)) = 1;
role(order(half + 1:end)) = 2;
end


function archive = empty_archive(d, numFolds)
archive.masks = false(0, d);
archive.fitness = zeros(0, 1);
archive.error = zeros(0, 1);
archive.ratio = zeros(0, 1);
archive.foldError = zeros(0, numFolds);
end


function archive = update_robust_archive(archive, masks, fitness, errors, ...
    ratios, foldErrors, opts)
% Preserve historically competitive subsets without reevaluating them.
allMasks = [archive.masks; logical(masks)];
allFitness = [archive.fitness; fitness(:)];
allError = [archive.error; errors(:)];
allRatio = [archive.ratio; ratios(:)];
allFoldError = [archive.foldError; foldErrors];

valid = isfinite(allFitness) & isfinite(allError) & ...
    all(isfinite(allFoldError), 2);
allMasks = allMasks(valid, :);
allFitness = allFitness(valid);
allError = allError(valid);
allRatio = allRatio(valid);
allFoldError = allFoldError(valid, :);

% Identical subsets have identical deterministic inner-fold predictions;
% retain a single copy to prevent the archive from losing diversity.
[allMasks, uniqueIndex] = unique(allMasks, 'rows', 'stable');
allFitness = allFitness(uniqueIndex);
allError = allError(uniqueIndex);
allRatio = allRatio(uniqueIndex);
allFoldError = allFoldError(uniqueIndex, :);

bestFitness = min(allFitness);
competitive = allFitness <= bestFitness + opts.archiveRetentionTolerance;
allMasks = allMasks(competitive, :);
allFitness = allFitness(competitive);
allError = allError(competitive);
allRatio = allRatio(competitive);
allFoldError = allFoldError(competitive, :);

foldVariation = std(allFoldError, 0, 2);
robustness = allError + opts.archiveRobustnessWeight * foldVariation;
[~, objectiveBest] = min(allFitness);
[~, robustOrder] = sortrows([robustness, allFitness, allRatio], [1, 2, 3]);
keep = unique([objectiveBest; robustOrder], 'stable');
keep = keep(1:min(opts.archiveCapacity, numel(keep)));

archive.masks = allMasks(keep, :);
archive.fitness = allFitness(keep);
archive.error = allError(keep);
archive.ratio = allRatio(keep);
archive.foldError = allFoldError(keep, :);
end


function [accuracyLeader, sparseLeader] = robust_archive_leaders(archive, progress, opts)
% The fitness band contracts as the search progresses. Robustness affects
% leader selection only; the original scalar objective still updates all
% personal and global best memories.
guideTolerance = opts.archiveGuideToleranceEnd + ...
    (opts.archiveGuideToleranceStart - opts.archiveGuideToleranceEnd) * ...
    (1 - progress) ^ 2;
bestFitness = min(archive.fitness);
eligible = find(archive.fitness <= bestFitness + guideTolerance);

foldVariation = std(archive.foldError, 0, 2);
robustness = archive.error + ...
    opts.archiveRobustnessWeight * foldVariation;
[~, localOrder] = sortrows( ...
    [robustness(eligible), archive.fitness(eligible), archive.ratio(eligible)], ...
    [1, 2, 3]);
robustOrder = eligible(localOrder);

eliteCount = max(1, ceil(opts.archiveConsensusFraction * numel(robustOrder)));
accuracyElite = robustOrder(1:eliteCount);
weights = (eliteCount:-1:1)';
accuracyLeader = (weights' * double(archive.masks(accuracyElite, :))) / ...
    sum(weights);

% The sparse role first stays inside the same objective-quality band, then
% inside a robustness band, and only then minimizes subset size.
robustLimit = min(robustness(eligible)) + guideTolerance;
sparseEligible = eligible(robustness(eligible) <= robustLimit);
[~, sparseOrder] = sortrows( ...
    [archive.ratio(sparseEligible), robustness(sparseEligible), ...
    archive.fitness(sparseEligible)], [1, 2, 3]);
sparseIndex = sparseEligible(sparseOrder(1));
sparseLeader = double(archive.masks(sparseIndex, :));
end


function [stableScores, classScores] = ...
    stable_filter_scores(feat, label, opts)
if isfield(opts, 'cvTrain') && numel(opts.cvTrain) > 1
    numFolds = numel(opts.cvTrain);
    foldScores = zeros(numFolds, size(feat, 2));
    foldClassScores = zeros(numFolds, size(feat, 2));
    for fold = 1:numFolds
        tr = opts.cvTrain{fold};
        foldScores(fold, :) = SU_TOOL(feat(tr, :), label(tr), opts.suBins);
        foldClassScores(fold, :) = class_balanced_scores(feat(tr, :), label(tr));
    end
    meanScore = mean(foldScores, 1);
    variability = std(foldScores, 0, 1);
    stableScores = max(0, meanScore - opts.suStabilityPenalty * variability);
    meanClassScore = mean(foldClassScores, 1);
    classVariability = std(foldClassScores, 0, 1);
    classScores = max(0, meanClassScore - ...
        opts.suStabilityPenalty * classVariability);
else
    stableScores = SU_TOOL(feat, label, opts.suBins);
    classScores = class_balanced_scores(feat, label);
end
end


function scores = class_balanced_scores(feat, label)
% Equal class weighting prevents minority or difficult classes from being
% hidden by a single global relevance score. Absolute point-biserial
% correlation is scale invariant and can be evaluated efficiently.
centeredFeat = feat - mean(feat, 1);
featureNorm = sqrt(sum(centeredFeat .^ 2, 1));
classes = unique(label, 'stable');
perClass = zeros(numel(classes), size(feat, 2));
for c = 1:numel(classes)
    indicator = double(label == classes(c));
    indicator = indicator - mean(indicator);
    indicatorNorm = sqrt(sum(indicator .^ 2));
    perClass(c, :) = abs((indicator' * centeredFeat) ./ ...
        (indicatorNorm * featureNorm + eps));
end
scores = mean(perClass, 1);
scores(~isfinite(scores)) = 0;
end


function z = normalize01(x)
x = x(:);
z = (x - min(x)) / (max(x) - min(x) + eps);
end

function mask = guarantee_nonempty(mask, position, candidateMask, rankIndex)
if any(mask), return; end
activeRanked = rankIndex(candidateMask(rankIndex));
if isempty(activeRanked)
    [~, index] = max(position);
else
    [~, localIndex] = max(position(activeRanked));
    index = activeRanked(localIndex);
end
mask(index) = true;
end
