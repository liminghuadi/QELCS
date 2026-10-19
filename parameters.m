function opts = parameters(mode)
if nargin < 1, mode = 'paper'; end
opts.mode = lower(char(mode));
opts.seed = 42;
opts.k = 1;
opts.outerK = 10;
opts.innerK = 5;
opts.verbose = false;

switch opts.mode
    case 'paper'
        opts.N = 100;
        opts.T = 100;
        opts.runNum = 5;
    case 'quick'
        opts.N = 10;
        opts.T = 10;
        opts.runNum = 1;
    otherwise
        error('qelcs:UnknownMode', 'Mode must be paper or quick.');
end

% Particle swarm.
opts.c1 = 1.49445;
opts.c2 = 1.49445;
opts.wMax = 0.9;
opts.wMin = 0.4;
opts.vMax = 0.5;

% Stable dual-score filtering and progressive initialization.
opts.initialCoreRatio = 0.05;
opts.suBins = 10;
opts.suCandidateRatio = 0.10;
opts.classBalanceWeight = 0.30;
opts.suStabilityPenalty = 0.25;

% Cross-guided dynamic thresholds.
opts.thresholdAccuracyStart = 0.40;
opts.thresholdSparseStart = 0.60;
opts.thresholdConverged = 0.50;
opts.thresholdDecayExponent = 2.0;

% Robust elite archive.
opts.archiveCapacity = max(10, ceil(0.30 * opts.N));
opts.archiveRetentionTolerance = 0.05;
opts.archiveGuideToleranceStart = 0.03;
opts.archiveGuideToleranceEnd = 0.005;
opts.archiveRobustnessWeight = 0.50;
opts.archiveConsensusFraction = 0.25;
end
