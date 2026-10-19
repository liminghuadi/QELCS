function scores = SU_TOOL(feat, label, numBins)
%SU_TOOL Symmetric uncertainty for continuous features.

if nargin < 3
    numBins = 10;
end
feat = double(feat);
[~, ~, y] = unique(label(:));
n = size(feat, 1);
d = size(feat, 2);
numClasses = max(y);
pyCount = accumarray(y, 1, [numClasses, 1]);
hy = local_entropy(pyCount / n);
scores = zeros(1, d);

for j = 1:d
    x = feat(:, j);
    if any(~isfinite(x))
        finiteX = x(isfinite(x));
        if isempty(finiteX), continue; end
        x(~isfinite(x)) = median(finiteX);
    end
    if max(x) <= min(x), continue; end

    edges = quantile(x, linspace(0, 1, numBins + 1));
    edges = unique(edges(:)');
    if numel(edges) < 3, continue; end
    edges(1) = -inf;
    edges(end) = inf;
    xb = discretize(x, edges);
    bx = numel(edges) - 1;

    jointCount = accumarray([xb, y], 1, [bx, numClasses]);
    pxy = jointCount / n;
    px = sum(pxy, 2);
    py = sum(pxy, 1);
    hx = local_entropy(px);
    expected = px * py;
    nz = pxy > 0 & expected > 0;
    mi = sum(pxy(nz) .* log2(pxy(nz) ./ expected(nz)));
    denom = hx + hy;
    if denom > 0
        scores(j) = max(0, min(1, 2 * mi / denom));
    end
end
end

function h = local_entropy(p)
p = p(p > 0);
h = -sum(p .* log2(p));
end
