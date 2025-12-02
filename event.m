Contents
============================================================
1. Load data from CSV
2. Standardise features (recommended for PCA & distances)
3. PCA on the 5D data
4. PCA scatter plot: Normal vs Anomalous
5. Epsilon-neighbourhood density computation
6. Density histograms: normal vs anomalous
7. Density-based anomaly flag and separability table
8. (Optional) Save figures for MDPI
============================================================
5D Behavioural Manifold Analysis in MATLAB
- Reads 5D data + labels from CSV
- Performs PCA
- Plots PCA projection (Normal vs Anomalous)
- Computes epsilon-neighbourhood density
- Plots density histograms
- Builds separability table (density vs true label)
============================================================
clear; clc; close all;
1. Load data from CSV
Make sure 'data5D.csv' is in the current MATLAB folder. The file should at least contain: d1, d2, d3, d4, d5, label where label = 0 for normal, 1 for anomalous.

filename = 'data5D.csv';   % <-- change this if your file has another name

if ~isfile(filename)
    error('File "%s" not found. Put it in the same folder or change the name.', filename);
end

T = readtable(filename);

% If your column names are different, change them here:
featureNames = {'d1','d2','d3','d4','d5'};  % 5D manifold coordinates
labelName    = 'label';                     % class label (0/1)

X = T{:, featureNames};   % feature matrix (N x 5)
label = T.(labelName);    % class label vector (N x 1)
2. Standardise features (recommended for PCA & distances)
Xz = zscore(X);   % zero-mean, unit-variance per dimension
3. PCA on the 5D data
[coeff, score, ~, ~, explained] = pca(Xz);

PC1 = score(:,1);
PC2 = score(:,2);
4. PCA scatter plot: Normal vs Anomalous
idxNormal = (label == 0);
idxAnom   = (label == 1);

figure;
scatter(PC1(idxNormal), PC2(idxNormal), 25, 'b', 'filled'); hold on;
scatter(PC1(idxAnom),   PC2(idxAnom),   40, 'r', 'x', 'LineWidth', 1.5);
grid on;
xlabel(sprintf('PC1 (%.1f%%%% variance)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%%% variance)', explained(2)));
title('PCA Projection of the 5D Density–Metric Manifold');
legend({'Normal','Anomalous'}, 'Location','best');
set(gcf,'Color','w');
5. Epsilon-neighbourhood density computation
We compute density as the number of neighbours within radius eps using Euclidean distance in the standardised 5D space.

eps = 1.0;   % neighbourhood radius (tune if needed)

% Compute pairwise distances
D = squareform(pdist(Xz, 'euclidean'));   % requires Statistics and Machine Learning Toolbox

% Density: number of points within distance <= eps (including itself)
density = sum(D <= eps, 2);
6. Density histograms: normal vs anomalous
figure;
histogram(density(idxNormal), 'FaceAlpha',0.6, 'FaceColor',[0 0 1]); hold on;
histogram(density(idxAnom),   'FaceAlpha',0.6, 'FaceColor',[1 0 0]);
grid on;
xlabel('Neighbourhood density  \rho(x)  (number of points within \epsilon)');
ylabel('Count');
title(sprintf('Density Histogram (\\epsilon = %.2f)', eps));
legend({'Normal','Anomalous'}, 'Location','best');
set(gcf,'Color','w');
7. Density-based anomaly flag and separability table
MinPts = 5;   % density threshold (as in the paper)

isAnomByDensity = density < MinPts;   % 1 = anomaly, 0 = normal

% Confusion-style table: rows = true class, columns = detected as anomaly
[confMat, grpLabelsTrue, grpLabelsDetected] = crosstab(label, isAnomByDensity);

disp('Confusion table (rows = true class, columns = detected as anomaly):');
disp(array2table(confMat, ...
    'VariableNames', { 'DetectedNormal_0', 'DetectedAnomaly_1' }, ...
    'RowNames', {'TrueNormal_0','TrueAnomaly_1'}));

% Optional: basic rates
TP = confMat(2,2);  % true anomalies flagged
FN = confMat(2,1);  % anomalies missed
FP = confMat(1,2);  % normals flagged as anomalies
TN = confMat(1,1);  % normals correctly kept

TPR = TP / (TP + FN);   % recall / sensitivity
FPR = FP / (FP + TN);   % false positive rate

fprintf('\nTrue Positive Rate (TPR / Recall): %.2f\n', TPR);
fprintf('False Positive Rate (FPR): %.2f\n', FPR);
8. (Optional) Save figures for MDPI
saveas(gcf, 'density_histogram.png'); saveas(gcf, 'pca_projection.png');


Published with MATLAB® R2025b
