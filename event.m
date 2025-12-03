https://matlab-1b.mathworks.com/matlab/ui/R563ba786de/ui/webgui/src/window.html?winId=fff47026-9d58-44f2-8a85-95956807e5aa&websocket=on

Contents
Density–Metric Manifold Demo (5D, PCA, Density, Separability)
1. Load data from CSV if available, otherwise create synthetic 5D data
2. PCA (2 components) and scatter plot
3. Epsilon-neighbourhood density in 5D
4. Density histograms for normal vs. anomalous
5. Density-based anomaly detection and separability table
6. (Optional) Print some summary stats of densities
Density–Metric Manifold Demo (5D, PCA, Density, Separability)
This script: 1) Loads a 5D dataset with a 'label' column, OR creates a synthetic one. 2) Computes PCA and plots PC1 vs PC2. 3) Computes epsilon-neighbourhood densities. 4) Plots density histograms for normal vs anomalous events. 5) Computes a separability table based on MinPts.

clear; clc; close all;
1. Load data from CSV if available, otherwise create synthetic 5D data
fileName = "data5D.csv";   % If you later have a real file, put it here

if isfile(fileName)
    % --- REAL DATA PATH ---
    T = readtable(fileName);            % assumes first 5 columns are d1..d5
    X = table2array(T(:,1:5));          % 5D features
    if any(strcmpi(T.Properties.VariableNames, "label"))
        label = T.label;                % 0 = normal, 1 = anomaly
    else
        error("Column 'label' not found in the table.");
    end
else
    % --- SYNTHETIC DATA PATH ---
    rng(42);                      % for reproducibility
    nNormal = 400;
    nAnom   = 40;

    normal = randn(nNormal, 5);          % N(0, I5)
    anom   = 3 + 0.7*randn(nAnom, 5);    % N(3, 0.7^2 I5)

    X     = [normal; anom];
    label = [zeros(nNormal,1); ones(nAnom,1)];   % 0 = normal, 1 = anomaly

    % (Optional) save so you can inspect in Excel / Stata later
    % T = array2table(X, "VariableNames", {'d1','d2','d3','d4','d5'});
    % T.label = label;
    % writetable(T, "data5D.csv");
end
2. PCA (2 components) and scatter plot
[coeff, score, latent, tsq, explained] = pca(X);
PC1 = score(:,1);
PC2 = score(:,2);

idxNormal = (label == 0);
idxAnom   = (label == 1);

figure;
scatter(PC1(idxNormal), PC2(idxNormal), 25, "b", "filled"); hold on;
scatter(PC1(idxAnom),   PC2(idxAnom),   40, "r", "x", "LineWidth", 1.5);
xlabel("PC1");
ylabel("PC2");
title(sprintf("PCA Projection of the 5D Manifold (Var: %.1f%% + %.1f%%)", ...
      explained(1), explained(2)));
legend("Normal", "Anomalous", "Location", "best");
grid on;

3. Epsilon-neighbourhood density in 5D
% Requires Statistics and Machine Learning Toolbox for pdist + squareform
D = squareform(pdist(X));   % full pairwise distance matrix
eps = 1.0;                  % epsilon radius, same as in the paper

density = sum(D <= eps, 2); % number of neighbours within radius eps
% Note: includes the point itself (distance 0)
4. Density histograms for normal vs. anomalous
figure;
histogram(density(idxNormal), "BinMethod", "integers", "FaceAlpha", 0.6);
hold on;
histogram(density(idxAnom), "BinMethod", "integers", "FaceAlpha", 0.6);
xlabel("Local density \rho(x)");
ylabel("Count");
title("Density Distribution: Normal vs. Anomalous Events");
legend("Normal", "Anomalous", "Location", "best");
grid on;

5. Density-based anomaly detection and separability table
MinPts = 5;   % density threshold, as in the article

isAnomByDensity = density < MinPts;   % 1 = anomaly by density rule

% Confusion-like table: rows = true class, cols = detected as anomaly?
[confMat, grpTrue, grpPred] = crosstab(label, isAnomByDensity);

disp("Confusion-like table (rows: true class, cols: detected as anomaly):");
disp(array2table(confMat, ...
     "VariableNames", {'DetectedNormal_0','DetectedAnomaly_1'}, ...
     "RowNames", {'TrueNormal_0','TrueAnomaly_1'}));
Confusion-like table (rows: true class, cols: detected as anomaly):
                     DetectedNormal_0    DetectedAnomaly_1
                     ________________    _________________

    TrueNormal_0           126                  274       
    TrueAnomaly_1            0                   40       

6. (Optional) Print some summary stats of densities
normalDensity = density(idxNormal);
anomDensity   = density(idxAnom);

fprintf("\nNormal density: mean = %.2f, median = %.2f, min = %d, max = %d\n", ...
        mean(normalDensity), median(normalDensity), ...
        min(normalDensity), max(normalDensity));

fprintf("Anomalous density: mean = %.2f, median = %.2f, min = %d, max = %d\n\n", ...
        mean(anomDensity), median(anomDensity), ...
        min(anomDensity), max(anomDensity));
Normal density: mean = 3.78, median = 3.00, min = 1, max = 13
Anomalous density: mean = 1.85, median = 2.00, min = 1, max = 4


Published with MATLAB® R2025b
