
clear; close all; clc;
rng(0);

% User parameter
last_digit_of_student_id = 0; 
H = mod(last_digit_of_student_id, 3) + 2;
fprintf('Hidden units H = %d\n', H);


T = readtable('Social_Network_Ads.csv');

vars = T.Properties.VariableNames;
vars_l = lower(vars);

%  Age
iAge = find(contains(vars_l,'age'), 1);
if isempty(iAge)
    error('No column matching "age" found.');
end
% find EstimatedSalary 
iSalary = find(contains(vars_l,'salary') | contains(vars_l,'estimated'), 1);
if isempty(iSalary)
    error('No column matching "salary" or "estimated" found.');
end
% find Purchased
iPurchased = find(contains(vars_l,'purchas'), 1);
if isempty(iPurchased)
    error('No column matching "purchased" found.');
end

age = T{:, iAge};
salary = T{:, iSalary};
y = T{:, iPurchased};


if iscell(y)
    y = str2double(y);
end
y = double(y);
if ~all(ismember(unique(y(~isnan(y))), [0 1]))
    error('Purchased column must contain 0/1 values.');
end

n = numel(y);
fprintf('Loaded %d rows. Columns used: %s, %s, %s\n', n, vars{iAge}, vars{iSalary}, vars{iPurchased});
fprintf('Missing values: Age=%d, Salary=%d, Purchased=%d\n', sum(isnan(age)), sum(isnan(salary)), sum(isnan(y)));


fprintf('\n== EDA ==\n');
fprintf('Age: mean=%.2f std=%.2f min=%.1f max=%.1f\n', mean(age), std(age), min(age), max(age));
fprintf('EstimatedSalary: mean=%.2f std=%.2f min=%.1f max=%.1f\n', mean(salary), std(salary), min(salary), max(salary));

figure('Name','Histograms & Boxplots','NumberTitle','off');
subplot(2,2,1); histogram(age); title('Age histogram'); xlabel('Age');
subplot(2,2,2); histogram(salary); title('EstimatedSalary histogram'); xlabel('Salary');
subplot(2,2,3); boxplot(age); title('Age boxplot');
subplot(2,2,4); boxplot(salary); title('Salary boxplot');

figure('Name','Age vs Salary by Purchase','NumberTitle','off');
gscatter(age, salary, y, 'br', 'ox'); xlabel('Age'); ylabel('Estimated Salary');
title('Age vs Salary (color = Purchased)');

figure('Name','Age-Salary density','NumberTitle','off');
histogram2(age, salary, [20 20], 'DisplayStyle','tile'); colorbar;
xlabel('Age'); ylabel('EstimatedSalary'); title('2D histogram/density');

R = corrcoef([age, salary, y]);
disp('Correlation matrix (Age, Salary, Purchased):'); disp(R);


X = [age, salary];
mu = mean(X,1);
sigma = std(X,[],1);
Xz = (X - mu) ./ sigma;

idx = randperm(n);
nTrain = round(0.70 * n);
nVal   = round(0.15 * n);
train_idx = idx(1:nTrain);
val_idx   = idx(nTrain+1:nTrain+nVal);
test_idx  = idx(nTrain+nVal+1:end);

X_train = Xz(train_idx, :); y_train = y(train_idx);
X_val   = Xz(val_idx, :);   y_val   = y(val_idx);
X_test  = Xz(test_idx, :);  y_test  = y(test_idx);


% --- Manual gradient descent ---
Xtr = [ones(size(X_train,1),1), X_train];
Xv  = [ones(size(X_val,1),1), X_val];
Xt  = [ones(size(X_test,1),1), X_test];

sigmoid = @(z) 1 ./ (1 + exp(-z));
bce = @(p,t) -mean(t .* log(max(p,eps)) + (1-t) .* log(max(1-p,eps)));

alpha = 0.1;
maxEpochs = 5000;
tol = 1e-9;
w = zeros(size(Xtr,2),1);
trainLoss = zeros(maxEpochs,1);
valLoss = zeros(maxEpochs,1);

for epoch=1:maxEpochs
    p = sigmoid(Xtr * w);
    loss = bce(p, y_train);
    grad = (Xtr' * (p - y_train)) / numel(y_train);
    w = w - alpha * grad;
    trainLoss(epoch) = loss;
    valLoss(epoch) = bce(sigmoid(Xv * w), y_val);
    if epoch>10 && abs(trainLoss(epoch)-trainLoss(epoch-1))<tol
        trainLoss = trainLoss(1:epoch); valLoss = valLoss(1:epoch);
        break;
    end
end

p_test_manual = sigmoid(Xt * w);
yhat_test_manual = double(p_test_manual >= 0.5); 
acc_manual = mean(yhat_test_manual == y_test);
fprintf('\nLogistic (manual) test accuracy: %.3f\n', acc_manual);


true_cat = categorical(y_test); pred_cat = categorical(yhat_test_manual);
figure('Name','Confusion - Logistic Manual','NumberTitle','off');
confusionchart(true_cat, pred_cat, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion - Logistic Manual');

[fp_m, tp_m, ~, auc_m] = perfcurve(y_test, p_test_manual, 1);
figure('Name','ROC - Logistic Manual','NumberTitle','off');
plot(fp_m, tp_m, 'LineWidth',1.5); xlabel('FPR'); ylabel('TPR');
title(sprintf('ROC - Logistic Manual (AUC=%.3f)', auc_m)); grid on;

% Decision boundary (manual)
w_manual = w;
age_plot = linspace(min(age)-2, max(age)+2, 200);
z_age_plot = (age_plot - mu(1)) / sigma(1);
z_salary_on_boundary = -(w_manual(1) + w_manual(2) * z_age_plot) / w_manual(3);
salary_on_boundary = z_salary_on_boundary * sigma(2) + mu(2);

figure('Name','Decision Boundary - Logistic Manual','NumberTitle','off');
gscatter(age, salary, y, 'br', 'ox'); hold on;
plot(age_plot, salary_on_boundary, 'k-', 'LineWidth', 2);
xlabel('Age'); ylabel('Estimated Salary'); title('Decision boundary - Logistic Manual'); grid on;
legend({'Not Purchased','Purchased','Decision boundary'}, 'Location','best');


Ttrain = table(X_train(:,1), X_train(:,2), y_train, 'VariableNames', {'Age_z','Salary_z','Purchased'});
glm = fitglm(Ttrain, 'Purchased ~ Age_z + Salary_z', 'Distribution','binomial');
Ttest = table(X_test(:,1), X_test(:,2), 'VariableNames', {'Age_z','Salary_z'});
p_test_glm = predict(glm, Ttest);
yhat_test_glm = double(p_test_glm >= 0.5); 
acc_glm = mean(yhat_test_glm == y_test);
fprintf('Logistic (built-in fitglm) test accuracy: %.3f\n', acc_glm);


true_cat = categorical(y_test); pred_cat_glm = categorical(yhat_test_glm);
figure('Name','Confusion - Logistic Built-in','NumberTitle','off');
confusionchart(true_cat, pred_cat_glm, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion - Logistic Built-in');

[fp_g, tp_g, ~, auc_g] = perfcurve(y_test, p_test_glm, 1);
figure('Name','ROC - Logistic Built-in','NumberTitle','off');
plot(fp_g, tp_g, 'LineWidth',1.5); xlabel('FPR'); ylabel('TPR');
title(sprintf('ROC - Logistic Built-in (AUC=%.3f)', auc_g)); grid on;


coef = glm.Coefficients.Estimate; 
b = coef;
z_salary_on_boundary_b = -(b(1) + b(2) * z_age_plot) / b(3);
salary_on_boundary_b = z_salary_on_boundary_b * sigma(2) + mu(2);

figure('Name','Decision Boundaries - Logistic','NumberTitle','off');
gscatter(age, salary, y, 'br', 'ox'); hold on;
plot(age_plot, salary_on_boundary, 'k-', 'LineWidth', 2);
plot(age_plot, salary_on_boundary_b, 'g--', 'LineWidth', 2);
xlabel('Age'); ylabel('Estimated Salary'); title('Decision boundaries (manual vs built-in)'); grid on;
legend({'Not Purchased','Purchased','Manual','Built-in'}, 'Location','best');

Xtr_col = X_train'; Ytr_col = y_train';
Xv_col = X_val'; Yv_col = y_val';
Xt_col = X_test'; Yt_col = y_test';


eps_init = 0.12;
W1 = rand(H, 2) * 2*eps_init - eps_init; b1 = zeros(H,1);
W2 = rand(1, H) * 2*eps_init - eps_init; b2 = 0;

sig = @(z) 1 ./ (1 + exp(-z));
bce_col = @(p,t) -mean(t .* log(max(p,eps)) + (1-t) .* log(max(1-p,eps)));

lr_nn = 0.5;
maxEpochs_nn = 5000;
patience = 200;

best_val_loss = Inf; best_epoch = 0;
trainLoss_nn = nan(maxEpochs_nn,1);
valLoss_nn = nan(maxEpochs_nn,1);
trainAcc_nn = nan(maxEpochs_nn,1);
valAcc_nn = nan(maxEpochs_nn,1);

m = size(Xtr_col,2);

for epoch=1:maxEpochs_nn
    % forward train
    Z1 = W1 * Xtr_col + b1; A1 = sig(Z1);
    Z2 = W2 * A1 + b2; A2 = sig(Z2);
    loss_tr = bce_col(A2, Ytr_col);
    % forward val
    Z1v = W1 * Xv_col + b1; A1v = sig(Z1v);
    Z2v = W2 * A1v + b2; A2v = sig(Z2v);
    loss_val = bce_col(A2v, Yv_col);
    % accuracy
    acc_tr = mean(double(A2 >= 0.5) == Ytr_col);
    acc_val = mean(double(A2v >= 0.5) == Yv_col);
    trainLoss_nn(epoch) = loss_tr; valLoss_nn(epoch) = loss_val;
    trainAcc_nn(epoch) = acc_tr; valAcc_nn(epoch) = acc_val;

    if loss_val < best_val_loss - 1e-9
        best_val_loss = loss_val; best_epoch = epoch;
        bestW1 = W1; bestb1 = b1; bestW2 = W2; bestb2 = b2;
    end

    if epoch - best_epoch >= patience
        trainLoss_nn = trainLoss_nn(1:epoch);
        valLoss_nn = valLoss_nn(1:epoch);
        trainAcc_nn = trainAcc_nn(1:epoch);
        valAcc_nn = valAcc_nn(1:epoch);
        break;
    end

    delta2 = A2 - Ytr_col; dW2 = (delta2 * A1') / m; db2 = mean(delta2,2);
    delta1 = (W2' * delta2) .* (A1 .* (1 - A1)); dW1 = (delta1 * Xtr_col') / m; db1 = mean(delta1,2);

    W2 = W2 - lr_nn * dW2; b2 = b2 - lr_nn * db2;
    W1 = W1 - lr_nn * dW1; b1 = b1 - lr_nn * db1;
end

W1 = bestW1; b1 = bestb1; W2 = bestW2; b2 = bestb2;


Z1t = W1 * Xt_col + b1; A1t = sig(Z1t);
Z2t = W2 * A1t + b2; A2t = sig(Z2t);
yhat_test_nn_manual = double(A2t >= 0.5); 
acc_nn_manual = mean(yhat_test_nn_manual' == y_test);
fprintf('\nManual NN test accuracy: %.3f\n', acc_nn_manual);


true_cat = categorical(y_test); pred_cat_nn_manual = categorical(yhat_test_nn_manual');
figure('Name','Confusion - Manual NN','NumberTitle','off');
confusionchart(true_cat, pred_cat_nn_manual, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion - Manual NN');


inputs_all = Xz'; targets_all = y';
net = patternnet(H);
net.divideFcn = 'divideind';
net.divideParam.trainInd = train_idx;
net.divideParam.valInd   = val_idx;
net.divideParam.testInd  = test_idx;
for li = 1:numel(net.layers); net.layers{li}.transferFcn = 'logsig'; end
[net, tr] = train(net, inputs_all, targets_all);
y_pred_all = net(inputs_all);
ytest_built_prob = y_pred_all(:, test_idx);
ytest_built_label = double(ytest_built_prob >= 0.5); 
acc_nn_built = mean(ytest_built_label' == y_test);
fprintf('Built-in NN (patternnet) test accuracy: %.3f\n', acc_nn_built);


true_cat = categorical(y_test); pred_cat_nn_built = categorical(ytest_built_label');
figure('Name','Confusion - Built-in NN','NumberTitle','off');
confusionchart(true_cat, pred_cat_nn_built, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
title('Confusion - Built-in NN');


figure('Name','Manual NN Learning Curves','NumberTitle','off');
subplot(2,1,1);
plot(1:numel(trainLoss_nn), trainLoss_nn, 'b-', 1:numel(valLoss_nn), valLoss_nn, 'r--', 'LineWidth',1.2);
xlabel('Epoch'); ylabel('Loss'); legend('Train','Val'); title('Manual NN Loss'); grid on;
subplot(2,1,2);
plot(1:numel(trainAcc_nn), trainAcc_nn, 'b-', 1:numel(valAcc_nn), valAcc_nn, 'r--', 'LineWidth',1.2);
xlabel('Epoch'); ylabel('Accuracy'); legend('Train','Val'); title('Manual NN Accuracy'); grid on;


age_range = linspace(min(age)-2, max(age)+2, 200);
sal_range = linspace(min(salary)-2000, max(salary)+2000, 200);
[Agrid, Sgrid] = meshgrid(age_range, sal_range);
gridPts = [Agrid(:), Sgrid(:)];
gridZ = (gridPts - mu) ./ sigma;
gridZt = gridZ';


Z1g = W1 * gridZt + b1; Ag1 = sig(Z1g);
Z2g = W2 * Ag1 + b2; Ag2 = sig(Z2g);
probs_manual_grid = reshape(Ag2', size(Agrid));

figure('Name','Decision Boundary - Manual NN','NumberTitle','off');
contourf(Agrid, Sgrid, probs_manual_grid, [0 0.5 1], 'LineStyle','none'); hold on;
contour(Agrid, Sgrid, probs_manual_grid, [0.5 0.5], 'k-', 'LineWidth', 2);
gscatter(age(test_idx), salary(test_idx), y(test_idx), 'br', 'ox');
xlabel('Age'); ylabel('Estimated Salary'); title('Manual NN Decision Boundary (0.5) with Test Data'); grid on;


probs_built_grid = reshape(net(gridZt)', size(Agrid));
figure('Name','Decision Boundary - Built-in NN','NumberTitle','off');
contourf(Agrid, Sgrid, probs_built_grid, [0 0.5 1], 'LineStyle','none'); hold on;
contour(Agrid, Sgrid, probs_built_grid, [0.5 0.5], 'k--', 'LineWidth', 2);
gscatter(age(test_idx), salary(test_idx), y(test_idx), 'br', 'ox');
xlabel('Age'); ylabel('Estimated Salary'); title('Built-in NN Decision Boundary (0.5) with Test Data'); grid on;


fprintf('\n--- Final summary ---\n');
fprintf('Logistic manual test acc: %.3f, AUC: %.3f\n', acc_manual, auc_m);
fprintf('Logistic built-in test acc: %.3f, AUC: %.3f\n', acc_glm, auc_g);
fprintf('Manual NN test acc: %.3f\n', acc_nn_manual);
fprintf('Built-in NN test acc: %.3f\n', acc_nn_built);

fprintf('\nAnalysis (brief):\n');
fprintf('- Convergence: built-in trainers generally converge faster and use optimized algorithms.\n');
fprintf('- Overfitting & capacity: H controls capacity; monitor train vs val curves; early stopping is used for manual NN.\n');
fprintf('- Manual implementation gives full control; built-in offers robust defaults and usually better training heuristics.\n');


