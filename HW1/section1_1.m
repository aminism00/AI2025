
clear; close all; clc;
rng(0);

%% parameters 
numEpochs_full   = 400;  
numEpochs_output = 80;    

N = 400;   % number of samples
d = 10;    % input dimensionality
m = 20;    % hidden units


lr_full = 0.01;      % learning rate 
lr_output_gd = 0.005;% learning rate for output-layer 
lambdaLM_init = 1e-3;% initial LM damping


use_whitening = false; 

X = rand(d, N);
scaleVec = ones(d,1);
scaleVec(1:round(d/2)) = 1;
scaleVec(round(d/2)+1:end) = 1e4;
X = bsxfun(@times, X, scaleVec);


w_true = randn(d,1) .* (1./scaleVec);
y = (w_true' * X) + 2*sin(0.0005*(X(1,:)+X(2,:))) + 0.5*randn(1,N);


X_clip = X;
for j = 1:d
    col = X_clip(j,:);
    med = median(col);
    madv = median(abs(col - med));
    if madv == 0, madv = 1e-6; end
    cutoff = 6 * madv; 
    top = med + cutoff;
    bot = med - cutoff;
    col(col > top) = top;
    col(col < bot) = bot;
    X_clip(j,:) = col;
end

% 2) Standardization 
mu = mean(X_clip,2);
sigma = std(X_clip,0,2);
sigma(sigma==0) = 1;
X_std = bsxfun(@rdivide, bsxfun(@minus, X_clip, mu), sigma);


if use_whitening
    C = cov(X_std');
    [U,Smat] = eig(C);
    Sdiag = diag(Smat);
    [Sdiag_sorted, idx] = sort(Sdiag,'descend');
    U = U(:,idx); Sdiag_sorted = Sdiag_sorted(:);
    eps_reg = 1e-3; % regularizer for small eigenvalues
    Sdiag_inv_sqrt = 1 ./ sqrt(Sdiag_sorted + eps_reg);
    Whiten = diag(Sdiag_inv_sqrt) * U';
    X_proc = Whiten * X_std;
else
    X_proc = X_std;
end


cond_raw = cond(cov(X'));
cond_std = cond(cov(X_std'));
if use_whitening, cond_proc = cond(cov(X_proc')); else cond_proc = NaN; end
fprintf('Condition numbers (covariance): raw=%.3e, standardized=%.3e, whitened=%.3e\n', cond_raw, cond_std, cond_proc);

%%   MLP

W1_raw = 0.05 * randn(m,d); b1_raw = zeros(m,1);
w2_raw = 0.05 * randn(m,1); b2_raw = 0;

W1_proc = 0.05 * randn(m,d); b1_proc = zeros(m,1);
w2_proc = 0.05 * randn(m,1); b2_proc = 0;

loss_raw = zeros(numEpochs_full,1);
loss_proc = zeros(numEpochs_full,1);

fprintf('\nTraining full MLP for %d epochs (full-batch GD)\n', numEpochs_full);
for epoch = 1:numEpochs_full
    % --- raw inputs
    Zr = W1_raw * X + b1_raw * ones(1,N);
    Hr = tanh(Zr);
    Yr = w2_raw' * Hr + b2_raw;
    Er = Yr - y;
    loss_raw(epoch) = 0.5 * mean(Er.^2);
   
    dw2_r = (1/N) * (Hr * Er');
    db2_r = (1/N) * sum(Er,2);
    delta1_r = (w2_raw * Er) .* (1 - Hr.^2);  % m x N
    dW1_r = (1/N) * (delta1_r * X');
    db1_r = (1/N) * sum(delta1_r,2);
    % gradient step
    W1_raw = W1_raw - lr_full * dW1_r;
    b1_raw = b1_raw - lr_full * db1_r;
    w2_raw = w2_raw - lr_full * dw2_r;
    b2_raw = b2_raw - lr_full * db2_r;

    
    Zp = W1_proc * X_proc + b1_proc * ones(1,N);
    Hp = tanh(Zp);
    Yp = w2_proc' * Hp + b2_proc;
    Ep = Yp - y;
    loss_proc(epoch) = 0.5 * mean(Ep.^2);
   
    dw2_p = (1/N) * (Hp * Ep');
    db2_p = (1/N) * sum(Ep,2);
    delta1_p = (w2_proc * Ep) .* (1 - Hp.^2);
    dW1_p = (1/N) * (delta1_p * X_proc');
    db1_p = (1/N) * sum(delta1_p,2);
    % gradient step
    W1_proc = W1_proc - lr_full * dW1_p;
    b1_proc = b1_proc - lr_full * db1_p;
    w2_proc = w2_proc - lr_full * dw2_p;
    b2_proc = b2_proc - lr_full * db2_p;

    
    if mod(epoch, max(1,floor(numEpochs_full/5))) == 0
        fprintf('Epoch %d/%d: loss_raw=%.6e  loss_proc=%.6e\n', epoch, numEpochs_full, loss_raw(epoch), loss_proc(epoch));
    end
end

% Plot MSE 
figure;
plot(1:numEpochs_full, loss_raw, 'LineWidth',1.4); hold on;
plot(1:numEpochs_full, loss_proc, 'LineWidth',1.4);
xlabel('Epoch'); ylabel('Training loss (MSE)');
legend('Raw inputs','Preprocessed inputs','Location','northeast');
title('Full MLP training: raw vs preprocessed inputs');
grid on;

fprintf('Final full-MLP losses: raw = %.6e, processed = %.6e\n', loss_raw(end), loss_proc(end));



W1_fixed = W1_proc; b1_fixed = b1_proc;
H = tanh(W1_fixed * X_proc + b1_fixed * ones(1,N)); 
H_aug = [H; ones(1,N)];                              
P = size(H_aug,1);                                   


JtJ = H_aug * H_aug';  % P x P
condJ = cond(JtJ);
fprintf('\nOutput-layer Gram condition number cond(J^T J) = %.3e (P=%d)\n', condJ, P);


loss_theta = @(theta) 0.5 * mean(((theta' * H_aug) - y).^2);
JtE = @(theta) H_aug * ((theta' * H_aug) - y)';  % P x 1


theta0 = 0.01 * randn(P,1);
theta_GD = theta0; theta_GN = theta0; theta_LM = theta0; theta_Newton = theta0;

loss_GD = zeros(numEpochs_output,1);
loss_GN = zeros(numEpochs_output,1);
loss_LM = zeros(numEpochs_output,1);
loss_Newton = zeros(numEpochs_output,1);

lambda = lambdaLM_init;

fprintf('Comparing output-layer updates for %d iterations\n', numEpochs_output);
for it = 1:numEpochs_output
    % Gradient Descent
    g = (1/N) * JtE(theta_GD);
    theta_GD = theta_GD - lr_output_gd * g;
    loss_GD(it) = loss_theta(theta_GD);

    % Newton Method

    rhs = JtE(theta_Newton);
    delta_Newton = JtJ \ rhs;
    theta_Newton = theta_Newton - delta_Newton;
    loss_Newton(it) = loss_theta(theta_Newton);

    % Gauss-Newton
    
    rhsGN = JtE(theta_GN);
    delta_GN = JtJ \ rhsGN;
    theta_GN = theta_GN - delta_GN;
    loss_GN(it) = loss_theta(theta_GN);

    % Levenberg-Marquardt 

    rhsLM = JtE(theta_LM);
    A = JtJ + lambda * eye(P);
    delta_LM = A \ rhsLM;
    cand_theta = theta_LM - delta_LM;
    loss_cand = loss_theta(cand_theta);
    cur_loss = loss_theta(theta_LM);
    if loss_cand < cur_loss
        theta_LM = cand_theta;
        lambda = lambda / 10; 
    else
        lambda = lambda * 10; 
        
    end
    loss_LM(it) = loss_theta(theta_LM);
end


figure;
plot(1:numEpochs_output, loss_GD,'LineWidth',1.4); hold on;
plot(1:numEpochs_output, loss_GN,'LineWidth',1.4);
plot(1:numEpochs_output, loss_LM,'LineWidth',1.4);
plot(1:numEpochs_output, loss_Newton,'--','LineWidth',1.0);
xlabel('Iteration'); ylabel('Loss (MSE)');
legend('GD (output)','Gauss-Newton','Levenberg-Marquardt','Newton','Location','northeast');
title('Output-layer update methods (hidden frozen)');
grid on;

fprintf('\nFinal output-layer losses after %d iters: GD=%.3e, GN=%.3e, LM=%.3e, Newton=%.3e\n', ...
    numEpochs_output, loss_GD(end), loss_GN(end), loss_LM(end), loss_Newton(end));

fprintf('\nComplexity per iteration (P parameters, N examples):\n');
fprintf(' - Vanilla GD: O(N P) time, O(NP) memory to hold H_aug (or O(P) if streaming). \n');
fprintf(' - Newton / Gauss-Newton / LM: form J^T J costs O(N P^2), solving/invert costs O(P^3). Memory O(NP + P^2).\n');
fprintf('LM is often preferred because damping improves robustness: when far from optimum LM behaves like small-step GD, \nwhile near a good solution it recovers GN/Newton fast convergence.\n');

fprintf('\nScript complete. To change epochs, edit numEpochs_full and numEpochs_output at the top.\n');
