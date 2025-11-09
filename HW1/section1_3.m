
clear; close all; rng(0);

%%  Generate nonlinear time series
T = 2000;
x = zeros(T,1);
x(1:2) = [0.1; -0.2];
for t = 3:T
    x(t) = 0.6*x(t-1) - 0.35*x(t-2) + 0.25*sin(x(t-1)) + 0.1*randn();
end


mu = mean(x); sigma = std(x);
xs = (x - mu) / sigma;


train_frac = 0.8;
Ttrain = floor(T * train_frac);
x_train = xs(1:Ttrain);
x_test  = xs(Ttrain+1:end);


p = 10;   
[X_train, Y_train] = make_windows(x_train, p);
[X_test,  Y_test ] = make_windows(x_test,  p);


function [X, Y] = make_windows(series, p)
    N = length(series) - p;
    X = zeros(p, N);
    Y = zeros(1, N);
    for i = 1:N
        X(:,i) = series(i:i+p-1);
        Y(1,i)  = series(i+p);
    end
end


fprintf('Training MLP ...\n');
nx = p; nh = 30; ny = 1;

W1 = 0.1*randn(nh, nx); b1 = zeros(nh,1);
W2 = 0.1*randn(ny, nh); b2 = zeros(ny,1);

epochs = 200;
lr = 0.01;
batch = 64;
Ntrain = size(X_train,2);

for ep = 1:epochs
    idx = randperm(Ntrain);
    for k = 1:batch:Ntrain
        batch_idx = idx(k:min(k+batch-1,Ntrain));
        Xb = X_train(:,batch_idx);   
        Yb = Y_train(:,batch_idx);   
        % forward
        Z1 = W1*Xb + b1;            
        A1 = tanh(Z1);              
        Z2 = W2*A1 + b2;             
        Yhat = Z2;                  
        % MSE
        E = (Yhat - Yb);             
        dW2 = (E * A1') / size(Xb,2);
        db2 = mean(E,2);
        dA1 = W2' * E;              
        dZ1 = dA1 .* (1 - A1.^2);    
        dW1 = (dZ1 * Xb') / size(Xb,2);
        db1 = mean(dZ1,2);
        % update
        W2 = W2 - lr * dW2;
        b2 = b2 - lr * db2;
        W1 = W1 - lr * dW1;
        b1 = b1 - lr * db1;
    end
 
    if mod(ep,50)==0
     
        Yhat_train = W2 * tanh(W1 * X_train + b1) + b2;
        L = mean(0.5*(Yhat_train - Y_train).^2);
        fprintf('  epoch %d   train MSE=%.5f\n', ep, L);
    end
end

% evaluate on test
Yhat_mlp = W2 * tanh(W1 * X_test + b1) + b2;
mse_mlp = mean((Yhat_mlp - Y_test).^2);
fprintf('MLP test MSE = %.5f\n\n', mse_mlp);


fprintf('Training simple vanilla RNN (BPTT)...\n');

hidden = 40;
Wx = 0.1*randn(hidden,1);    
Wh = 0.1*randn(hidden,hidden); 
bh = zeros(hidden,1);
Wy = 0.1*randn(1,hidden);
by = 0;
lr_rnn = 0.005;
epochs_rnn = 40;
trunc = 50;   
clip_grad = 5;


seq = x_train(:)';
Tseq = length(seq);

for ep = 1:epochs_rnn
    h_prev = zeros(hidden,1);
    total_loss = 0;
    t = 1;
    while t <= Tseq - 1
       
        t_end = min(t + trunc - 1, Tseq-1);
        L = t_end - t + 1;
        xs_chunk = seq(t:t_end);       
        ys_chunk = seq(t+1:t_end+1);   
        hs = zeros(hidden, L);
        zs = zeros(hidden, L);
        yhs = zeros(1, L);
        h = h_prev;
        for k = 1:L
            z = Wh * h + Wx * xs_chunk(k) + bh;
            h = tanh(z);
            yhat = Wy * h + by;
            hs(:,k) = h;
            zs(:,k) = z;
            yhs(:,k) = yhat;
        end
      
        e = yhs - ys_chunk;  
        total_loss = total_loss + sum(0.5*e.^2);
      
        dWh = zeros(size(Wh)); dWx = zeros(size(Wx)); dbh = zeros(size(bh));
        dWy = zeros(size(Wy)); dby = 0;
        dh_next = zeros(hidden,1);
        for k = L:-1:1
            dy = e(k);                      
            dWy = dWy + dy * hs(:,k)';      
            dby = dby + dy;
            dh = (Wy' * dy) + dh_next;      
            dz = dh .* (1 - hs(:,k).^2);    
            dbh = dbh + dz;
            dWx = dWx + dz * xs_chunk(k);
            if k==1
                h_prev_for = h_prev;
            else
                h_prev_for = hs(:,k-1);
            end
            dWh = dWh + dz * h_prev_for';
            dh_next = Wh' * dz;
        end

        grad_norm = sqrt(sum(dWh(:).^2) + sum(dWx(:).^2) + sum(dbh(:).^2) + sum(dWy(:).^2) + dby^2);
        if grad_norm > clip_grad
            scale = clip_grad / grad_norm;
            dWh = dWh * scale; dWx = dWx * scale; dbh = dbh * scale;
            dWy = dWy * scale; dby = dby * scale;
        end
        % SGD 
        Wh = Wh - lr_rnn * dWh / L;
        Wx = Wx - lr_rnn * dWx / L;
        bh = bh - lr_rnn * dbh / L;
        Wy = Wy - lr_rnn * dWy / L;
        by = by - lr_rnn * dby / L;
    
        h_prev = hs(:,end);
        t = t + L;
    end
    if mod(ep,10)==0
        fprintf('  epoch %d   train loss (sum) = %.4f\n', ep, total_loss);
    end
end


h = zeros(hidden,1);
y_preds = zeros(length(x_test)-1,1);
for i = 1:length(x_test)-1
    h = tanh(Wh * h + Wx * x_test(i) + bh);
    y_pred = Wy * h + by;
    y_preds(i) = y_pred;
end
Y_true_rnn = x_test(2:end);
mse_rnn = mean((y_preds - Y_true_rnn).^2);
fprintf('RNN test MSE = %.5f\n\n', mse_rnn);


if exist('trainNetwork','file') == 2
    fprintf('Training LSTM using trainNetwork ...\n');
    seqLen = 50;
    Xs = {}; Ys = {};
    for i = 1:(length(x_train)-seqLen)
        Xs{i} = x_train(i:i+seqLen-1)';
        Ys{i} = x_train(i+seqLen)';   % scalar
    end
    % network
    layers = [ sequenceInputLayer(1)
               lstmLayer(40,'OutputMode','last')
               fullyConnectedLayer(1)
               regressionLayer ];
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.005, ...
        'Shuffle','every-epoch', ...
        'Verbose',false);
    try
        net = trainNetwork(Xs', Ys', layers, options);
      
        Ypreds_lstm = zeros(length(x_test)-seqLen,1);
        for i = 1:(length(x_test)-seqLen)
            seq_in = x_test(i:i+seqLen-1)';
            Ypreds_lstm(i) = predict(net, seq_in);
        end
        Ytrue_lstm = x_test(seqLen+1:end);
        mse_lstm = mean((Ypreds_lstm - Ytrue_lstm).^2);
        fprintf('LSTM test MSE = %.5f\n\n', mse_lstm);
    catch ME
        fprintf('trainNetwork call failed or aborted: %s\n', ME.message);
    end
else
    fprintf('trainNetwork not found. Skipping LSTM example.\n\n');
end

fprintf('Summary (test MSE):\n');
fprintf('  MLP  (sliding window) : %.5f\n', mse_mlp);
fprintf('  RNN  (vanilla BPTT)   : %.5f\n', mse_rnn);
if exist('mse_lstm','var'), fprintf('  LSTM (trainNetwork)   : %.5f\n', mse_lstm); end



