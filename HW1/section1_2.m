
clear; clc; rng(0);

% Given example
x = [0.5; -0.3];    
y = 0.8;            
eta = 0.1;          


W1 = 0.1 * randn(3,2);   
b1 = 0.1 * randn(3,1);  
W2 = 0.1 * randn(2,3);   
b2 = 0.1 * randn(2,1);   
W3 = 0.1 * randn(1,2);  
b3 = 0.1 * randn(1,1);  


W1_orig = W1; b1_orig = b1;
W2_orig = W2; b2_orig = b2;
W3_orig = W3; b3_orig = b3;


sig = @(z) 1 ./ (1 + exp(-z));


z1 = W1 * x + b1;      
a1 = sig(z1);          

z2 = W2 * a1 + b2;     
a2 = sig(z2);          

z3 = W3 * a2 + b3;     
a3 = sig(z3);          

yhat_before = a3;
L_before = 0.5 * (y - yhat_before)^2;   % MSE loss


fprintf('--- Forward pass ---\n');
fprintf('z1 =\n'); disp(z1);
fprintf('a1 =\n'); disp(a1);
fprintf('z2 =\n'); disp(z2);
fprintf('a2 =\n'); disp(a2);
fprintf('z3 = %g   yhat = %g\n', z3, a3);
fprintf('Loss L = %g\n\n', L_before);


%  Backward propagation 

sigp3 = a3 .* (1 - a3);         
dL_da3 = (a3 - y);             
delta3 = dL_da3 * sigp3;        
dW3 = delta3 * a2';             
db3 = delta3;                   

sigp2 = a2 .* (1 - a2);         
delta2 = (W3' * delta3) .* sigp2;   
dW2 = delta2 * a1';                 
db2 = delta2;                      

sigp1 = a1 .* (1 - a1);         
delta1 = (W2' * delta2) .* sigp1;  
dW1 = delta1 * x';                  
db1 = delta1;                       


fprintf('--- Backpropagation gradients ---\n');
fprintf('dW3 =\n'); disp(dW3);
fprintf('db3 =\n'); disp(db3);
fprintf('dW2 =\n'); disp(dW2);
fprintf('db2 =\n'); disp(db2);
fprintf('dW1 =\n'); disp(dW1);
fprintf('db1 =\n'); disp(db1);


W1 = W1 - eta * dW1;
b1 = b1 - eta * db1;
W2 = W2 - eta * dW2;
b2 = b2 - eta * db2;
W3 = W3 - eta * dW3;
b3 = b3 - eta * db3;


fprintf('--- Weights before and ---\n', eta);
fprintf('W1 before:\n'); disp(W1_orig);      % original copy
fprintf('W1 after:\n');  disp(W1);
fprintf('b1 before:\n'); disp(b1_orig);
fprintf('b1 after:\n');  disp(b1);

fprintf('W2 before:\n'); disp(W2_orig);
fprintf('W2 after:\n');  disp(W2);
fprintf('b2 before:\n'); disp(b2_orig);
fprintf('b2 after:\n');  disp(b2);

fprintf('W3 before:\n'); disp(W3_orig);
fprintf('W3 after:\n');  disp(W3);
fprintf('b3 before:\n'); disp(b3_orig);
fprintf('b3 after:\n');  disp(b3);


z1_up = W1 * x + b1; a1_up = sig(z1_up);
z2_up = W2 * a1_up + b2; a2_up = sig(z2_up);
z3_up = W3 * a2_up + b3; a3_up = sig(z3_up);

yhat_after = a3_up;
L_after = 0.5 * (y - yhat_after)^2;

fprintf('\n--- Forward pass ---\n');
fprintf('yhat before = %g   loss before = %g\n', yhat_before, L_before);
fprintf('yhat after  = %g   loss after  = %g\n\n', yhat_after, L_after);


phi = @(v) double(v >= 0);


a1_bin = phi(W1_orig * x + b1_orig);        
a2_bin = phi(W2_orig * a1_bin + b2_orig);   
z3_bin = W3_orig * a2_bin + b3_orig;        
a3_bin = phi(z3_bin);                       

fprintf('--- Binary (hard-threshold) forward with ORIGINAL weights ---\n');
fprintf('a1 (binary) =\n'); disp(a1_bin);
fprintf('a2 (binary) =\n'); disp(a2_bin);
fprintf('z3 (pre-output) = %g\n', z3_bin);
fprintf('a3 (binary output) = %g\n\n', a3_bin);


x00 = [0;0];
a1_bin_00 = phi(W1_orig * x00 + b1_orig);    
a2_bin_00 = phi(W2_orig * a1_bin_00 + b2_orig);
z3_bin_00 = W3_orig * a2_bin_00 + b3_orig;
a3_bin_00 = phi(z3_bin_00);

fprintf('--- Binary output for input [0;0] using ORIGINAL weights ---\n');
fprintf('b1 (original biases) =\n'); disp(b1_orig);
fprintf('a1_binary([0;0]) = double(b1 >= 0) =\n'); disp(a1_bin_00);
fprintf('a2_binary([0;0]) =\n'); disp(a2_bin_00);
fprintf('z3 for [0;0] = %g    a3_binary([0;0]) = %g\n\n', z3_bin_00, a3_bin_00);


