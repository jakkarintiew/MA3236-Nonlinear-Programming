function [x,p,iter] = conjugateGrad(fun)
% Load and store matrix in A, b (Ab.mat must be in the same folder)
A = load('Ab.mat', 'A');
b = load('Ab.mat', 'b');
A = cell2mat(struct2cell(A));
b = cell2mat(struct2cell(b));

% Initalize x0, r0, p0, and iteration limit
x0 = zeros(size(A, 1),1);
[fx,r0] = feval(fun, x0, A, b);
p0 = -r0;
iter = 0;
maxit = 10;

% Print resutls table header
fprintf('\n iter   ||x - x*||');
fprintf('\n ------------------ \n');

% Compute optimal solution x_min
x_min = A^(-1) * b;

% Initalize interates
x = x0;
r = r0;
p = p0;

while (iter <= maxit & r ~= 0)
    % Compute and print error ||x - x*||
    err = norm(x - x_min);
    fprintf('%3.0f    %9.3f \n', iter, err);
    
    % Compute step-size (alpha), and update x
    alpha = -(r'*p)/(p'*A*p);
    x = x + alpha*p;
    
    % Update r, p, and compute beta
    [fx,r] = feval(fun, x, A, b);
    beta = (r'*A*p)/(p'*A*p);
    p = -r + beta*p;
    
    iter = iter + 1;
end
end
%%***********************************************************

function [fx,grad] = quadFn(x, A, b)
fx = 0.5*x'*A*x-b'*x;
grad = A*x-b;
end
