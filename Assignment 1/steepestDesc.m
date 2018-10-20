function [x,p,iter] = steepestDesc(fun)
% Load and store matrix in A, b (Ab.mat must be in the same folder)
A = load('Ab.mat', 'A'); 
b = load('Ab.mat', 'b'); 
A = cell2mat(struct2cell(A));
b = cell2mat(struct2cell(b));

% Initalize x0 and iteration limit
x0 = zeros(size(A, 1),1);
maxit = 10;

% Print resutls table header
fprintf('\n iter   ||x - x*||');
fprintf('\n ------------------ \n');

% Compute optimal solution x_min
x_min = A^(-1) * b;

% Initalize interates
x = x0;

for iter = 0:maxit
    % Compute and print error ||x - x*||
    err = norm(x - x_min);
    fprintf('%3.0f    %9.3f \n', iter, err);
    
    % Compute gradient(grad), descent direction(p), and step size(t)
    [fx,grad] = feval(fun, x, A, b);
    p = -grad;
    t = steplength(grad, A);
    
    % Update x
    x = x + t*p; 
end
end
%%***********************************************************

function t = steplength(grad, A)
t = (grad'*grad)/(grad'*A*grad);
end

function [fx,grad] = quadFn(x, A, b)
fx = 0.5*x'*A*x-b'*x;
grad = A*x-b;
end
