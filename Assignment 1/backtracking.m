function [x, iter] = backtracking(impMethod,x0,rho,c,printyes)

if (nargin <= 4); printyes = 0; end

if (printyes)
    fprintf('\n iter    x1    x2    f(x)     step-len');
    fprintf('\n---------------------------------------\n');
end

% Initialize step size = 1
alpha_init = 1;
x = x0;
iter = 0;
maxIter = 10;

% Plot contour map and inital point
syms x_1 x_2
hfc = fcontour(100*(x_2-(x_1)^2)^2 + (1-x_1)^2, [-1.5 1.5 -0.5 1.5], 'MeshDensity', 2000);
contour(hfc.XData, hfc.YData, hfc.ZData, 'ShowText','on');
hold on;
contour(hfc.XData, hfc.YData, hfc.ZData, [5,5], 'ShowText','on');
contour(hfc.XData, hfc.YData, hfc.ZData, [1,1], 'ShowText','on');
plot(x0(1), x0(2),'bx')
text(x0(1), x0(2),strcat('[',num2str(x0(1)), ';  ', num2str(x0(2)), ']'))
colorbar
xlabel('x_1'); ylabel('x_2');
title(strcat('Path of descent on contour plot for Method:  ', impMethod, '  at initial point  ', strcat('[',num2str(x0(1)), ';  ', num2str(x0(2)), ']')));


while(iter <= maxIter)
    % Reset alpha to initial value at every iteration
    alpha = alpha_init;
    
    % Determine decent direction depends on method of implementation
    if strcmp(impMethod, 'newton')
        d = - rosenbrockHessian(x)^-1 * rosenbrockGradient(x);
    elseif strcmp(impMethod,'steepest descent')
        d = - rosenbrockGradient(x);
    end
    
    % Determine step size
    while (rosenbrockFunction(x) + alpha * c * rosenbrockGradient(x).' * d < rosenbrockFunction(x + alpha * d))
        alpha = rho * alpha;
    end
    
    if (printyes)
        fprintf('%2.0f %9.3f %6.3f %6.3f %9.3f\n',iter,x(1),x(2),rosenbrockFunction(x),alpha);
    end
    
    % Store previous x value
    x_prev = x;
    % Update x
    x = x + alpha * d;  
    
    % Plot x and join line with previous x 
    plot(x(1), x(2),'rx');
    % text(x(1), x(2),strcat('[',num2str(x(1)), ';  ', num2str(x(2)), ']'))
    plot( [x(1), x_prev(1)], [x(2), x_prev(2)], 'r-', 'LineWidth', 2);
    
    iter = iter + 1;
    
end

hold off
fprintf('---------------------------------------\n');

end

function f = rosenbrockFunction(x)
f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
end

function df = rosenbrockGradient(x)
df = [2 * x(1) - 400 * x(1) * (- x(1)^2 + x(2)) - 2; 200 * (x(2) - x(1)^2)];
end

function d2f = rosenbrockHessian(x)
d2f = [2 + 1200 * x(1)^2 - 400 * x(2), -400*x(1); -400 * x(1), 200];
end