function [t, x] = Forward_Euler(vector_field, initial_condition, h, duration)
% Forward Euler time integrator
%
% :param vector_field: right hand side of the ODE
% :param initial_condition: initial condition
% :param h: time step size
% :param duration: duration of simulation
%
% :returns: [t, x] : time grid, discrete trajectory

N = round(duration/h, 0);
t = zeros(N+1,1);
x = zeros(size(initial_condition, 2), N+1);

t(1) = 0;
x(:, 1) = initial_condition;

for i = 2:N+1
    x(:, i) = x(:, i-1) + h*vector_field(x(:, i-1));
    t(i) = t(i-1) + h;
end
end