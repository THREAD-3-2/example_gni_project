% An example showing how to use the `Forward_Euler` function.
clear all;
clc;

% Define parameters.
vector_field = @(x) [x(2); -x(1)];
initial_condition = [1 0];
h = 0.1;
duration = 0.3;

% Call the `Forward_Euler` function.
[t, x] = Forward_Euler(vector_field, initial_condition, h, duration);

% Plot the first component of the discrete trajectory.
plot(t, x(1, :));
