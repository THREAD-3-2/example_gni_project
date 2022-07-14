function [t,x] = Forward_Euler(x0,dt,T)
% Forward Euler time integrator
%
% :param x0: initial value, dt: time step, T: total time
% :returns: t, x

N = T/dt;
t=zeros(N+1);
x=zeros(N+1);

t(1) = 0;
x(1) = x0;

for i=2:N+1
    % update x
    x(i) = x(i-1) + dt*fun(t(i-1),x(i-1));
    % update t
    t(i) = t(i-1) + dt;
end
end