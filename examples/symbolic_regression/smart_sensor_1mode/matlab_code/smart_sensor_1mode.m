% Simulation setup
T = 10;
dt = 0.01;
t = 0:dt:T;
N = length(t);

% Input signal
u = 2.5 * sin(2 * pi * 0.2 * t) + 2;  % Smooth varying signal >= 0

% Use only one mode (e.g., compression: y = sqrt(u + 1))
y = sqrt(u + 1);  % Vectorized computation

% Plot results
figure;
subplot(2,1,1);
plot(t, u, 'b'); grid on;
ylabel('Input u(t)');
title('Single-mode System Simulation');

subplot(2,1,2);
plot(t, y, 'r'); grid on;
xlabel('Time (s)');
ylabel('Output y(t)');

% Save to CSV (no mode column now)
T_data = table(t', u', y', ...
    'VariableNames', {'time', 'input_u', 'output_y'});
writetable(T_data, 'single_mode_sensor_data.csv');
