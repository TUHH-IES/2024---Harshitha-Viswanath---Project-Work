T = 10;
dt = 0.01;
t = 0:dt:T;
N = length(t);

u = 2.5 * sin(2 * pi * 0.2 * t) + 2;

q = 1;                     % Initial mode
y = zeros(1, N);           % Output
q_history = zeros(1, N);   % Mode history

for k = 1:N
    u_k = u(k);

    % Save current mode before switching (so output matches mode at time k)
    q_history(k) = q;

    % Switching logic
    if q == 1 && u_k >= 2
        q = 2;
    elseif q == 2 && u_k <= 1
        q = 1;
    end

    % Output logic
    if q == 1
        y(k) = sqrt(u_k + 1);
    elseif q == 2
        y(k) = 5 * u_k^2 + 3;
    end
end

% Plot
figure;
subplot(3,1,1);
plot(t, u, 'b'); grid on;
ylabel('Input u(t)');
title('Hybrid system Simulation');

subplot(3,1,2);
plot(t, y, 'r'); grid on;
ylabel('Output y(t)');

subplot(3,1,3);
stairs(t, q_history, 'k'); grid on;
xlabel('Time (s)');
ylabel('Mode q');
yticks([1 2]);

% Save to CSV
T_data = table(t', u', y', q_history', ...
    'VariableNames', {'time', 'input_u', 'output_y', 'mode_q'});
writetable(T_data, 'hybrid_sensor_data.csv');
