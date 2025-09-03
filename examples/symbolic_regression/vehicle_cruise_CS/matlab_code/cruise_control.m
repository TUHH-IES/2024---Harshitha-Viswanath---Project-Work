% Two-Mode Vehicle Control: Accelerate and Brake

% Parameters
dt = 1;                      % Time step (seconds)
T = 1000;                    % Total simulation time (seconds)
time = 0:dt:T;               % Time vector
velocity = zeros(size(time));% Velocity vector (km/h)

% Initial condition
velocity(1) = 35;    % Start below the threshold
dx_target = 0;
% Mode and step storage
modes = strings(size(time));
dxs = zeros(size(time));     % Store dx at each step
v_curr = zeros(size(time));  % Store velocity(t)
v_next = zeros(size(time));  % Store velocity(t+1)

for t = 1:length(time)-1
    v_curr(t) = velocity(t);
    if velocity(t) < 50 - hyst 
        dx = +5; 
        modes(t) = "Accelerate"; 
    elseif velocity(t) >= 50 + hyst 
        dx = -7; 
        modes(t) = "Brake"; 
    end 

    alpha = 0.2; %easies the triangular spikes
    dx_target = dx_target + alpha * (dx - dx_target);
    dxs(t) = dx_target; 
    velocity(t+1) = velocity(t) + dx_target * dt; %Pysr models this equation: all the datapoints are grouped into a single group
    v_next(t) = velocity(t+1);
end
% For the last entry
v_curr(end) = velocity(end);
v_next(end) = velocity(end);
dxs(end) = dxs(end-1);
modes(end) = modes(end-1);

% --- Plot velocity vs time with mode coloring ---
figure;

subplot(2,1,1);
hold on;
for i = 1:length(time)-1
    if modes(i) == "Accelerate"
        plot(time(i:i+1), velocity(i:i+1), 'b', 'LineWidth', 2);
    else
        plot(time(i:i+1), velocity(i:i+1), 'r', 'LineWidth', 2);
    end
end
xlabel('Time (s)');
ylabel('Velocity (km/h)');
title('Vehicle Velocity vs. Time (Accelerate: Blue, Brake: Red)');
grid on;
legend({'Accelerate','Brake'});
hold off;

% --- Plot modes vs time as categorical scatter ---
subplot(2,1,2);
mode_numeric = double(modes == "Brake") + 1; % 1 for Accelerate, 2 for Brake
scatter(time, mode_numeric, 10, mode_numeric, 'filled');
yticks([1 2]);
yticklabels({'Accelerate','Brake'});
xlabel('Time (s)');
ylabel('Mode');
title('Control Mode vs. Time');
grid on;

% --- Save results to CSV ---
t = time(:);                   % Column vector
v_curr = v_curr(:);
v_next = v_next(:);
dxs = dxs(:);
dts = dt * ones(size(t));      % Constant dt
modes = modes(:);

resultsTable = table(t, v_curr, v_next, dxs, dts, modes, ...
    'VariableNames', {'Time_s', 'Velocity_t', 'Velocity_tplus1', 'dx', 'dt', 'Mode'});

writetable(resultsTable, 'C:\Users\49157\Desktop\PA\2024---Harshitha-Viswanath---Project-Work\examples\symbolic_regression\vehicle_cruise_CS\matlab_code\vehicle_modes_output.csv');

disp('Simulation results with velocities, dx, and dt saved to vehicle_modes_output.csv');
