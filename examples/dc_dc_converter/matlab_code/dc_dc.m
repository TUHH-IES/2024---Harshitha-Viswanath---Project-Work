% Define the parameters
L = 1e-3; % Inductance in Henry
C = 1e-6; % Capacitance in Farad
RL = 0.1; % Inductor resistance in Ohm
R = 10; % Load resistance in Ohm
RC = 0.01; % Capacitor resistance in Ohm
E = 10; % Input voltage in Volts


params.A1 = [-RL/L, 0; 0, -1/(R+RC)*C];
params.B1 = [1/L; 0] * E;

params.A2 = [-1/L * (RL + RC*R/(R+RC)), 1/L * (-1 + R/(R+RC)); 
             R/(R+RC)*C, -1/(R+RC)*C];
params.B2 = [1/L; 0] * E;

params.A3 = [0, 0; 0, -1/(R+RC)*C];
params.B3 = [0; 0]; % Should be zero, as there's no input in state 3

% Define the initial conditions and time span
tspan = 0:1e-6:1e-3; % Simulation time span with small steps for better resolution
x0 = [0.2; 0.1]; % Initial conditions [i_L; v_C]

% Define control parameters
n = 10;        % Example value for n
u1 = 0.5;      % Duty cycle (example value)
Ts = 1e-3;     % Switching period in seconds

% Initialize arrays to store results
T = []; % To store time points
X = []; % To store state variables

% Simulate the system using a loop to handle state transitions
current_state = 1;
for k = 1:length(tspan)-1
    % Call ode45 with the additional parameter
    [t, x] = ode45(@(t, x) boostConverterDynamics(t, x, current_state, params), ...
               [tspan(k) tspan(k+1)], x0);
    
    % Store results
    T = [T; t]; % Concatenate time points
    X = [X; x]; % Concatenate state results
    
    % Update the state based on switching logic
    current_state = switchLogic(t(end), x(end, :), current_state, n, u1, Ts);
    x0 = x(end, :); % Update initial conditions for the next time step
end

% Plotting the results
figure;
subplot(2,1,1);
plot(T * 1e3, X(:, 1)); % Plot inductor current i_L vs. time in ms
xlabel('Time [ms]');
ylabel('i_L [A]');
title('Inductor Current');

subplot(2,1,2);
plot(T * 1e3, X(:, 2)); % Plot capacitor voltage v_C vs. time in ms
xlabel('Time [ms]');
ylabel('v_C [V]');
title('Capacitor Voltage');

figure;
plot(X(:, 1), X(:, 2)); % Plot limit cycle of i_L vs. v_C
xlabel('i_L [A]');
ylabel('v_C [V]');
title('Limit Cycle of Boost Converter');

%% Modified switchLogic function
function state = switchLogic(t, x, current_state, n, u1, Ts)
    % Define transition conditions based on current state and conditions
    i_L = x(1); % Inductor current from the state vector

    % Example transition conditions (adjust based on specific application logic)
    switch current_state
        case 1 % Transition from state 1 to state 2 based on time
            if t >= (n-1+u1)*Ts
                state = 2;
            else
                state = current_state;
            end
        case 2 % Transition from state 2 to state 3 based on time
            if t >= n*Ts
                state = 3;
            else
                state = current_state;
            end
        case 3 % Transition from state 3 back to state 1 if inductor current is low
            if i_L <= 0
                state = 1;
            else
                state = current_state;
            end
        otherwise
            error('Invalid state');
    end
end

%% Dynamics function
function dxdt = boostConverterDynamics(t, x, current_state, params)
    % Extract parameters from the input struct
    A1 = params.A1;
    B1 = params.B1;
    A2 = params.A2;
    B2 = params.B2;
    A3 = params.A3;
    B3 = params.B3;
    
    % Define system dynamics based on the current state
    switch current_state
        case 1
            A = A1;
            B = B1;
        case 2
            A = A2;
            B = B2;
        case 3
            A = A3;
            B = B3;
        otherwise
            error('Invalid state');
    end
    
    % Calculate the derivative
    dxdt = A * x + B;
end
