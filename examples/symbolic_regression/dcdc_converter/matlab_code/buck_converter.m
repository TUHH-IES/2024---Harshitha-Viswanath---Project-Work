% Buck Converter Simulation in MATLAB (ODE-based, no Simulink required)
% This script models a buck converter as a hybrid system using two modes.

clear; clc;

%% Parameters
Vin = 24;           % Input voltage [V]
L = 1e-3;           % Inductance [H]
C = 470e-6;         % Capacitance [F]
R = 10;             % Load resistance [Ohm]
Vref = 12;          % Reference voltage [V]
hyst = 0.02;        % Smaller hysteresis for better switching [V]
Tsim = 0.1;         % Longer simulation time [s]

% Initial conditions
x0 = [0; 0];        % [iL; Vout]

%% Time settings
dt = 1e-6;                      % Time step
t = 0:dt:Tsim;                  % Time vector
x = zeros(2, length(t));        % State vector: [iL; Vout]
x(:,1) = x0;
mode = zeros(1, length(t));     % 1 = switch ON, 0 = switch OFF
mode(1) = 1;                    % Start with switch ON

%% Simulation loop (hybrid dynamics)
for k = 1:length(t)-1
    iL = x(1,k);
    Vout = x(2,k);

    % Hysteresis control logic
    if mode(k) == 1 && Vout >= Vref + hyst
        mode(k+1) = 0;  % Turn OFF
    elseif mode(k) == 0 && Vout <= Vref - hyst
        mode(k+1) = 1;  % Turn ON
    else
        mode(k+1) = mode(k);  % Hold previous state
    end

    % Continuous dynamics
    if mode(k+1) == 1
        % Switch ON
        diL = (Vin - Vout)/L;
    else
        % Switch OFF (diode conducts)
        diL = -Vout/L;
    end
    dVout = (iL - Vout/R)/C;

    % Euler integration
    x(1,k+1) = iL + dt*diL;
    x(2,k+1) = Vout + dt*dVout;
end

%% Plot results
figure;
subplot(3,1,1);
plot(t, x(1,:), 'b');
ylabel('Inductor Current i_L [A]');
title('Buck Converter Simulation');
grid on;

subplot(3,1,2);
plot(t, x(2,:));
hold on; 
yline(Vref, '--k'); 
yline(Vref + hyst, '--r'); 
yline(Vref - hyst, '--g');
ylabel('Capacitor Voltage V_C [V]');
legend('V_C','V_{ref}', 'V_{ref} + hyst', 'V_{ref} - hyst');
grid on;

subplot(3,1,3);
stairs(t, mode, 'k');
xlabel('Time [s]');
ylabel('Switch State');
yticks([0 1]); yticklabels({'OFF', 'ON'});
title('Switching Behavior');
grid on;

%% Export to CSV
T = table(t', x(1,:)', x(2,:)', mode', ...
    'VariableNames', {'Time_s', 'InductorCurrent_A', 'CapacitorVoltage_V', 'SwitchState'});

writetable(T, 'buck_converter_output.csv');

%% Show which switch states were seen
disp('Switch states seen:');
disp(unique(mode));
