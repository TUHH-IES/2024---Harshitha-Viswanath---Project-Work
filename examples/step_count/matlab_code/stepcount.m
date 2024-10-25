% Define states for the hybrid system
IDLE = 1;
DETECTION = 2;
COUNTING = 3;

% Initialize parameters
currentState = IDLE;  % Initial state
stepCount = 0;        % Step count
thresholdDetect = 1.0;   % Threshold to detect movement (arbitrary units)
thresholdCount = 1.2;    % Threshold to transition to step counting
timeStep = 0.01;         % Time step for integration (seconds)
totalTime = 10;          % Total simulation time (seconds)
time = 0:timeStep:totalTime;  % Time vector

% Simulated accelerometer data 
acceleration = sin(2 * pi * 1 * time) + 0.5 * randn(size(time));  % Noisy sine wave acceleration

% Array to store step events
stepEvents = zeros(size(time));

% Parameters for differential equation (foot vertical displacement)
omega = 2 * pi;        % Frequency of oscillation (natural frequency of foot movement)
y = 20;                 % Initial vertical displacement
v = 10;                 % Initial velocity
y_threshold = 17.5;     % Threshold for step detection (based on vertical displacement)

for i = 1:length(time)
    accelMag = abs(acceleration(i));  % Use magnitude of acceleration
    
    switch currentState
        case IDLE
            % Detect movement to switch to detection mode
            if accelMag > thresholdDetect
                currentState = DETECTION;
            end
        
        case DETECTION
            % Detect if acceleration surpasses the walking threshold
            if accelMag > thresholdCount
                currentState = COUNTING;
            elseif accelMag < thresholdDetect
                currentState = IDLE;  % Return to IDLE if no further activity
            end
        
        case COUNTING
            % Use a differential equation to model the foot's vertical displacement
            % Simple harmonic motion: d^2y/dt^2 + omega^2 * y = 0
            % Euler's method to update y (displacement) and v (velocity)
            v = v - omega^2 * y * timeStep;  % Update velocity
            y = y + v * timeStep;            % Update displacement
            
            % Count step if the displacement exceeds the threshold
            if abs(y) > y_threshold
                stepCount = stepCount + 1;   % Increment step count
                stepEvents(i) = 1;           % Mark step event
                y = 10;  % Reset displacement after step is counted
            end
            
            % Transition back to DETECTION mode if acceleration drops
            if accelMag < thresholdDetect
                currentState = DETECTION;
            end
    end
end

% Plot the results
figure;
subplot(3, 1, 1);
plot(time, acceleration);
title('Simulated Acceleration Data');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

subplot(3, 1, 2);
plot(time, stepEvents);
title('Step Events');
xlabel('Time (s)');
ylabel('Step Detected');

subplot(3, 1, 3);
plot(time, y);
title('Foot Displacement (y)');
xlabel('Time (s)');
ylabel('Displacement (m)');

disp(['Total steps counted: ', num2str(stepCount)]);
