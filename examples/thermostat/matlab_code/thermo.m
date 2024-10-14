% Define the states
IDLE = 1;
DETECTION = 2;
WALKING = 3;

% Initialize parameters
currentState = IDLE;
stepCount = 0;
thresholdDetect = 1.0;   % Threshold to detect movement (arbitrary units)
thresholdWalk = 1.2;     % Threshold to transition to walking
timeStep = 0.01;         % Time step for integration (seconds)
totalTime = 10;          % Total simulation time (seconds)
time = 0:timeStep:totalTime;

% Simulated accelerometer data (you can replace this with actual sensor data)
acceleration = sin(2 * pi * 1 * time) + 0.5 * randn(size(time));

% Store step events
stepEvents = zeros(size(time));

% Parameters for differential equation (simple harmonic oscillator model)
omega = 2 * pi;  % Angular frequency of foot oscillation
y = 0;           % Initial displacement
v = 0;           % Initial velocity
y_threshold = 0.5;  % Threshold for step detection (based on vertical displacement)

% Main loop for state transitions and step counting
for i = 1:length(time)
    accelMag = abs(acceleration(i));  % Use magnitude of acceleration
    
    switch currentState
        case IDLE
            % Transition to Detection if acceleration exceeds the detection threshold
            if accelMag > thresholdDetect
                currentState = DETECTION;
            end
        
        case DETECTION
            % Transition to Walking if acceleration exceeds the walking threshold
            if accelMag > thresholdWalk
                currentState = WALKING;
            elseif accelMag < thresholdDetect
                currentState = IDLE;  % Return to IDLE if no further activity
            end
        
        case WALKING
            % Use a differential equation to model the foot's vertical displacement
            % Simple harmonic motion: d^2y/dt^2 + omega^2 * y = 0
            % Euler's method to update y and v:
            v = v - omega^2 * y * timeStep;  % Update velocity
            y = y + v * timeStep;            % Update displacement
            
            % Count step if the displacement exceeds the threshold
            if abs(y) > y_threshold
                stepCount = stepCount + 1;  % Increment step count
                stepEvents(i) = 1;          % Mark step event
                y = 0;  % Reset displacement after step is counted
            end
            
            % Return to Detection if acceleration drops below threshold
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
