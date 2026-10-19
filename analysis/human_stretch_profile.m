function result = human_stretch_profile(t,onset,peak_lambda,ramp_ms,hold_ms)
% Finite-ramp length pulse: ramp up, hold, ramp down, return to lambda=1.
if t < onset
    result = [1,0];
elseif t < onset+ramp_ms
    result = [1+(peak_lambda-1)*(t-onset)/ramp_ms, ...
        (peak_lambda-1)/ramp_ms];
elseif t < onset+ramp_ms+hold_ms
    result = [peak_lambda,0];
elseif t < onset+2*ramp_ms+hold_ms
    result = [peak_lambda-(peak_lambda-1)* ...
        (t-onset-ramp_ms-hold_ms)/ramp_ms, ...
        -(peak_lambda-1)/ramp_ms];
else
    result = [1,0];
end
end
