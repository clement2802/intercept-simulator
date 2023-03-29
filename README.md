# intercept-simulator
Hornets must intercept bees, using reinforcement learning

![image](https://user-images.githubusercontent.com/94045630/228486137-07e16242-09ae-4d4c-956b-a23af6d6bb97.png)

In this simulator, the hornet can receive 3 actions : up, down, none. On the window you can see the action action at each step : up and down arrows and round for none.

The observations are :
- miss distance (relative distance on y axis), 
- vertical speed
- relative distance on x axis, 
- fuel level (very big but you can reduce it)
- relative speed between hornet and bee (constant because the hornet and the bee has the same speed).

You can train the code with the ray_run function in affecta_hpt.py, and custom the config in the function.
