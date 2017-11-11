def rollout(env, policy):
    ob = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        stochastic = True        
        ac, vpred = policy.act(stochastic, ob)
        ob, rew, done, _ = env.step(ac)        
        total_reward += rew
    print("Total reward = {}.".format(total_reward))
