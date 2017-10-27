def rollout(env, policy):
    ob = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        stochastic = True
        #~ print(env.observation_space.observationAsString(ob))        
        ac, vpred = policy.act(stochastic, ob)
        #~ print(env.action_space.actionAsString(ac))    
        ob, rew, done, _ = env.step(ac)
        total_reward += rew
    print("Total reward = {}.".format(total_reward))
