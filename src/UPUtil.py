def rollout(env, policy):
    ob = env.reset()
    done = False
    while not done:
        stochastic = True
        ac, vpred = policy.act(stochastic, ob)
        ob, rew, done, _ = env.step(ac)
