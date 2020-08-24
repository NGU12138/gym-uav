from gym.envs.registration import register

register(
    id='uav-v0',
    entry_point='gym_uav.envs:UavEnv',
    max_episode_steps=2000
)

# register(
#     id='uav-v1',
#     entry_point='gym_uav.envs:UavDenseEnv',
#     max_episode_steps=2000
# )

# register(
#     id='uav-v2',
#     entry_point='gym_uav.envs:UavGoalEnv',
#     max_episode_steps=2000
# )
