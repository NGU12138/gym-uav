import gym
from gym import error, spaces, utils
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import time
import vtk
import threading
import itertools
import copy

from utils import TimerCallback
from utils import Config
from utils import Smoother_soft


class UavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        Common = Config()
        self.basic_directions = Common.basic_directions
        self.extra_directions = Common.extra_directions
        self.original_observation_length = Common.original_observation_length
        self.extra_length = len(self.extra_directions)

        self.observation_space = Box(-np.inf, np.inf, [self.original_observation_length + self.extra_length], float)
        self.action_space = Box(-1.0, 1.0, [2], float)
        self._env_step_counter = 0

        self.state = np.zeros([self.observation_space.shape[0]])

        #
        self.level = Common.level
        self.position = np.zeros([2])
        self.target = np.zeros([2])
        self.orient = np.zeros([1])
        self.speed = np.zeros([1])
        self.max_speed = Common.max_speed
        self.min_distance_to_target = Common.min_distance_to_target
        self.real_action_range = Common.real_action_range

        #
        self.min_distance_to_obstacle = Common.min_distance_to_obstacle
        self.min_initial_starts = Common.min_initial_starts
        self.expand = Common.expand
        self.num_circle = Common.num_circle
        self.radius = Common.radius
        self.period = Common.period
        self.mat_height = None  #
        self.mat_exist = None  #
        self.lowest = Common.lowest
        self.delta = Common.delta
        self.total = Common.total

        #
        self.scope = Common.scope
        self.min_step = Common.min_step
        self.directions = self.basic_directions + self.extra_directions
        self.end_points = [None for _ in range(len(self.directions))]

        #
        self.margin = Common.margin
        self.env_params = {'cylinders': None, 'size': 1.5*(self.num_circle+self.margin*2)*self.period}
        self.agent_params = {'position': self.position, 'target': self.target, 'direction':None, 'rangefinders': self.end_points}
        self.agent_params_pre = None
        self.first_render = True
        self.terminate_render = False
        self.camera_alpha = Common.camera_alpha

        #
        self.is_reset = False
        assert self.scope > self.max_speed

    def _fast_range_finder(self, position, theta, forward_dist, min_dist=0.0, find_type='normal'):
        end_cache = copy.deepcopy(position)
        position_integer = np.floor(end_cache / self.period).astype(np.int)
        judge = end_cache - (position_integer * self.period + self.period / 2)
        if judge[0] >= 0 and judge[1] > 0:
            down_left = position_integer * self.period + self.period / 2
            down_right = (position_integer + np.array([1, 0])) * self.period + self.period / 2
            up_left = (position_integer + np.array([0, 1])) * self.period + self.period / 2
            up_right = (position_integer + np.array([1, 1])) * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[position_integer[0] + 1 + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[position_integer[0] + self.expand, position_integer[1] + 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] + 1 + self.expand, position_integer[1] + 1 + self.expand]]))
        elif judge[0] >= 0 and judge[1] < 0:
            down_left = (position_integer + np.array([0, -1])) * self.period + self.period / 2
            down_right = (position_integer + np.array([1, -1])) * self.period + self.period / 2
            up_left = position_integer * self.period + self.period / 2
            up_right = (position_integer + np.array([1, 0])) * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[position_integer[0] + 1 + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[position_integer[0] + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[
                                   position_integer[0] + 1 + self.expand, position_integer[1] + self.expand]]))
        elif judge[0] < 0 and judge[1] > 0:
            down_left = (position_integer + np.array([-1, 0])) * self.period + self.period / 2
            down_right = position_integer * self.period + self.period / 2
            up_left = (position_integer + np.array([-1, 1])) * self.period + self.period / 2
            up_right = (position_integer + np.array([0, 1])) * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] -1 + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[position_integer[0] - 1 + self.expand, position_integer[1] + 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] + 1 + self.expand]]))
        else:
            down_left = (position_integer + np.array([-1, -1])) * self.period + self.period / 2
            down_right = (position_integer + np.array([0, -1])) * self.period + self.period / 2
            up_left = (position_integer + np.array([-1, 0])) * self.period + self.period / 2
            up_right = position_integer * self.period + self.period / 2
            exists = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'],
                              [self.mat_exist[position_integer[0] - 1 + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] - 1 + self.expand],
                               self.mat_exist[
                                   position_integer[0] - 1 + self.expand, position_integer[1] + self.expand],
                               self.mat_exist[
                                   position_integer[0] + self.expand, position_integer[1] + self.expand]]))

        base_points = dict(zip(['down_left', 'down_right', 'up_left', 'up_right'], [down_left, down_right, up_left, up_right]))

        dist = []
        end = []
        for base in base_points.keys():
            theta_base = np.arctan(np.abs((base_points[base] - end_cache)[0] / (base_points[base] - end_cache)[1]))
            if base == 'down_left':
                theta_base = np.pi + theta_base
            if base == 'down_right':
                theta_base = np.pi - theta_base
            if base == 'up_left':
                theta_base = 2 * np.pi - theta_base
            if base == 'up_right':
                theta_base = theta_base

            theta_base = np.mod(theta_base, 2*np.pi)

            dist_to_base = np.linalg.norm(end_cache - base_points[base])

            delta_theta = theta - theta_base
            if dist_to_base - (self.radius + min_dist) >= forward_dist or exists[base] < 0:
                dist.append(1.0)
                end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
            elif (dist_to_base - (self.radius + min_dist) >= 0) and (np.cos(delta_theta) <= 0):
                dist.append(1.0)
                end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
            else:
                min_dist_to_origin = np.abs(np.sin(delta_theta)) * dist_to_base

                if min_dist_to_origin >= (self.radius + min_dist):
                    dist.append(1.0)
                    end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
                else:
                    dist_inner = np.sqrt((self.radius + min_dist) ** 2 - min_dist_to_origin ** 2)
                    final_dist = np.cos(delta_theta) * dist_to_base - dist_inner
                    final_dist = final_dist[0]
                    if final_dist >= forward_dist:
                        dist.append(1.0)
                        end.append(end_cache + np.array([forward_dist * np.sin(theta[0]), forward_dist * np.cos(theta[0])]))
                    else:
                        dist.append(final_dist / forward_dist)
                        end.append(end_cache + np.array([final_dist * np.sin(theta[0]), final_dist * np.cos(theta[0])]))
        dist = np.array(dist)

        return np.min(dist), end[np.argmin(dist)]

    def _prepare_background_for_render(self):
        small_mat_height = self.mat_height[self.expand - self.margin: self.expand + self.num_circle + self.margin,
                           self.expand - self.margin: self.expand + self.num_circle + self.margin]
        small_mat_exist = self.mat_exist[self.expand - self.margin: self.expand + self.num_circle + self.margin,
                          self.expand - self.margin: self.expand + self.num_circle + self.margin]
        index_tmp = [i - self.margin for i in range(np.shape(small_mat_height)[0])]
        position_tmp = list(itertools.product(index_tmp, index_tmp))
        position_tmp = [list(pos) for pos in position_tmp]
        position_tmp = np.array(position_tmp)
        position_tmp = position_tmp * self.period + self.period / 2

        cylinders = []
        small_mat_height = list(small_mat_height.reshape(1,-1)[0])
        small_mat_exist = list(small_mat_exist.reshape(1,-1)[0])
        position_tmp = list(position_tmp)

        for hei, exi, pos in zip(small_mat_height, small_mat_exist, position_tmp):
            if exi > 0:
                p1 = np.concatenate([pos, np.array([0])])
                p2 = np.concatenate([pos, np.array([hei])])
                r = self.radius
                cylinders.append([p1, p2, r])

        self.env_params['cylinders'] = copy.deepcopy(cylinders)

    def _get_observation(self, position, target, orient):
        global_counter = 0
        basic_counter = 0
        extra_counter = 0

        for dir in self.basic_directions:
            theta = np.mod(dir + orient, 2 * np.pi)  #
            self.state[basic_counter], end_cache = self._fast_range_finder(position, theta, self.scope)
            self.end_points[global_counter] = [np.concatenate([position, np.array([self.level])]),
                                                   np.concatenate([end_cache, np.array([self.level])])]

            global_counter += 1
            basic_counter += 1

        # adding extra range finders
        for dir in self.extra_directions:
            theta = np.mod(dir + orient, 2 * np.pi)  #
            self.state[15 + extra_counter], end_cache = self._fast_range_finder(position, theta, self.scope)
            self.end_points[global_counter] = [np.concatenate([position, np.array([self.level])]),
                                                   np.concatenate([end_cache, np.array([self.level])])]
            global_counter += 1
            extra_counter += 1

        dist = np.linalg.norm(target - position)
        self.state[9] = 2*(dist / (np.sqrt(2)*self.period * self.num_circle) - 0.5)
        #
        theta_target = np.arctan((target[0] - position[0]) / (target[1] - position[1]))
        if (target[0] >= position[0]) and (target[1] >= position[1]):
            self.state[10] = np.sin(theta_target)
            self.state[11] = np.cos(theta_target)
        elif target[1] < position[1]:
            self.state[10] = np.sin(theta_target + np.pi)
            self.state[11] = np.cos(theta_target + np.pi)
        else:
            self.state[10] = np.sin(theta_target + 2 * np.pi)
            self.state[11] = np.cos(theta_target + 2 * np.pi)

        #
        self.state[12] = np.sin(orient)  # normalization
        self.state[13] = np.cos(orient)  # normalization
        #
        self.state[14] = 2*(self.speed / self.max_speed - 0.5)

        self.agent_params_pre = copy.deepcopy(self.agent_params)
        self.agent_params['position'] = copy.deepcopy(np.concatenate([position, np.array([self.level])]))
        self.agent_params['target'] = copy.deepcopy(np.concatenate([target, np.array([self.level])]))
        self.agent_params['rangefinders'] = copy.deepcopy(self.end_points)
        self.agent_params['direction'] = copy.deepcopy(np.mod(90 - orient/2/np.pi*360, 360))
        self.agent_params['direction_camera'] = copy.deepcopy(np.mod(90 - np.mod(self.orient_render, 2*np.pi)/2/np.pi*360, 360))

    def step(self, action):
        assert self.is_reset, 'the environment must be reset before it is called'
        self._env_step_counter += 1
        self.orient = np.mod(self.real_action_range[0] * action[0] * np.pi + self.orient, 2 * np.pi)
        self.orient_total_pre = copy.deepcopy(self.orient_total)
        self.orient_render_pre = copy.deepcopy(self.orient_render)
        self.orient_total = self.real_action_range[0] * action[0] * np.pi + self.orient_total
        self.orient_render = self.orient_total * self.camera_alpha\
                             + self.orient_render * (1-self.camera_alpha)
        self.speed = np.where(action[1] >= 0,
                              self.speed + self.real_action_range[1] * action[1] * (-np.tanh(0.5 * (self.speed - self.max_speed))),
                              self.speed + self.real_action_range[1] * action[1] * np.tanh(0.5 * self.speed))

        done1, end_cache = self._fast_range_finder(np.copy(self.position), np.copy(self.orient), self.speed[0], self.min_distance_to_obstacle, 'forward')

        self.position = np.copy(end_cache)

        self._get_observation(np.copy(self.position), np.copy(self.target), np.copy(self.orient))
        next_observation = np.copy(self.state)

        done1 = True if done1 < 1.0 else False
        done2 = (np.linalg.norm(self.position - self.target) <= self.min_distance_to_target)
        done3 = (np.linalg.norm(self.position - self.target) >= 1e4)

        # terminal judgement
        done = done1 + done2 + done3

        if done1:
            print('agent collides with obstacles!')
        if done2:
            print('agent arrived at the destination!')
        if done3:
            print('agent is too far from the target position!')

        reward_sparse = np.where(done2, np.zeros([1]) + 1.0, np.zeros([1]))
        reward = reward_sparse[0]

        info = {}
        if done2:
            info.update({'is_success': True})
        elif done1:
            info.update({'is_crash': True})
        elif done3:
            info.update({'is_termination': True})
        else:
            pass
        if done:
            self.terminate_render = True
        return next_observation, reward, done, info

    def reset(self):
        self.is_reset = True
        self.first_render = True
        self.terminate_render = False
        self._env_step_counter = 0
        self.mat_height = np.random.randint(
            1, self.total, size=(self.num_circle + 2 * self.expand, self.num_circle + 2 * self.expand)) * self.delta + self.lowest  # 建筑物高度
        W, H = np.shape(self.mat_height)
        for i in range(W):
            for j in range(H):
                if self.mat_height[i, j] > self.level + self.delta:
                    self.mat_height[i, j] = self.mat_height[i, j] + np.int(np.random.uniform(0, 200))

        self.mat_exist = self.mat_height - self.level  #
        while True:
            position = np.random.uniform(0, self.period, size=(2,))
            if np.linalg.norm(position - np.array([self.period / 2, self.period / 2])) - (self.radius + self.min_distance_to_obstacle) > 0:
                break
        relative_position = np.random.randint(0, self.num_circle, size=(2,)).astype(np.float)
        self.position = position + relative_position * self.period

        while True:
            target = np.random.uniform(0, self.period, size=(2,))
            if np.linalg.norm(target - np.array([self.period / 2, self.period / 2])) - self.radius > 0:
                break

        counter = 0
        while True:  # ensure that the minimum distance between initial position and target position is larger than 200m
            counter += 1
            relative_target = np.random.randint(0, self.num_circle, size=(2,)).astype(np.float)
            target_temp = np.array(target + relative_target * self.period)
            if np.linalg.norm(target_temp - self.position) >= self.min_initial_starts:

                self.target = target + relative_target * self.period
                self.orient = np.random.uniform(0, 2 * np.pi, size=(1,))
                self.orient_render = copy.deepcopy(self.orient)
                self.orient_total = copy.deepcopy(self.orient)
                self.orient_render_pre = copy.deepcopy(self.orient_render)
                self.orient_total_pre = copy.deepcopy(self.orient_total)
                self.speed = np.zeros([1])
                self._prepare_background_for_render()
                self._get_observation(np.copy(self.position), np.copy(self.target), np.copy(self.orient))
                observation = np.copy(self.state)
                break
            elif counter > 20:
                print("reset again")
                return self.reset()
            else:
                pass

        return observation

    def render(self, mode='human'):
        sleep_time = 0.1
        print('orient={}, speed={}'.format(self.orient, self.speed), self.orient_render)
        print("orient and total", self.orient, np.mod(self.orient_total, 2*np.pi), self.orient_total, self.orient_render)

        assert self.is_reset, 'the environment must be reset before rendering'
        if self.first_render:
            time.sleep(sleep_time)
            self.first_render = False
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(.2, .2, .2)
            # Render Window
            renderWindow = vtk.vtkRenderWindow()
            renderWindow.AddRenderer(renderer)
            renderWindow.SetSize(1600, 1600)
            self.Timer = TimerCallback(renderer)
            self.Timer.env_params = self.env_params

            def environment_render():
                renderWindowInteractor = vtk.vtkRenderWindowInteractor()
                renderWindowInteractor.SetRenderWindow(renderWindow)
                renderWindowInteractor.Initialize()
                renderWindowInteractor.AddObserver('TimerEvent', self.Timer.execute)
                timerId = renderWindowInteractor.CreateRepeatingTimer(30)
                self.Timer.timerId = timerId
                renderWindow.Start()
                renderWindowInteractor.Start()

            self.th = threading.Thread(target=environment_render, args=())
            self.th.start()
            time.sleep(1.0)
        else:
            self.Timer.terminate_render = self.terminate_render
            positions, directions, directions_camera = \
                Smoother_soft(self.agent_params_pre['position'], self.agent_params['position'],
                              self.orient_total_pre[0],
                              self.orient_total[0],
                              self.orient_render_pre[0],
                              self.orient_render[0])

            for i in range(len(directions)):
                directions[i] = np.array([np.mod(90 - np.mod(directions[i] / 2 / np.pi * 360.0, 360), 360)])

            for i in range(len(directions_camera)):
                directions_camera[i] = np.array([np.mod(165 - np.mod(directions_camera[i] / 2 / np.pi * 360.0, 360), 360)])

            for i in range(len(positions)):
                time.sleep(sleep_time / len(positions))
                agent_params_tmp = copy.deepcopy(self.agent_params)
                agent_params_tmp['position'] = positions[i]
                agent_params_tmp['direction'] = directions[i]
                agent_params_tmp['direction_camera'] = directions_camera[i]
                if i < len(positions) - 1:
                    agent_params_tmp['rangefinders'] = None
                self.Timer.agent_params = agent_params_tmp

        return self.speed, self.orient

    def seed(self, seed=None):
        if seed:
            if seed >= 0:
                np.random.seed(seed)


if __name__ == '__main__':
    env = UavEnv()
    env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        action[0] = 0.25 * action[0]
        obe, rew, done, info = env.step(action)
        env.render()
        if done:
            exit(0)
            env.reset()


