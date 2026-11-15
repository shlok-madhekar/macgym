"""Walker2D: planar biped (torso + 2x thigh/shin/foot) on the MLX 2D rigid-body engine."""
import math

import mlx.core as mx

from macgym.envs.physics2d import Capsule, Joint, LocoEnv, Model

TORSO_C = "#E8ECF4"
RIGHT_C = "#4F8CFF"
LEFT_C = "#FF6B9A"
INIT_TORSO_Y = 1.25  # torso center height at reset


class Walker2D(LocoEnv):
    name = "walker"
    description = ("Planar biped, 7 capsules / 6 torque joints. Reward = forward velocity "
                   "+ 1 alive bonus - 1e-3|a|^2; falls end the episode.")
    obs_dim = 24
    act_dim = 6
    max_episode_steps = 1000
    dt = 0.02
    torso = 0
    feet = (3, 6)
    view_half_width = 2.5
    view_y = (-0.3, 2.0)

    def build_model(self):
        caps = [Capsule((0, 1.05), (0, 1.45), 0.05, color=TORSO_C, name="torso")]
        joints = []
        for side, col in (("r", RIGHT_C), ("l", LEFT_C)):
            base = len(caps)
            caps += [
                Capsule((0, 1.05), (0, 0.6), 0.05, color=col, name=side + "_thigh"),
                Capsule((0, 0.6), (0, 0.1), 0.04, color=col, name=side + "_shin"),
                Capsule((0, 0.1), (0.2, 0.1), 0.06, color=col, name=side + "_foot"),
            ]
            joints += [
                Joint(0, base, (0, 1.05), -0.8, 2.0, gear=100, damping=0.1),         # hip
                Joint(base, base + 1, (0, 0.6), -2.6, 0.0, gear=100, damping=0.1),   # knee
                Joint(base + 1, base + 2, (0, 0.1), -0.8, 0.8, gear=100, damping=0.1),  # ankle
            ]
        return Model(caps, joints, dt=self.dt, substeps=8, iters=4, mu=0.9)

    def observe(self, state):
        m = self.model
        px, py, th, vx, vy, w = m.split(state)
        ang = th[0] - math.pi / 2
        q, qd = m.joint_angles(state)
        contacts = self.foot_contacts(state, self.feet)
        feet = []
        for f in self.feet:
            feet += [px[f] - px[0], py[f]]
        cols = [py[0], mx.sin(ang), mx.cos(ang), vx[0], vy[0], w[0] * 0.1]
        obs = mx.concatenate([mx.stack(cols, axis=1), q, qd * 0.1,
                              mx.stack(contacts + feet, axis=1)], axis=1)
        return mx.clip(obs, -50.0, 50.0)

    def reward_done(self, prev, state, action):
        B = self.model.B
        x0, x1 = prev[:, 0], state[:, 0]
        y = state[:, B]
        ang = state[:, 2 * B] - math.pi / 2
        fwd = (x1 - x0) / self.dt
        reward = fwd + 1.0 - 1e-3 * mx.sum(action * action, axis=1)
        done = (y < 0.7 * INIT_TORSO_Y) | (mx.abs(ang) > 1.0)
        return reward, done
