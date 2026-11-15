"""Cheetah2D: HalfCheetah-like planar runner (torso + 2x thigh/shin/foot) on the MLX 2D engine."""
import mlx.core as mx

from macgym.envs.physics2d import Capsule, Joint, LocoEnv, Model

TORSO_C = "#E8ECF4"
BACK_C = "#FFB547"
FRONT_C = "#3DDC97"
H = 0.7  # torso height at reset


class Cheetah2D(LocoEnv):
    name = "cheetah"
    description = ("HalfCheetah-like planar runner, 7 capsules / 6 torque joints. "
                   "Reward = forward velocity - 0.1|a|^2; never falls, 1000-step episodes.")
    obs_dim = 17
    act_dim = 6
    max_episode_steps = 1000
    dt = 0.02
    torso = 0
    feet = (3, 6)
    view_half_width = 2.5
    view_y = (-0.3, 1.6)

    def build_model(self):
        r = 0.046
        hip_b, knee_b, ank_b, toe_b = (-0.5, H), (-0.34, H - 0.25), (-0.62, H - 0.39), (-0.56, H - 0.58)
        hip_f, knee_f, ank_f, toe_f = (0.5, H), (0.36, H - 0.24), (0.49, H - 0.42), (0.58, H - 0.56)
        caps = [
            Capsule((-0.5, H), (0.5, H), r, color=TORSO_C, name="torso"),
            Capsule(hip_b, knee_b, r, color=BACK_C, name="bthigh"),
            Capsule(knee_b, ank_b, r, color=BACK_C, name="bshin"),
            Capsule(ank_b, toe_b, r, color=BACK_C, name="bfoot"),
            Capsule(hip_f, knee_f, r, color=FRONT_C, name="fthigh"),
            Capsule(knee_f, ank_f, r, color=FRONT_C, name="fshin"),
            Capsule(ank_f, toe_f, r, color=FRONT_C, name="ffoot"),
        ]
        joints = [
            Joint(0, 1, hip_b, -1.05, 0.52, gear=120, damping=6.0, stiffness=240),
            Joint(1, 2, knee_b, -0.785, 0.785, gear=90, damping=4.5, stiffness=180),
            Joint(2, 3, ank_b, -0.785, 0.4, gear=60, damping=3.0, stiffness=120),
            Joint(0, 4, hip_f, -0.7, 1.0, gear=120, damping=4.5, stiffness=180),
            Joint(4, 5, knee_f, -0.87, 1.2, gear=60, damping=3.0, stiffness=120),
            Joint(5, 6, ank_f, -0.5, 0.5, gear=30, damping=1.5, stiffness=60),
        ]
        return Model(caps, joints, dt=self.dt, substeps=8, iters=4, mu=0.4)

    def observe(self, state):
        m = self.model
        px, py, th, vx, vy, w = m.split(state)
        q, qd = m.joint_angles(state)
        head = mx.stack([py[0], th[0]], axis=1)
        vel = mx.stack([vx[0], vy[0], w[0] * 0.1], axis=1)
        obs = mx.concatenate([head, q, vel, qd * 0.1], axis=1)
        return mx.clip(obs, -50.0, 50.0)

    def reward_done(self, prev, state, action):
        fwd = (state[:, 0] - prev[:, 0]) / self.dt
        reward = fwd - 0.1 * mx.sum(action * action, axis=1)
        done = mx.zeros(reward.shape, dtype=mx.bool_)
        return reward, done
