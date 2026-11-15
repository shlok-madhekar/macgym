"""Tiny batched 2D articulated rigid-body engine in MLX.

Maximal coordinates: every capsule is a free rigid body (x, y, theta, vx, vy, omega).
Revolute joints, joint limits and ground contacts are velocity-level constraints solved
with sequential impulses (Gauss-Seidel over joints, Baumgarte position correction),
inside a few substeps of semi-implicit Euler. Joint motors are direct torques
(torque = gear * action), like MuJoCo motor actuators.

Everything is elementwise math on (N,) arrays, one array per body/quantity, so
mx.compile fuses it into a handful of big GPU kernels. Batch size N is arbitrary;
the body/joint count B is small and static (Python loops over bodies are unrolled).

State layout: (N, 6*B) float32 = [px(B), py(B), th(B), vx(B), vy(B), w(B)].
"""
import math

import mlx.core as mx
import numpy as np

GRAVITY = -9.81


class Capsule:
    def __init__(self, p1, p2, radius, density=1000.0, color="#FFFFFF", name=""):
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.p2 = np.asarray(p2, dtype=np.float64)
        self.radius = float(radius)
        self.density = density
        self.color = color
        self.name = name


class Joint:
    def __init__(self, a, b, point, lo, hi, gear, damping=0.0, stiffness=0.0):
        """Revolute joint between body a (parent) and b (child) at world point `point`
        (initial configuration). Angle q = (th_b - th_a) - rest, limits [lo, hi] rad."""
        self.a, self.b = a, b
        self.point = np.asarray(point, dtype=np.float64)
        self.lo, self.hi = float(lo), float(hi)
        self.gear = float(gear)
        self.damping = float(damping)
        self.stiffness = float(stiffness)  # passive spring toward rest angle (Nm/rad)


class Model:
    def __init__(self, capsules, joints, dt=0.02, substeps=8, iters=4, mu=0.9,
                 joint_beta=0.2, contact_beta=0.2, slop=0.002, limit_beta=0.2,
                 max_v=30.0, max_w=60.0, motor_max_speed=10.0):
        self.capsules = capsules
        self.joints = joints
        self.B = len(capsules)
        self.J = len(joints)
        self.dt = dt
        self.substeps = substeps
        self.iters = iters
        self.mu = mu
        self.joint_beta = joint_beta
        self.contact_beta = contact_beta
        self.limit_beta = limit_beta
        self.slop = slop
        self.max_v = max_v
        self.max_w = max_w
        # DC-motor torque-speed curve: full torque at rest, zero when the joint already spins
        # at motor_max_speed (rad/s) in the commanded direction. Stops unbounded joint speeds.
        self.motor_max_speed = motor_max_speed

        # per-body static params + initial pose
        self.half, self.rad, self.inv_m, self.inv_i = [], [], [], []
        init = np.zeros((6, self.B))
        for k, c in enumerate(capsules):
            d = c.p2 - c.p1
            L = float(np.linalg.norm(d))
            r = c.radius
            m = c.density * (math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3)
            Ltot = L + 2 * r
            I = m * (Ltot * Ltot / 12.0 + r * r / 4.0)
            self.half.append(L / 2.0)
            self.rad.append(r)
            self.inv_m.append(1.0 / m)
            self.inv_i.append(1.0 / I)
            mid = (c.p1 + c.p2) / 2.0
            init[0, k], init[1, k] = mid
            init[2, k] = math.atan2(d[1], d[0])
        self.masses = [1.0 / im for im in self.inv_m]
        self.init_state = init.reshape(-1).astype(np.float32)  # (6B,)

        # joints: local anchors + rest angle
        self.la, self.lb, self.q0 = [], [], []
        for j in joints:
            for (body, store) in ((j.a, self.la), (j.b, self.lb)):
                cx, cy, th = init[0, body], init[1, body], init[2, body]
                dx, dy = j.point[0] - cx, j.point[1] - cy
                c, s = math.cos(th), math.sin(th)
                store.append((c * dx + s * dy, -s * dx + c * dy))  # R^T * d
            self.q0.append(init[2, j.b] - init[2, j.a])
        self.gears = [j.gear for j in joints]

    # ------------------------------------------------------------------ helpers
    def split(self, state):
        B = self.B
        cols = [state[:, i] for i in range(6 * B)]
        return [cols[q * B:(q + 1) * B] for q in range(6)]

    def joint_angles(self, state):
        """(N, J) joint angles and (N, J) joint velocities."""
        px, py, th, vx, vy, w = self.split(state)
        q = [th[j.b] - th[j.a] - q0 for j, q0 in zip(self.joints, self.q0)]
        qd = [w[j.b] - w[j.a] for j in self.joints]
        return mx.stack(q, axis=1), mx.stack(qd, axis=1)

    def endpoints(self, state):
        """Returns list of B tuples (x1, y1, x2, y2) of (N,) arrays."""
        px, py, th, vx, vy, w = self.split(state)
        out = []
        for k in range(self.B):
            hx = self.half[k] * mx.cos(th[k])
            hy = self.half[k] * mx.sin(th[k])
            out.append((px[k] - hx, py[k] - hy, px[k] + hx, py[k] + hy))
        return out

    def capsules_np(self, state_np):
        """state_np (R, 6B) numpy -> (R, B, 5) capsules."""
        B = self.B
        s = state_np.reshape(-1, 6, B)
        px, py, th = s[:, 0], s[:, 1], s[:, 2]
        half = np.asarray(self.half, dtype=np.float32)[None]
        hx, hy = half * np.cos(th), half * np.sin(th)
        rad = np.broadcast_to(np.asarray(self.rad, dtype=np.float32)[None], px.shape)
        return np.stack([px - hx, py - hy, px + hx, py + hy, rad], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------ physics
    def step(self, state, action):
        """One control step (dt) for all envs. state (N, 6B), action (N, J) in [-1, 1]."""
        B, J = self.B, self.J
        h = self.dt / self.substeps
        inv_h = 1.0 / h
        im, ii = self.inv_m, self.inv_i
        px, py, th, vx, vy, w = [list(x) for x in self.split(state)]
        action = mx.clip(action, -1.0, 1.0)

        # motor torques (constant over the control step)
        tau = [None] * B
        for jn, j in enumerate(self.joints):
            a_ = action[:, jn]
            qd = w[j.b] - w[j.a]
            t = self.gears[jn] * a_ * mx.clip(1.0 - mx.where(a_ >= 0, qd, -qd) / self.motor_max_speed, 0.0, 1.0)
            tau[j.b] = t if tau[j.b] is None else tau[j.b] + t
            tau[j.a] = -t if tau[j.a] is None else tau[j.a] - t

        # accumulated impulses, warm-started across substeps (Box2D style): the solver starts
        # each substep from last substep's contact/joint forces, so the normal force carried by a
        # light foot under a heavy torso converges in a few iterations and the Coulomb friction
        # bound mu*N is the real one (without it N is underestimated and feet skate).
        nC = 2 * B
        jacc = [[0.0, 0.0] for _ in range(J)]
        lim_acc = [[0.0, 0.0] for _ in range(J)]
        cn_acc = [0.0] * nC
        ct_acc = [0.0] * nC
        for _ in range(self.substeps):
            # ---- external forces: gravity, motors, passive joint damping
            for k in range(B):
                vy[k] = vy[k] + h * GRAVITY
                if tau[k] is not None:
                    w[k] = w[k] + (h * ii[k]) * tau[k]
            for jn, j in enumerate(self.joints):
                if j.damping > 0 or j.stiffness > 0:
                    d = (-j.damping * (w[j.b] - w[j.a])
                         - j.stiffness * (th[j.b] - th[j.a] - self.q0[jn]))
                    w[j.b] = w[j.b] + (h * ii[j.b]) * d
                    w[j.a] = w[j.a] - (h * ii[j.a]) * d

            cs = [mx.cos(t) for t in th]
            sn = [mx.sin(t) for t in th]

            # ---- joint setup (anchors, effective mass inverse, bias)
            jdat = []
            for jn, j in enumerate(self.joints):
                a, b = j.a, j.b
                lax, lay = self.la[jn]
                lbx, lby = self.lb[jn]
                rax = cs[a] * lax - sn[a] * lay
                ray = sn[a] * lax + cs[a] * lay
                rbx = cs[b] * lbx - sn[b] * lby
                rby = sn[b] * lbx + cs[b] * lby
                cx = (px[b] + rbx) - (px[a] + rax)
                cy = (py[b] + rby) - (py[a] + ray)
                msum = im[a] + im[b]
                k11 = msum + ii[a] * ray * ray + ii[b] * rby * rby
                k22 = msum + ii[a] * rax * rax + ii[b] * rbx * rbx
                k12 = -ii[a] * rax * ray - ii[b] * rbx * rby
                det = k11 * k22 - k12 * k12
                idet = 1.0 / mx.maximum(det, 1e-9)
                bx = (self.joint_beta * inv_h) * cx
                by = (self.joint_beta * inv_h) * cy
                # joint limits
                q = th[b] - th[a] - self.q0[jn]
                lo_c = q - j.lo  # >= 0 ok
                hi_c = j.hi - q  # >= 0 ok
                lo_bias = mx.where(lo_c < 0, self.limit_beta * inv_h * lo_c, lo_c * inv_h)
                hi_bias = mx.where(hi_c < 0, self.limit_beta * inv_h * hi_c, hi_c * inv_h)
                ang_m = 1.0 / (ii[a] + ii[b])
                jdat.append((a, b, rax, ray, rbx, rby, k11 * idet, k22 * idet, k12 * idet,
                             bx, by, lo_bias, hi_bias, ang_m))

            # ---- contact setup: two endpoint spheres per capsule (speculative)
            cdat = []
            for k in range(B):
                for sgn in (-1.0, 1.0):
                    ox = sgn * self.half[k] * cs[k]
                    oy = sgn * self.half[k] * sn[k] - self.rad[k]  # bottom of sphere
                    pen = -(py[k] + oy)  # >0 means penetrating
                    bias = mx.where(pen > self.slop,
                                    self.contact_beta * inv_h * (pen - self.slop),
                                    mx.minimum(pen, 0.0) * inv_h)
                    kn = 1.0 / (im[k] + ii[k] * ox * ox)
                    kt = 1.0 / (im[k] + ii[k] * oy * oy)
                    cdat.append((k, ox, oy, bias, kn, kt))

            # ---- warm start: re-apply last substep's accumulated impulses
            for jn in range(J):
                a, b, rax, ray, rbx, rby = jdat[jn][:6]
                lx, ly = jacc[jn]
                dl = lim_acc[jn][0] - lim_acc[jn][1]
                vx[a] = vx[a] - im[a] * lx
                vy[a] = vy[a] - im[a] * ly
                w[a] = w[a] - ii[a] * (rax * ly - ray * lx) - ii[a] * dl
                vx[b] = vx[b] + im[b] * lx
                vy[b] = vy[b] + im[b] * ly
                w[b] = w[b] + ii[b] * (rbx * ly - rby * lx) + ii[b] * dl
            for ci, (k, ox, oy, bias, kn, kt) in enumerate(cdat):
                dn, dt_ = cn_acc[ci], ct_acc[ci]
                vy[k] = vy[k] + im[k] * dn
                vx[k] = vx[k] + im[k] * dt_
                w[k] = w[k] + ii[k] * (ox * dn - oy * dt_)

            # ---- sequential impulse iterations
            for _it in range(self.iters):
                for jn in range(J):
                    (a, b, rax, ray, rbx, rby, i11, i22, i12, bx, by,
                     lo_bias, hi_bias, ang_m) = jdat[jn]
                    # point constraint
                    dvx = (vx[b] - w[b] * rby) - (vx[a] - w[a] * ray) + bx
                    dvy = (vy[b] + w[b] * rbx) - (vy[a] + w[a] * rax) + by
                    # lambda = -K^-1 dv ; K^-1 = [[k22,-k12],[-k12,k11]]/det
                    lx = -(i22 * dvx - i12 * dvy)
                    ly = -(-i12 * dvx + i11 * dvy)
                    jacc[jn][0] = jacc[jn][0] + lx
                    jacc[jn][1] = jacc[jn][1] + ly
                    vx[a] = vx[a] - im[a] * lx
                    vy[a] = vy[a] - im[a] * ly
                    w[a] = w[a] - ii[a] * (rax * ly - ray * lx)
                    vx[b] = vx[b] + im[b] * lx
                    vy[b] = vy[b] + im[b] * ly
                    w[b] = w[b] + ii[b] * (rbx * ly - rby * lx)
                    # lower limit: (w_b - w_a) >= -lo_bias
                    rel = w[b] - w[a]
                    old = lim_acc[jn][0]
                    new = mx.maximum(old - ang_m * (rel + lo_bias), 0.0)
                    dl = new - old
                    lim_acc[jn][0] = new
                    # upper limit: (w_a - w_b) >= -hi_bias
                    w[b] = w[b] + ii[b] * dl
                    w[a] = w[a] - ii[a] * dl
                    rel = w[a] - w[b]
                    old = lim_acc[jn][1]
                    new = mx.maximum(old - ang_m * (rel + hi_bias), 0.0)
                    du = new - old
                    lim_acc[jn][1] = new
                    w[b] = w[b] - ii[b] * du
                    w[a] = w[a] + ii[a] * du

                for ci, (k, ox, oy, bias, kn, kt) in enumerate(cdat):
                    # normal (y): vn = vy + w*ox
                    vn = vy[k] + w[k] * ox
                    old = cn_acc[ci]
                    new = mx.maximum(old + kn * (bias - vn), 0.0)
                    dn = new - old
                    cn_acc[ci] = new
                    vy[k] = vy[k] + im[k] * dn
                    w[k] = w[k] + ii[k] * ox * dn
                    # friction (x): vt = vx - w*oy
                    vt = vx[k] - w[k] * oy
                    lim = self.mu * new
                    oldt = ct_acc[ci]
                    newt = mx.clip(oldt - kt * vt, -lim, lim)
                    dt_ = newt - oldt
                    ct_acc[ci] = newt
                    vx[k] = vx[k] + im[k] * dt_
                    w[k] = w[k] - ii[k] * oy * dt_

            # ---- clamp + integrate
            for k in range(B):
                vx[k] = mx.clip(vx[k], -self.max_v, self.max_v)
                vy[k] = mx.clip(vy[k], -self.max_v, self.max_v)
                w[k] = mx.clip(w[k], -self.max_w, self.max_w)
                px[k] = px[k] + h * vx[k]
                py[k] = py[k] + h * vy[k]
                th[k] = th[k] + h * w[k]

        return mx.stack(px + py + th + vx + vy + w, axis=1)

    # ------------------------------------------------------------------ fast path
    def _metal_source(self):
        B, J = self.B, self.J
        def arr(name, vals, typ="float"):
            return "constant %s %s[%d] = {%s};\n" % (typ, name, len(vals), ", ".join(
                ("%.9ef" % v) if typ == "float" else str(int(v)) for v in vals))
        hdr = ""
        hdr += arr("IM", self.inv_m) + arr("II", self.inv_i) + arr("HALF", self.half) + arr("RAD", self.rad)
        hdr += arr("JA", [j.a for j in self.joints], "int") + arr("JB", [j.b for j in self.joints], "int")
        hdr += arr("LAX", [a[0] for a in self.la]) + arr("LAY", [a[1] for a in self.la])
        hdr += arr("LBX", [a[0] for a in self.lb]) + arr("LBY", [a[1] for a in self.lb])
        hdr += arr("Q0", self.q0) + arr("LO", [j.lo for j in self.joints]) + arr("HI", [j.hi for j in self.joints])
        hdr += arr("GEAR", self.gears) + arr("DAMP", [j.damping for j in self.joints]) + arr("STIFF", [j.stiffness for j in self.joints])
        h = self.dt / self.substeps
        consts = dict(B=B, J=J, C=2 * B, SUB=self.substeps, ITERS=self.iters, H="%.9ef" % h,
                      INVH="%.9ef" % (1.0 / h), G="%.9ef" % GRAVITY, MU="%.9ef" % self.mu,
                      JBETA="%.9ef" % self.joint_beta, CBETA="%.9ef" % self.contact_beta,
                      LBETA="%.9ef" % self.limit_beta, SLOP="%.9ef" % self.slop,
                      MAXV="%.9ef" % self.max_v, QDMAX="%.9ef" % self.motor_max_speed, MAXW="%.9ef" % self.max_w)
        for k, v in consts.items():
            hdr += "#define %s %s\n" % (k, v)
        src = r"""
        uint e = thread_position_in_grid.x;
        float px[B], py[B], th[B], vx[B], vy[B], w[B], tau[B];
        uint base = e * 6 * B;
        for (int k = 0; k < B; k++) {
            px[k] = state[base + k]; py[k] = state[base + B + k]; th[k] = state[base + 2*B + k];
            vx[k] = state[base + 3*B + k]; vy[k] = state[base + 4*B + k]; w[k] = state[base + 5*B + k];
            tau[k] = 0.0f;
        }
        for (int j = 0; j < J; j++) {
            float a_ = clamp((float)action[e * J + j], -1.0f, 1.0f);
            float qd = w[JB[j]] - w[JA[j]];
            float t = GEAR[j] * a_ * clamp(1.0f - (a_ >= 0.0f ? qd : -qd) / QDMAX, 0.0f, 1.0f);
            tau[JB[j]] += t; tau[JA[j]] -= t;
        }
        float jlx[J], jly[J], la0[J], la1[J], cn[C], ct[C];
        for (int j = 0; j < J; j++) { jlx[j] = 0.0f; jly[j] = 0.0f; la0[j] = 0.0f; la1[j] = 0.0f; }
        for (int c = 0; c < C; c++) { cn[c] = 0.0f; ct[c] = 0.0f; }
        for (int s = 0; s < SUB; s++) {
            for (int k = 0; k < B; k++) { vy[k] += H * G; w[k] += H * II[k] * tau[k]; }
            for (int j = 0; j < J; j++) {
                int a = JA[j], b = JB[j];
                float d = -DAMP[j] * (w[b] - w[a]) - STIFF[j] * (th[b] - th[a] - Q0[j]);
                w[b] += H * II[b] * d; w[a] -= H * II[a] * d;
            }
            float cs[B], sn[B];
            for (int k = 0; k < B; k++) { sn[k] = metal::sincos(th[k], cs[k]); }
            float rax[J], ray[J], rbx[J], rby[J], i11[J], i22[J], i12[J], bx[J], by[J], lob[J], hib[J], angm[J];
            for (int j = 0; j < J; j++) {
                int a = JA[j], b = JB[j];
                rax[j] = cs[a] * LAX[j] - sn[a] * LAY[j]; ray[j] = sn[a] * LAX[j] + cs[a] * LAY[j];
                rbx[j] = cs[b] * LBX[j] - sn[b] * LBY[j]; rby[j] = sn[b] * LBX[j] + cs[b] * LBY[j];
                float cx = (px[b] + rbx[j]) - (px[a] + rax[j]);
                float cy = (py[b] + rby[j]) - (py[a] + ray[j]);
                float ms = IM[a] + IM[b];
                float k11 = ms + II[a] * ray[j] * ray[j] + II[b] * rby[j] * rby[j];
                float k22 = ms + II[a] * rax[j] * rax[j] + II[b] * rbx[j] * rbx[j];
                float k12 = -II[a] * rax[j] * ray[j] - II[b] * rbx[j] * rby[j];
                float idet = 1.0f / max(k11 * k22 - k12 * k12, 1e-9f);
                i11[j] = k11 * idet; i22[j] = k22 * idet; i12[j] = k12 * idet;
                bx[j] = JBETA * INVH * cx; by[j] = JBETA * INVH * cy;
                float q = th[b] - th[a] - Q0[j];
                float loc = q - LO[j], hic = HI[j] - q;
                lob[j] = loc < 0.0f ? LBETA * INVH * loc : loc * INVH;
                hib[j] = hic < 0.0f ? LBETA * INVH * hic : hic * INVH;
                angm[j] = 1.0f / (II[a] + II[b]);
            }
            float ox[C], oy[C], cb[C], kn[C], kt[C];
            for (int c = 0; c < C; c++) {
                int k = c >> 1; float sg = (c & 1) ? 1.0f : -1.0f;
                ox[c] = sg * HALF[k] * cs[k];
                oy[c] = sg * HALF[k] * sn[k] - RAD[k];
                float pen = -(py[k] + oy[c]);
                cb[c] = pen > SLOP ? CBETA * INVH * (pen - SLOP) : min(pen, 0.0f) * INVH;
                kn[c] = 1.0f / (IM[k] + II[k] * ox[c] * ox[c]);
                kt[c] = 1.0f / (IM[k] + II[k] * oy[c] * oy[c]);
            }
            for (int j = 0; j < J; j++) {
                int a = JA[j], b = JB[j];
                float lx = jlx[j], ly = jly[j], dl = la0[j] - la1[j];
                vx[a] -= IM[a] * lx; vy[a] -= IM[a] * ly; w[a] = w[a] - II[a] * (rax[j] * ly - ray[j] * lx) - II[a] * dl;
                vx[b] += IM[b] * lx; vy[b] += IM[b] * ly; w[b] = w[b] + II[b] * (rbx[j] * ly - rby[j] * lx) + II[b] * dl;
            }
            for (int c = 0; c < C; c++) {
                int k = c >> 1;
                vy[k] += IM[k] * cn[c]; vx[k] += IM[k] * ct[c];
                w[k] += II[k] * (ox[c] * cn[c] - oy[c] * ct[c]);
            }
            for (int it = 0; it < ITERS; it++) {
                for (int j = 0; j < J; j++) {
                    int a = JA[j], b = JB[j];
                    float dvx = (vx[b] - w[b] * rby[j]) - (vx[a] - w[a] * ray[j]) + bx[j];
                    float dvy = (vy[b] + w[b] * rbx[j]) - (vy[a] + w[a] * rax[j]) + by[j];
                    float lx = -(i22[j] * dvx - i12[j] * dvy);
                    float ly = -(-i12[j] * dvx + i11[j] * dvy);
                    jlx[j] += lx; jly[j] += ly;
                    vx[a] -= IM[a] * lx; vy[a] -= IM[a] * ly; w[a] -= II[a] * (rax[j] * ly - ray[j] * lx);
                    vx[b] += IM[b] * lx; vy[b] += IM[b] * ly; w[b] += II[b] * (rbx[j] * ly - rby[j] * lx);
                    float rel = w[b] - w[a];
                    float nw = max(la0[j] - angm[j] * (rel + lob[j]), 0.0f);
                    float dl = nw - la0[j]; la0[j] = nw;
                    w[b] += II[b] * dl; w[a] -= II[a] * dl;
                    rel = w[a] - w[b];
                    nw = max(la1[j] - angm[j] * (rel + hib[j]), 0.0f);
                    float du = nw - la1[j]; la1[j] = nw;
                    w[b] -= II[b] * du; w[a] += II[a] * du;
                }
                for (int c = 0; c < C; c++) {
                    int k = c >> 1;
                    float vn = vy[k] + w[k] * ox[c];
                    float nw = max(cn[c] + kn[c] * (cb[c] - vn), 0.0f);
                    float dn = nw - cn[c]; cn[c] = nw;
                    vy[k] += IM[k] * dn; w[k] += II[k] * ox[c] * dn;
                    float vt = vx[k] - w[k] * oy[c];
                    float lim = MU * nw;
                    float nt = clamp(ct[c] - kt[c] * vt, -lim, lim);
                    float dt_ = nt - ct[c]; ct[c] = nt;
                    vx[k] += IM[k] * dt_; w[k] -= II[k] * oy[c] * dt_;
                }
            }
            for (int k = 0; k < B; k++) {
                vx[k] = clamp(vx[k], -MAXV, MAXV); vy[k] = clamp(vy[k], -MAXV, MAXV); w[k] = clamp(w[k], -MAXW, MAXW);
                px[k] += H * vx[k]; py[k] += H * vy[k]; th[k] += H * w[k];
            }
        }
        for (int k = 0; k < B; k++) {
            out[base + k] = px[k]; out[base + B + k] = py[k]; out[base + 2*B + k] = th[k];
            out[base + 3*B + k] = vx[k]; out[base + 4*B + k] = vy[k]; out[base + 5*B + k] = w[k];
        }
        """
        return hdr, src

    def step_metal(self, state, action):
        """Same algorithm as step(), as one fused Metal kernel (one GPU thread per env)."""
        if not hasattr(self, "_kernel"):
            hdr, src = self._metal_source()
            self._kernel = mx.fast.metal_kernel(
                name="macgym_phys2d_%d" % id(self), input_names=["state", "action"],
                output_names=["out"], source=src, header=hdr)
        n = state.shape[0]
        return self._kernel(
            inputs=[state.astype(mx.float32), action.astype(mx.float32)],
            grid=(n, 1, 1), threadgroup=(min(64, n), 1, 1),
            output_shapes=[state.shape], output_dtypes=[mx.float32])[0]


def state_bad(state, limit=1e3):
    """(N,) bool: any non-finite or absurd value."""
    return mx.any(mx.isnan(state) | (mx.abs(state) > limit), axis=1)


# ---------------------------------------------------------------------------
# Shared batched locomotion env on top of Model (used by walker.py / cheetah.py)
# ---------------------------------------------------------------------------
from macgym.envs.base import BatchedEnv  # noqa: E402


class LocoEnv(BatchedEnv):
    """Subclasses set: name, description, obs_dim, act_dim, max_episode_steps, dt and
    implement build_model(), observe(state), reward_done(prev, state, action)."""
    torso = 0
    init_vel_noise = 0.1
    init_ang_noise = 0.02
    view_half_width = 2.5
    view_y = (-0.3, 2.0)
    backend = "metal"  # or "mx" (pure mx ops, slower to compile/run)

    def build_model(self) -> Model:
        raise NotImplementedError

    def observe(self, state):
        raise NotImplementedError

    def reward_done(self, prev, state, action):
        raise NotImplementedError

    def __init__(self, num_envs: int, seed: int = 0, backend: str = None):
        super().__init__(num_envs, seed)
        if backend is not None:
            self.backend = backend
        self.model = self.build_model()
        self.dt = self.model.dt
        self.init_state = mx.array(self.model.init_state)
        # torso start x (reset noise is velocity-only, so every episode starts here)
        self.start_x0 = float(np.asarray(self.model.init_state)[self.torso])
        self.colors = [c.color for c in self.model.capsules]
        self._step_fn = mx.compile(self._env_step)
        self._sim_fn = mx.compile(self.model.step_metal if self.backend == "metal" else self.model.step)
        self.reset()

    # fresh randomized states (pure mx)
    def _fresh(self, key, n):
        B = self.model.B
        k1, k2 = mx.random.split(key)
        base = mx.broadcast_to(self.init_state[None], (n, 6 * B))
        noise = mx.random.uniform(-1.0, 1.0, (n, 6 * B), key=k1)
        scale = mx.concatenate([
            mx.zeros((3 * B,)),
            mx.full((2 * B,), self.init_vel_noise),
            mx.full((B,), self.init_vel_noise * 2.0),
        ])
        return base + noise * scale[None]

    def foot_contacts(self, state, feet):
        eps = self.model.endpoints(state)
        out = []
        for f in feet:
            x1, y1, x2, y2 = eps[f]
            out.append((mx.minimum(y1, y2) - self.model.rad[f] < 0.02).astype(mx.float32))
        return out

    def _env_step(self, state, t, ret, action, key):
        n = state.shape[0]
        new_state = (self.model.step_metal if self.backend == "metal" else self.model.step)(state, action)
        reward, done = self.reward_done(state, new_state, action)
        bad = state_bad(new_state)
        reward = mx.where(bad, 0.0, reward)
        t = t + 1.0
        done = done | bad | (t >= self.max_episode_steps)
        donef = done.astype(mx.float32)
        ret = ret + reward
        ep_ret = ret * donef
        ep_len = t * donef
        # torso x travelled this episode (start-relative), valid where done
        ep_dist = mx.where(bad, 0.0, new_state[:, self.torso] - self.start_x0) * donef
        new_state = mx.where(done[:, None], self._fresh(key, n), new_state)
        t = mx.where(done, 0.0, t)
        ret = mx.where(done, 0.0, ret)
        obs = self.observe(new_state)
        return new_state, t, ret, obs, reward, donef, ep_ret, ep_len, ep_dist

    def reset(self) -> mx.array:
        n = self.num_envs
        self.state = self._fresh(self._split(), n)
        self.t = mx.zeros((n,), dtype=mx.float32)
        self.ret = mx.zeros((n,), dtype=mx.float32)
        self.obs = self.observe(self.state)
        mx.eval(self.state, self.t, self.ret, self.obs)
        return self.obs

    def step(self, action: mx.array):
        action = action.astype(mx.float32).reshape(self.num_envs, self.act_dim)
        (self.state, self.t, self.ret, self.obs, reward, done, ep_ret, ep_len, ep_dist) = self._step_fn(
            self.state, self.t, self.ret, action, self._split())
        info = {"episode_return": ep_ret, "episode_len": ep_len, "episode_dist": ep_dist}
        return self.obs, reward, done, info

    def render_state(self, R: int) -> dict:
        R = min(R, self.num_envs)
        s = np.array(self.state[:R], dtype=np.float32)
        caps = self.model.capsules_np(s)
        cam = s[:, self.torso]  # torso x
        cam = np.nan_to_num(cam).astype(np.float32)
        hw = self.view_half_width
        return {
            "capsules": caps,
            "targets": None,
            "colors": list(self.colors),
            "ground": 0.0,
            "world_bounds": [-hw, hw, self.view_y[0], self.view_y[1]],
            "camera_x": cam,
            "start_x": np.full((R,), self.start_x0, np.float32),  # per-env episode start x
            "dist": cam - self.start_x0,  # current start-relative distance (m)
        }

    def live_dist_mean(self) -> float:
        """Mean start-relative torso x over all envs (in-progress episodes)."""
        d = self.state[:, self.torso] - self.start_x0
        d = mx.where(mx.isnan(d), 0.0, d)
        return float(mx.mean(d))
