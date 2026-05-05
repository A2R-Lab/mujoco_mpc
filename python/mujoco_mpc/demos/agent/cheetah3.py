"""MPC rollout demo for the three-legged cheetah task."""

from pathlib import Path
import argparse

import mujoco
import numpy as np

from mujoco_mpc import agent as agent_lib


def repo_root() -> Path:
  return Path(__file__).resolve().parents[4]


def joint_index(model: mujoco.MjModel, name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if joint_id < 0:
    raise ValueError(f"Joint {name!r} not found")
  return model.jnt_qposadr[joint_id]


def camera_id(model: mujoco.MjModel, name: str) -> int:
  cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
  if cam_id < 0:
    raise ValueError(f"Camera {name!r} not found")
  return cam_id


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--planner_steps", type=int, default=2)
  parser.add_argument("--speed", type=float, default=2.0)
  parser.add_argument("--height", type=float, default=0.7)
  parser.add_argument("--camera", default="side")
  parser.add_argument("--no_render", action="store_true")
  parser.add_argument("--no_plots", action="store_true")
  args = parser.parse_args()

  root = repo_root()
  model_path = root / "build" / "mjpc" / "tasks" / "cheetah" / "task.xml"
  server_path = root / "build" / "bin" / "agent_server"
  if not model_path.exists():
    raise FileNotFoundError(
        f"{model_path} does not exist. Run: cmake --build build --target mjpc"
    )
  if not server_path.exists():
    raise FileNotFoundError(
        f"{server_path} does not exist. Run: cmake --build build --target agent_server"
    )

  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)
  cam_id = camera_id(model, args.camera)

  renderer = None
  frames = []
  if not args.no_render:
    renderer = mujoco.Renderer(model, height=480, width=640)

  agent = agent_lib.Agent(
      task_id="Three-Legged Cheetah",
      model=model,
      server_binary_path=str(server_path),
  )
  try:
    agent.set_cost_weights({
        "Control": 0.1,
        "Height": 10.0,
        "Rotation": 3.0,
        "Speed": 1.0,
    })
    agent.set_task_parameters({
        "Height Goal": args.height,
        "Speed Goal": args.speed,
    })
    print("Cost weights:", agent.get_cost_weights())
    print("Task params:", agent.get_task_parameters())

    steps = args.steps
    qpos = np.zeros((model.nq, steps))
    qvel = np.zeros((model.nv, steps))
    ctrl = np.zeros((model.nu, steps - 1))
    time = np.zeros(steps)
    cost_total = np.zeros(steps - 1)
    cost_terms = np.zeros((len(agent.get_cost_term_values()), steps - 1))

    mujoco.mj_resetData(model, data)
    qpos[:, 0] = data.qpos
    qvel[:, 0] = data.qvel
    time[0] = data.time

    for t in range(steps - 1):
      if t % 100 == 0:
        print(f"Step {t}/{steps}")

      agent.set_state(
          time=data.time,
          qpos=data.qpos,
          qvel=data.qvel,
          act=data.act,
          mocap_pos=data.mocap_pos,
          mocap_quat=data.mocap_quat,
          userdata=data.userdata,
      )
      for _ in range(args.planner_steps):
        agent.planner_step()

      data.ctrl = agent.get_action()
      ctrl[:, t] = data.ctrl
      cost_total[t] = agent.get_total_cost()
      for i, (_, value) in enumerate(agent.get_cost_term_values().items()):
        cost_terms[i, t] = value

      mujoco.mj_step(model, data)
      qpos[:, t + 1] = data.qpos
      qvel[:, t + 1] = data.qvel
      time[t + 1] = data.time

      if renderer is not None:
        renderer.update_scene(data, camera=cam_id)
        frames.append(renderer.render())

    print(f"Final qpos: {data.qpos}")
    print(f"Average total cost: {np.mean(cost_total):.6f}")

  finally:
    agent.close()

  if args.no_plots:
    return

  import matplotlib.pyplot as plt
  from matplotlib import animation

  plt.figure(figsize=(12, 8))
  plt.subplot(3, 1, 1)
  plt.plot(time, qpos[joint_index(model, "rootx")], label="rootx")
  plt.plot(time, qpos[joint_index(model, "rootz")], label="rootz")
  plt.legend()
  plt.ylabel("Root Position")
  plt.grid(True)
  plt.title("Three-Legged Cheetah Root Position")

  plt.subplot(3, 1, 2)
  for name in ["bthigh", "bshin", "bfoot"]:
    plt.plot(time, qpos[joint_index(model, name)], label=name)
  plt.legend()
  plt.ylabel("Back Leg Joints")
  plt.grid(True)

  plt.subplot(3, 1, 3)
  for name in ["mthigh", "mshin", "mfoot", "fthigh", "fshin", "ffoot"]:
    plt.plot(time, qpos[joint_index(model, name)], label=name)
  plt.legend()
  plt.xlabel("Time (s)")
  plt.ylabel("Middle/Front Joints")
  plt.grid(True)
  plt.tight_layout()

  plt.figure(figsize=(12, 8))
  actuator_names = [
      "bthigh",
      "bshin",
      "bfoot",
      "mthigh",
      "mshin",
      "mfoot",
      "fthigh",
      "fshin",
      "ffoot",
  ]
  for i, name in enumerate(actuator_names):
    plt.subplot(3, 3, i + 1)
    plt.plot(time[:-1], ctrl[i])
    plt.title(name)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
  plt.tight_layout()

  plt.figure(figsize=(10, 6))
  for i, name in enumerate(["Control", "Height", "Rotation", "Speed"]):
    plt.plot(time[:-1], cost_terms[i], label=name)
  plt.plot(time[:-1], cost_total, label="Total", color="black", linewidth=2)
  plt.legend()
  plt.xlabel("Time (s)")
  plt.ylabel("Cost")
  plt.title("Three-Legged Cheetah Cost Terms")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  if frames:
    fig_anim = plt.figure(figsize=(10, 6))
    img = plt.imshow(frames[0])
    plt.axis("off")
    plt.title("Three-Legged Cheetah MPC Rollout")

    def animate(i):
      img.set_data(frames[i])
      return [img]

    anim = animation.FuncAnimation(
        fig_anim,
        animate,
        frames=len(frames),
        interval=1000 * model.opt.timestep,
        blit=True,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
  main()
