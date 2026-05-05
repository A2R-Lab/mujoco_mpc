// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/quadruped_ant/quadruped_ant.h"

#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string QuadrupedAnt::XmlPath() const {
  return GetModelPath("quadruped_ant/task.xml");
}

std::string QuadrupedAnt::Name() const { return "Quadruped Ant Walk"; }

void QuadrupedAnt::ResidualFn::Residual(const mjModel* model,
                                        const mjData* data,
                                        double* residual) const {
  int counter = 0;

  // ----- control ----- //
  mju_copy(residual + counter, data->ctrl, model->nu);
  counter += model->nu;

  // ----- upright ----- //
  const double* torso_up = SensorByName(model, data, "torso_zaxis");
  residual[counter++] = 1.0 - torso_up[2];

  // ----- walk speed ----- //
  const double* torso_velocity = SensorByName(model, data, "velocimeter");
  residual[counter++] = parameters_[0] - torso_velocity[0];

  // ----- non-forward torso velocity ----- //
  residual[counter++] = torso_velocity[1];
  residual[counter++] = torso_velocity[2];

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // ----- posture ----- //
  mju_copy(residual + counter, data->qpos + 7, model->nq - 7);
  counter += model->nq - 7;

  CheckSensorDim(model, counter);
}

}  // namespace mjpc
