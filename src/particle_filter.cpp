/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles_ = 25;

  double weight = 1 / num_particles_;
  particles_.reserve(num_particles_);

  for (int ii = 0; ii < num_particles_; ii++) {
    particles_.emplace_back(ii, dist_x(gen_), dist_y(gen_), dist_theta(gen_),
                            weight);
  }
  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  for (Particle &p : particles_) {
    const double theta_dot = yaw_rate;
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    // Calculate positions with guassiann noise.
    const double noisy_theta = dist_theta(gen_);
    const double noisy_x = dist_x(gen_);
    const double noisy_y = dist_y(gen_);

    if (std::abs(theta_dot) < 0.00001) {
      p.x = noisy_x + velocity * cos(noisy_theta) * delta_t;
      p.y = noisy_y + velocity * sin(noisy_theta) * delta_t;
    } else {
      const double v_over_theta_dot = velocity / theta_dot;
      const double theta_dot_by_delta_t = theta_dot * delta_t;

      p.x = noisy_x +
            v_over_theta_dot *
                (sin(noisy_theta + theta_dot_by_delta_t) - sin(noisy_theta));
      p.y = noisy_y +
            v_over_theta_dot *
                (cos(noisy_theta) - cos(noisy_theta + theta_dot_by_delta_t));
      p.theta = noisy_theta + theta_dot_by_delta_t;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  const double normalizer = 1 / (M_2_PI * std_landmark[0] * std_landmark[1]);
  const double x_denom = 2 * std_landmark[0] * std_landmark[0];
  const double y_denom = 2 * std_landmark[1] * std_landmark[1];

  for (Particle &p : particles_) {
    p.weight = 1.0;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    for (const LandmarkObs &lobs : observations) {
      int nearest_landmark_id = -1;
      double x_m = std::nan("xm");
      double y_m = std::nan("ym");
      vehicle_to_map(p.x, p.y, p.theta, lobs.x, lobs.y, &x_m, &y_m);
      // find the closest landmark location.
      double nearest_distance = std::numeric_limits<double>::infinity();
      double nearest_lm_x = std::numeric_limits<double>::infinity();
      double nearest_lm_y = std::numeric_limits<double>::infinity();
      for (const auto &lm : map_landmarks.landmark_list) {
        double distance = dist(x_m, y_m, lm.x_f, lm.y_f);
        if (distance < sensor_range && distance < nearest_distance) {
          nearest_distance = distance;
          nearest_landmark_id = lm.id_i;
          nearest_lm_x = lm.x_f;
          nearest_lm_y = lm.y_f;
        }
      }

      // At this point we have the lobs associated to the nearest map point.
      // Add debug data point.
      p.associations.push_back(nearest_landmark_id);
      // WFT co-ordinates are these in ? I'll assume car? but if map coords then
      // use nearestXXX above.
      p.sense_x.push_back(x_m);
      p.sense_y.push_back(y_m);

      // Now for this observation calculate the multivariate probability.
      double diff_x = x_m - nearest_lm_x;
      double diff_y = y_m - nearest_lm_y;
      double exponent =
          -1 * (((diff_x * diff_x) / x_denom) + ((diff_y * diff_y) / y_denom));
      double prob = normalizer * exp(exponent);
      p.weight *= prob;
    }
  }
}

void ParticleFilter::resample() {
  std::vector<double> weights;
  weights.reserve(particles_.size());
  for (const Particle &p : particles_) {
    weights.push_back(p.weight);
  }
  std::vector<Particle> originals(particles_);

  std::discrete_distribution<double> d(weights.begin(), weights.end());
  for (int ii = 0; ii < particles_.size(); ii++) {
    particles_[ii] = originals[d(gen_)];
  }
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
