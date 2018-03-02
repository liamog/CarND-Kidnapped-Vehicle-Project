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
#include <numeric>
#include <limits>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles_ = 25;

  double weight = 1 / num_particles_;
  particles_.reserve(num_particles_);

  for (int ii = 0; ii < num_particles_; ii++) {
    particles_.emplace_back(ii, dist_x(gen), dist_y(gen), dist_theta(gen),
                           weight);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  default_random_engine gen;
  // TODO: do I need to deal with a yaw_rate of 0?
  for (Particle &p : particles_) {
    const double theta_dot = yaw_rate;
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    const double v_over_theta_dot = velocity / theta_dot;
    const double theta_dot_by_delta_t = theta_dot * delta_t;

    p.x = dist_x(gen) + v_over_theta_dot * (sin(p.theta + theta_dot_by_delta_t) - sin(p.theta));
    p.y = dist_y(gen) + v_over_theta_dot * (cos(p.theta) - cos(p.theta + theta_dot_by_delta_t));

    p.theta = p.theta + theta_dot_by_delta_t;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems. Keep in mind that this transformation requires
  //   both rotation AND translation (but no scaling). The following is a good
  //   resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html
  const double normalizer = 1 / (M_2_PI * std_landmark[0] * std_landmark[1]);
  const double x_denom = 2 * std_landmark[0] * std_landmark[0];
  const double y_denom = 2 * std_landmark[1] * std_landmark[1];

  for (Particle &p : particles_) {
    p.weight = 1.0;
    for (LandmarkObs lobs : observations) {
      lobs.id = -1;
      double x_m = std::nan("xm");
      double y_m = std::nan("ym");
      vehicle_to_map(p.x, p.y, p.theta, lobs.x, lobs.y, &x_m, &y_m);
      // find the closest landmark location.
      double nearest_distance = std::numeric_limits<double>::infinity();
      double nearest_lm_x = std::numeric_limits<double>::infinity();
      double nearest_lm_y = std::numeric_limits<double>::infinity();
      for (const auto &lm: map_landmarks.landmark_list) {
        double distance = dist(x_m, y_m, lm.x_f, lm.y_f);
        if (distance < sensor_range && distance < nearest_distance) {
          nearest_distance = distance;
          lobs.id = lm.id_i;
          nearest_lm_x = lm.x_f;
          nearest_lm_y = lm.y_f;
        }
      }
      // At this point we have the lobs associated to the nearest map point.
      p.associations.push_back(lobs.id);
      // WFT co-ordinates are these in ? I'll assume car? but if map coords then use nearestXXX above.
      p.sense_x.push_back(x_m);
      p.sense_y.push_back(y_m);

      // Now for this observation calculate the multivariate probability.

      double diff_x = x_m - nearest_lm_x;
      double diff_y = y_m - nearest_lm_y;
      double exponent = -1 * ( ((diff_x * diff_x) / x_denom) + ((diff_y * diff_y) / y_denom) );
      double prob = normalizer * exp(exponent);
      p.weight *= prob;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight. NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::vector<double> weights;
  weights.reserve(particles_.size());
  for (const Particle &p : particles_) {
    weights.push_back(p.weight);
  }
  std::vector<Particle> originals(particles_);

  std::discrete_distribution<double > d(weights.begin(), weights.end());
  for (int ii = 0; ii < originals.size(); ii++) {
    particles_[ii] = originals[d(gen)];
  }
}

void ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
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
