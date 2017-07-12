/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//extract standard deviations from std[] array
	double std_x = std[0];
	double std_y = std[1];
	double std_yaw = std[2];

	//define normal distributions for x,y and yaw
	normal_distribution<double> gauss_x(x, std_x);
	normal_distribution<double> gauss_y(y, std_y);
	normal_distribution<double> gauss_yaw(theta, std_yaw);

	//define a random number generator, that will be used to take random values from the gaussian distribution
	default_random_engine rand;

	//set number of particles
	num_particles = 10;

	cout << "particles initialization:\n";
	//Initialize position for all particles
	//Initialize also all particle weights to 1, so that they all have the same probability
	for (int i = 0; i < num_particles; i++) {
		//initialize i-th particle by extracting a random value from a gaussian fitted with the GPS coordinates received as initialization
		Particle particle;
		particle.id = i;
		particle.x = gauss_x(rand);
		particle.y = gauss_y(rand);
		particle.theta = gauss_yaw(rand);
		particle.weight = 1;

		//add particle to "particles" array and its weight to "weights" array (the 2 arrays will always have the same size)
		particles.push_back(particle);
		weights.push_back(1);

		/*cout << "particle " << i << ":\n";
		cout << "x: " << particle.x << ", y: " << particle.y << ", th: " << particle.theta << "w: " << particle.weight << endl;
		cout << "weight " << i << ": " << weights[i] << "\n";*/
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//define a random number generator, that will be used to take random values from the gaussian distribution
	default_random_engine rand;

	//cout << "prediction step\n";

	//extract standard deviations from std_pos[] array
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_yaw = std_pos[2];

	double v_yawrate = velocity / yaw_rate;
	double v_dt = velocity * delta_t;
	double yawrate_dt = yaw_rate * delta_t;

	//perform the prediction step for all the particles in the filter
	for (int i = 0; i < num_particles; i++) {
		//we will use different equations here, based on the value of yaw_rate
		//(if yaw_rate is 0, we have a straight movement, otherwise we are doing a turn)
		if (yaw_rate == 0) {
			particles[i].x += v_dt * cos(particles[i].theta);
			particles[i].y += v_dt * sin(particles[i].theta);
		}
		else {
			particles[i].x += v_yawrate * (sin(particles[i].theta + yawrate_dt) - sin(particles[i].theta));
			particles[i].y += v_yawrate * (-cos(particles[i].theta + yawrate_dt) + cos(particles[i].theta));
			particles[i].theta += yawrate_dt;
		}

		//define normal distributions for x,y and yaw
		normal_distribution<double> gauss_x(particles[i].x, std_x);
		normal_distribution<double> gauss_y(particles[i].y, std_y);
		normal_distribution<double> gauss_yaw(particles[i].theta, std_yaw);

		//add gaussian noise to the predicted value for this particle
		particles[i].x = gauss_x(rand);
		particles[i].y = gauss_y(rand);
		particles[i].theta = gauss_yaw(rand);

		//cout << "particle " << i << ":\n";
		//cout << "x: " << particles[i].x << ", y: " << particles[i].y << ", th: " << particles[i].theta << "w: " << particles[i].weight << endl;
		//cout << "weight " << i << ": " << weights[i] << "\n";
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//cout << "data association:\n";

	//loop through all the observations
	for (int i = 0; i < observations.size(); i++) {
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		//for each observation, loop through all the predicted measurements to find the one nearest to the current observation and set its id into observation id
		int nearestPredID = 0;
		double nearestPredX = 0;
		double nearestPredY = 0;
		double nearestDistance = -1;
		for (int k = 0; k < predicted.size(); k++) {
			double pred_x = predicted[k].x;
			double pred_y = predicted[k].y;
			double pred_id = predicted[k].id;

			double distance = sqrt(pow(pred_x - obs_x, 2) + pow(pred_y - obs_y, 2));
			if (distance < nearestDistance || nearestDistance == -1) {
				//update the nearest predicted landmark id and distance
				nearestDistance = distance;
				nearestPredID = pred_id;
				nearestPredX = pred_x;
				nearestPredY = pred_y;
			}
		}

		//associate the nearest predicted landrmark id to the current observation
		observations[i].id = nearestPredID;
		//observations[i].x = nearestPredX;
		//observations[i].y = nearestPredY;

		//cout << "observed landmark " << i << " (" << obs_x << "," << obs_y << "), nearest landmark ID: " << nearestPredID << " (" << nearestPredX << ", " << nearestPredY << ")\n";
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//define a random number generator, that will be used to take random values from the gaussian distribution
	default_random_engine rand;


	double totalWeight = 0;
	//for each particle, predict measurements to all map landmarks within sensor range
	for (int i = 0; i < num_particles; i++) {
		double part_x = particles[i].x;
		double part_y = particles[i].y;
		double part_yaw = particles[i].theta;
		//precalculate sin and cos values to speed up further calculations (avoid to calculate it for every observation in the loop below)
		double sin_yaw = sin(part_yaw);
		double cos_yaw = cos(part_yaw);

		//cout << "update weights:\n";
		vector<LandmarkObs> observationsMap;
		observationsMap.clear();
		//cout << "observations: " << endl;
		for (int k = 0; k < observations.size(); k++) {
			LandmarkObs landmarkObs;
			//convert observation data from vechicle coordinates to map coordinates
			landmarkObs.x = part_x * cos_yaw - part_y * sin_yaw + observations[k].x;
			landmarkObs.y = part_x * sin_yaw + part_y * cos_yaw + observations[k].y;

			//add the observed positions just converted in map coordinates to the "observationsMap" vector
			observationsMap.push_back(landmarkObs);

			//cout << "obs " << k << ": x=" << landmarkObs.x << "; y=" << landmarkObs.y << endl;
		}

		vector<LandmarkObs> surroundingLandmarks;
		surroundingLandmarks.clear();
		//cout << "landmarks: " << endl;
		for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
			//add to "surroundingLandmarks" only landmarks which distance from the current position is closer than "sensor_range"
			double landmarkDist = sqrt(pow(map_landmarks.landmark_list[k].x_f - part_x, 2) + pow(map_landmarks.landmark_list[k].y_f - part_y, 2));
			if (landmarkDist <= sensor_range) {
				LandmarkObs landmarkObs;

				landmarkObs.id = map_landmarks.landmark_list[k].id_i;
				landmarkObs.x = map_landmarks.landmark_list[k].x_f;
				landmarkObs.y = map_landmarks.landmark_list[k].y_f;

				surroundingLandmarks.push_back(landmarkObs);
				//cout << "landmark " << k << ": x=" << landmarkObs.x << "; y=" << landmarkObs.y << endl;
			}
		}

		//associate landmarks to observations
		dataAssociation(surroundingLandmarks, observationsMap);

		//calculate particle weight using a multi-variate Gaussian distribution
		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];
		double particleWeight = 1;
		//precalculate these values to speed up the execution (avoid calculating them for every observation in the loop below)
		double basePart = 1 / (2 * M_PI * sigma_x * sigma_y);
		double sigma_2x2 = 2 * pow(sigma_x, 2);
		double sigma_2y2 = 2 * pow(sigma_y, 2);
		//double sigma_2x2 = 2 / pow(sigma_x, 2);
		//double sigma_2y2 = 2 / pow(sigma_y, 2);

		for (int k = 0; k < observations.size(); k++) {
			double landmarkID = observationsMap[k].id;
			//TODO: come posizione x e y del landmark devo prendere il valore contenuto nell'osservazione o quello nel landmark che ci ho associato? (fare una prova con entrambi, ma probabilmente la risposta corretta è la seconda!)
			double landmarkObs_x = observationsMap[k].x;
			double landmarkObs_y = observationsMap[k].y;
			double landmark_x = map_landmarks.landmark_list[landmarkID - 1].x_f;
			double landmark_y = map_landmarks.landmark_list[landmarkID - 1].y_f;

			//TODO: QUESTO VALORE SEMBRA SBAGLIATO... (viene un valore ~E-118 solo per i primi 2 landmarks, dopodichè sempre 0... questo fa si che il total weight sia 0 e da luogo ad una divisione by 0)
			//double observationWeight = basePart * exp(-(pow(part_x - landmarkObs_x, 2) / sigma_2x2 + pow(part_y - landmarkObs_y, 2) / sigma_2y2));
			double observationWeight = basePart * exp(-(pow(landmark_x - landmarkObs_x, 2) / sigma_2x2 + pow(landmark_y - landmarkObs_y, 2) / sigma_2y2));
			//double observationWeight = basePart * exp(-(pow(part_x - landmarkObs_x, 2) / 2 * pow(sigma_x, 2) + pow(part_y - landmarkObs_y, 2) / 2 * pow(sigma_y, 2)));

			//cout << "basePart: " << basePart << endl;
			//cout << "landmark ID: " << observationsMap[k].id << endl;
			//cout << "landmark mapped id: " << map_landmarks.landmark_list[observationsMap[k].id - 1].id_i << endl;
			//cout << "landmark X: " << map_landmarks.landmark_list[observationsMap[k].id - 1].x_f << endl;
			//cout << "landmark y: " << map_landmarks.landmark_list[observationsMap[k].id - 1].y_f << endl;
			//cout << "obs x: " << observationsMap[k].x << endl;
			//cout << "obs y: " << observationsMap[k].y << endl;
			//cout << "x_land: " << map_landmarks.landmark_list[observationsMap[k].id - 1].x_f << endl;
			//cout << "y_land: " << map_landmarks.landmark_list[observationsMap[k].id - 1].y_f << endl;
			/*cout << "xDiff: " << landmark_x - landmarkObs_x << "(" << landmark_x << " - " << landmarkObs_x << ")" << endl;
			cout << "yDiff: " << landmark_y - landmarkObs_y << "(" << landmark_y << " - " << landmarkObs_y << ")" << endl;*/
			//cout << "exp power: " << -(pow(observations[k].x - landmarkObs_x, 2) / sigma_2x2 + pow(observations[k].y - landmarkObs_y, 2) / sigma_2y2) << endl;
			//cout << "observationWeight: " << observationWeight << endl;

			particleWeight = particleWeight * observationWeight;
		}

		cout << "particle " << i << "(" << part_x << "," << part_y << ") weight: " << particleWeight << endl;

		//set the particle weight
		particles[i].weight = particleWeight;
		weights[i] = particleWeight;
		totalWeight += particleWeight;
	}

	//cout << "total weight: " << totalWeight << endl;
	//normalize the weight of each particle
	for (int i = 0; i < num_particles; i++) {
		double particleWeight = particles[i].weight / totalWeight;
		particles[i].weight = particleWeight;
		weights[i] = particleWeight;
		cout << "particle " << i << " normalized weight: " << particleWeight << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampledParticles;
	vector<double> newWeights;
	/*for (int i = 0; i < num_particles; i++) {
		Particle particle = particles[i];
		double particleWeight = weights[i];

		


		resampledParticles.push_back();
	}*/
	//random_device rd;
	//mt19937 gen(rd());
	//discrete_distribution<> d(weights);
	//for (int n = 0; n<num_particles; ++n) {
	//	++m[d(gen)];
	//}
	
	double maxWeight = 0;
	for (int i = 0; i < weights.size(); i++) {
		if (weights[i] > maxWeight)
			maxWeight = weights[i];
	}

	double wMax2 = maxWeight * 2.0;
	double beta = 0.0;

	//cout << "wMax2: " << wMax2 << endl;
	//int wIndex = random.randint(0, N - 1);
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> d(0, num_particles - 1);
	int wIndex = d(gen);
	//cout << "wIndex: " << wIndex << endl;
	for (int i = 0; i < num_particles; i++) {
		beta = beta + wMax2;
		//cout << "beta: " << beta << endl;
		//cout << "wIndex: " << wIndex << endl;
		while (weights[wIndex] < beta) {
			beta = beta - weights[wIndex];
			wIndex = (wIndex + 1) % num_particles;
			//cout << "beta: " << beta << endl;
			//cout << "wIndex: " << wIndex << endl;
		}
		//cout << "weight at wIndex: " << weights[wIndex] << endl;
		newWeights.push_back(weights[wIndex]);
		resampledParticles.push_back(particles[wIndex]);
		//particles[i].weight = weights[wIndex];
	}
	
	weights = newWeights;
	particles = resampledParticles;

	cout << "particles post update:\n";
	for (int i = 0; i < num_particles; i++) {
		cout << "particle " << i << ":\n";
		cout << "x: " << particles[i].x << ", y: " << particles[i].y << ", th: " << particles[i].theta << "w: " << particles[i].weight << endl;
		cout << "weight " << i << ": " << weights[i] << "\n";
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
