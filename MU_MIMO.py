import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import erfc


class commModel(object):

    def __init__(self, c, freq, noise_power_dbm, bandwidth, path_loss_exponent, reference_distance, reference_loss_db, n_time_slots, step_size):
        '''
        the constants
        '''
        self.c = c  # light's speed (m/s)
        self.freq = freq # Carrier frequency (Hz)
        self.noise_power_dbm = noise_power_dbm  #  noise power in dBm/Hz
        self.bandwidth = bandwidth  # the total bandwidth (Hz)
        self.sigma2_dbm = noise_power_dbm + 10 * np.log10(bandwidth)  # noise power in dBm
        self.sigma2 = 10 ** (self.sigma2_dbm / 10)  # noise power in watts

        # I considered users have movements, then here we have path loss exponent etc.
        self.path_loss_exponent = path_loss_exponent # 3.5
        self.reference_distance = reference_distance #  1
        self.reference_loss_db = reference_loss_db # 30

        # Time Slot Parameters
        self.n_time_slots = n_time_slots  # thsi is the number of time slots for the simulation
        self.step_size = step_size  # the maximum step size for random walk in meters per time slot (users have random walk)
        '''
        We can define different movement models, but unnecessary
        '''


        # I used existing Q function (please see equation (6) and (7) in the manuscript)
    def Q_function(self, x):
        return 0.5 * erfc(x / np.sqrt(2))


    # Rayleigh fading channel |h_i| in dB (updated based on position of users)
    def path_loss(self, distance):
        return self.reference_loss_db + 10 * self.path_loss_exponent * np.log10(distance / self.reference_distance)


    # data rate for each user when transmission in uplink happens
    '''
    Parameters:
        - p_i_dbm = allocated power to the user (in dBm)
        - h_i_dbm = channel power gain (in dBm)
        - sigma2_dbm = noise power (in dBm)
        - bw = allocated BW (or resource block) to the user (in Hz)
    output is datartae in bits
    '''
    def data_rate(self, p_i_dbm, h_i_dbm, sigma2_dbm, bw):
        snr_linear = 10 ** ((p_i_dbm + h_i_dbm - sigma2_dbm) / 10)
        return bw * np.log2(1 + snr_linear)


    '''
    Here, equation (5) is calculated.
        - N_users = the total number of users
        - energy_per_bit_dbm = this is ξ is manuscript  (in dBm)
        - sigma2_dbm = noise power (in dBm)
        - rho = is the cross correlation between two user's symbols (which has the effect of interference)
    '''
    def snr_min_dB(self, N_users, energy_per_bit_dbm, sigma2_dbm, rho):
        energy_per_bit_linear = 10 ** (energy_per_bit_dbm / 10)  # from dBm to watts
        snr_min_linear = np.zeros(N_users)

        # Calculate SNR_min for each user (equation (5))
        for i in range(N_users):
            interference_sum = np.sum([np.sqrt(energy_per_bit_linear[j]) * rho[j][i] for j in range(N_users) if j != i])
            snr_min_linear[i] = (np.sqrt(energy_per_bit_linear[i]) - interference_sum) ** 2
        snr_min_dB = 10 * np.log10(snr_min_linear)
        return snr_min_dB + sigma2_dbm


    '''
    When we don't have the knowledge of channel, the round robin is of the method for subcarreir allocation
    '''
    # Round Robin subcarrier allocation (equal bw allocation)
    def round_robin_allocation(self, total_bw, N_users):
        bw_per_user = total_bw / N_users
        return np.full(N_users, bw_per_user)


    '''
    As a benchmark, maybe it is good to consider random allocation
    '''
    # random subcarrier allocation
    def random_allocation(self, total_bw, N_users):
        rand_allocation = np.random.rand(N_users)
        rand_allocation /= np.sum(rand_allocation)
        return total_bw * rand_allocation


    # general function to simulate our MU-MIMO uplink system over time slots
    def uplink_simulation(self, N_users, N_antennas, data_sizes, power_dbm, transmit_bw, MOD='QPSK', min_distance=100,
                          max_distance=1000):
        # users' positions are initialized uniformly, then they move during the simulation
        user_positions = np.random.uniform(min_distance, max_distance, (N_users, 2))  # Random initial positions

        MSE_over_time = np.zeros((self.n_time_slots, N_users))

        for t in range(self.n_time_slots):
            # user moevement based on random walk
            user_positions += np.random.uniform(-self.step_size, self.step_size, (N_users, 2))
            distances_from_bs = np.linalg.norm(user_positions, axis=1)  # Euclidean distance from BS
            user_positions = np.clip(user_positions, min_distance, max_distance)

            # new channel gains based on updated positions
            # this is quite simple, we won't need complex channel as we don't care
            h_dbm = -self.path_loss(distances_from_bs)
            '''
            x_dest, y_dest = user_positions
            L = 5.76 * np.log2(sqrt(pow((x - x_dest), 2) + pow((y - y_dest), 2)))
            c = -1 * L / 50
            antenna_gain = 0.9
            s = 0.8
            channel_gains= = pow(10, c) * math.sqrt((antenna_gain*s))
            '''

            # Calculate data rates and energy_per_bit_dbm (ξ) for each user (power in dBm)
            data_rates = np.array([self.data_rate(power_dbm[i], h_dbm[i], self.sigma2_dbm, transmit_bw[i]) for i in range(N_users)])
            energy_per_bit_dbm = power_dbm - 10 * np.log10(data_rates)  # ξ = p_i / r_i in dBm

            # random cross-correlation matrix ρ_jk(0) between users, which has the impact of interference in (5)
            correlation_matrix = np.random.rand(N_users, N_users) * 0.1

            '''
            for k in range(N_users):
                interference_sum_dbm = 10 * np.log10(np.sum(
                    [10 ** (0.1 * (power_dbm[j] + correlation_matrix[j, k])) for j in range(N_users) if j != k]
                ))
    
                min_SNR_db[k] = power_dbm[k] - interference_sum_dbm + sigma2_dbm
    
                # Cap min_SNR_db to a maximum reasonable value (e.g., 100 dB)
                min_SNR_dB_cap = 10
                min_SNR_db[k] = min(min_SNR_db[k], min_SNR_dB_cap)
            '''
            min_SNR_db = self.snr_min_dB(N_users, energy_per_bit_dbm, self.sigma2_dbm, correlation_matrix)

            # calculate the minimum distance between two symbols
            energy_per_bit_watt = 10 ** (energy_per_bit_dbm / 10)
            self.de = []
            if MOD == 'BPSK':
                for k in range(N_users):
                    self.de.append(2 * math.sqrt(energy_per_bit_watt[k]))
            elif MOD == 'QPSK':
                for k in range(N_users):
                    self.de.append(math.sqrt(2 * energy_per_bit_watt[k]))
            elif MOD == '16-QAM':
                for k in range(N_users):
                    self.de.append(2 * math.sqrt(energy_per_bit_watt[k] / 10))
            elif MOD == '64-QAM':
                for k in range(N_users):
                    self.de.append(2 * math.sqrt(energy_per_bit_watt[k] / 42))
            elif MOD == '256-QAM':
                for k in range(N_users):
                    self.de.append(2 * math.sqrt(energy_per_bit_watt[k] / 170))
            else:
                raise ValueError("Error! use 'BPSK' or 'QPSK' or '16-QAM' or '64-QAM' or '256-QAM'")

            # the MSE for each user based on the SNR and Q-function, formula (7) in manuscript
            for i in range(N_users):
                snr_linear = 10 ** (min_SNR_db[i] / 10)
                MSE_over_time[t, i] = self.Q_function(np.sqrt(2 * snr_linear)) * self.de[i]
        # print("de == === ==", self.de)
        # Return MSE for all time slots and users
        return MSE_over_time, user_positions, data_rates, energy_per_bit_dbm, min_SNR_db

    def run_simulation(self, N_users, N_antennas, data_sizes, transmit_power_dbm, allocation_type, MOD):
        if allocation_type == 'round_robin':
            transmit_bw = self.round_robin_allocation(self.bandwidth, N_users)
        elif allocation_type == 'random':
            transmit_bw = self.random_allocation(self.bandwidth, N_users)
        else:
            raise ValueError("Unknown allocation type. Use 'round_robin' or 'random'.")

        MSE_over_time, positions, rates, energy_per_bit_dbm, min_SNR_db = self.uplink_simulation(
            N_users, N_antennas, data_sizes, transmit_power_dbm, transmit_bw, MOD)

        # Calculate average MSE over all time slots for each user
        avg_MSE = np.mean(MSE_over_time, axis=0)

        print("------------------------------------------------")
        for i in range(N_users):
            print(f"User {i + 1}:")
            print(f"  Final Position: {positions[i]}")
            print(f"  Average Data Rate: {rates[i]:.2f} bps")
            print(f"  Average Energy per bit (ξ) (dBm): {energy_per_bit_dbm[i]:.2f} dBm")
            print(f"  Final Min SNR: {min_SNR_db[i]:.2f} dB")
            print(f"  Average MSE: {avg_MSE[i]:.6f}")
            print()
