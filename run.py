
import numpy as np
from MU_MIMO import commModel


c = 3e8  # light speed (m/s)
freq = 2.4e9  # carrier freq. (Hz)
noise_power_dbm = -174  # Noise power in dBm/Hz
bandwidth = 10e6  # BW (Hz)
path_loss_exponent = 3.5  # Path loss exponent
reference_distance = 1  # Reference distance in meters
reference_loss_db = 30  # Reference loss in dB at the reference distance
n_time_slots = 100000  # Number of time slots for simulation
step_size = 5  # Step size for random walk (meters)


model = commModel(
    c=c,
    freq=freq,
    noise_power_dbm=noise_power_dbm,
    bandwidth=bandwidth,
    path_loss_exponent=path_loss_exponent,
    reference_distance=reference_distance,
    reference_loss_db=reference_loss_db,
    n_time_slots=n_time_slots,
    step_size=step_size
)


N_users = 8
N_antennas = 8  # Number of antennas at the BS
data_sizes = np.random.randint(300e6, 350e6, N_users)  # Random data sizes for each user (bits)
transmit_power_dbm = np.random.uniform(20, 21, N_users)  # Random transmit power per user (dBm)
allocation_type = 'round_robin'  # Allocation method ('round_robin' or 'random')
MOD = '64-QAM'  # Modulation type ('BPSK', 'QPSK', '16-QAM', '64-QAM', '256-QAM')


model.run_simulation(
    N_users=N_users,
    N_antennas=N_antennas,
    data_sizes=data_sizes,
    transmit_power_dbm=transmit_power_dbm,
    allocation_type=allocation_type,
    MOD=MOD
)
