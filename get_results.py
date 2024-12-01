import numpy as np
import matplotlib.pyplot as plt



# Specify the path to your .npy file
file_path = 'PQWGAN/results_noise_new_param/013_14p_17l_25bs/wasserstein_distance.npy'

# Load the .npy file
data = np.load(file_path)


# Simple line plot for 1D data
plt.plot(data)
plt.xlabel("Batches")
plt.ylabel("Wasserstein Distance")
plt.show()


# Display the contents (optional)
print(data)
