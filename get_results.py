import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your .npy file
file_path = 'PQWGAN/wasserstein_distance.npy'

# Load the .npy file
data = np.load(file_path)


# Simple line plot for 1D data
plt.plot(data)
plt.xlabel("Batches")
plt.ylabel("Wasserstein Distance")
plt.show()


# Display the contents (optional)
print(data)
