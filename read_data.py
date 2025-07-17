#%%
import numpy as np
from matplotlib import pyplot as plt


test_number = 0 # between 0 and 89

data = np.load('training-set/%03d.npz' % test_number)

X = data['DIC_X']
disp = data['DIC_disp']
instron_disp = data['instron_disp']
instron_force = data['instron_force']
label = data['label']

# Calculate displacement magnitude at the last time step
disp_mag = np.linalg.norm(disp[-1], axis=-1)

#%%

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.pcolor(label, cmap='gray')
plt.title('Label')
plt.colorbar(label='material class')

plt.subplot(1, 3, 2)
plt.title('Displacement Magnitude')
plt.pcolor(X[...,0], X[...,1], disp_mag, cmap='jet')
plt.colorbar(label='Displacement Magnitude [mm]')

plt.subplot(1, 3, 3)
plt.title('Instron Force')
plt.plot(instron_disp, instron_force)
plt.xlabel('Instron Displacement [mm]')
plt.ylabel('Instron Force [N]')
plt.xlim([0,instron_disp.max()])
plt.ylim([0,instron_force.max()])

plt.tight_layout()
plt.savefig('images/test_%03d.png' % test_number, dpi=300)

# %%
