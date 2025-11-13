# %%
import os
import numpy as np
import json

import matplotlib.pyplot as plt
from skimage.transform import resize


# Base folder paths
base_folder_path = 'mechanical_testing'
images_folder = os.path.join(base_folder_path, '000_images')
instron_folder = os.path.join(base_folder_path, 'instron_data')

flips = np.genfromtxt('data/mnist_debug_up-down.csv', dtype=int, skip_header=1, delimiter=',')

failed = {}

#%%

# Iterate over all cases from 000 to 099
for case in range(100):
    try:
        case_str = f"{case:03d}"
        label = np.load(f"data/label_{case_str}.npz")['label']

        # Skip processing if the output file already exists
#        if os.path.exists(os.path.join(base_folder_path, f"{case_str}.npz")):
#            print(f"Case {case_str} already processed. Skipping.")
#            continue
        folder_path = images_folder.replace('000', case_str)
        instron_file = os.path.join(instron_folder, f"{case_str}.csv")
        output_file = os.path.join(base_folder_path, f"{case_str}.npz")

        # Get list of .csv files in alphabetical order
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

        # Read each .csv file into a numpy array and store in a list
        numpy_arrays = np.array([np.genfromtxt(os.path.join(folder_path, file), delimiter=';', skip_header=True) for file in csv_files])

        # Extract the last 5 digits from each file name and store them in a list
        if case == -10-10-10-10-10-10-10-10-1000:
            # Special case for 14
            time_stamps = np.array([int(file[-8:-4]) for file in csv_files])
        else:
            time_stamps = np.array([int(file[-9:-4]) for file in csv_files])

        DIC_times = (time_stamps - time_stamps[0]) / 1000  # seconds

        # the x and y are shared for all files
        Nx = Ny = np.sqrt(numpy_arrays.shape[1])
        print(case,numpy_arrays.shape[1], Nx, Ny, DIC_times.shape)
        if Nx % 1 != 0:
            if numpy_arrays.shape[1] == 50*51:
                Nx = 50; Ny = 51
            elif numpy_arrays.shape[1] == 48*49:
                Nx = 48; Ny = 49
            #raise ValueError("The number of columns in the numpy arrays is not a perfect square.")
        Nx = int(Nx); Ny = int(Ny)
        X = numpy_arrays[0, :, :2].reshape((Nx, Ny, 2))

        diff = X.max(0).max(0) - X.min(0).min(0)

        print((X[-2,-1,:] -X[0,0,:])/diff)

        disp = numpy_arrays[:, :, 2:4].reshape((numpy_arrays.shape[0], Nx, Ny, 2))

        # Read Instron data
        instron_data = np.genfromtxt(
            instron_file,
            delimiter=',',
            skip_header=2,
            dtype=float,
            converters={
                0: lambda s: float(s.strip(b'"').strip('"')) if isinstance(s, bytes) else float(s.strip('"')),
                1: lambda s: float(s.strip(b'"').strip('"')) if isinstance(s, bytes) else float(s.strip('"')),
                2: lambda s: float(s.strip(b'"').strip('"')) if isinstance(s, bytes) else float(s.strip('"'))
            }
        )

        # Interpolate Instron data
        interpF = np.interp(DIC_times, instron_data[:, 0], instron_data[:, 2])
        interp_disp = np.interp(DIC_times, instron_data[:, 0], instron_data[:, 1])

        # conversion factor from pixels to mm
        # Assuming the last displacement is the maximum displacement in pixels
        max_disp_DIC =np.abs(disp[...,1]).max()
        pix2mm = (interp_disp[-1] - interp_disp[0])/max_disp_DIC
        print('pix2mm',pix2mm, interp_disp[-1],max_disp_DIC)
        # Resize label to match the dimensions of disp

        Nmax = max(Nx, Ny)
        label_resized = resize(
            label,
            (Nmax, Nmax),
            order=0,  # Nearest neighbor interpolation
            mode='reflect',
            anti_aliasing=False,
            preserve_range=True
        ).astype(int)[:Nx, :Ny]
        label_resized = np.fliplr(label_resized)  # Flip horizontally to match DIC coordinates
        if flips[case, 1] == 1:
            label_resized = np.flipud(label_resized)

        # Save data to .npz file
        np.savez(output_file, DIC_disp=disp*pix2mm, DIC_X=X*pix2mm, instron_disp=interp_disp, instron_force=interpF, label=label_resized)
        # Compute displacement magnitude at the last time step
        disp_magnitude = np.linalg.norm(disp[-1], axis=-1)
        # Compute strain components
        dudx = np.diff(disp[-1,..., 0], axis=0) 
        dvdy = np.diff(disp[-1,..., 1], axis=1)
        dudy = np.diff(disp[-1,..., 0], axis=1)
        dvdx = np.diff(disp[-1,..., 1], axis=0)
        
        # Pad arrays to match dimensions
        dudx = dudx[:,:-1]
        dvdy = dvdy[:-1,:]
        dudy = dudy[:-1,:]
        dvdx = dvdx[:,:-1]
        
        # Compute strain tensor components
        exx = dudx
        eyy = dvdy
        exy = 0.5 * (dudy + dvdx)
        
        # Compute principal strains
        avg = 0.5 * (exx + eyy)
        diff = np.sqrt(0.25 * (exx - eyy)**2 + exy**2)
        e1 = avg + diff  # Maximum principal strain
        e2 = avg - diff  # Minimum principal strain

        print(X.shape, dvdy.shape, diff.shape)


        # Plot and save the displacement magnitude image
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(label_resized, cmap='viridis', vmin=0, vmax=9, origin='lower')
        plt.colorbar(label='Label Value')
        plt.subplot(1, 3, 2)
        plt.pcolor(X[:-1,:-1,0], X[:-1,:-1,1], dvdy,cmap='viridis' )
        plt.colorbar(label='strain y')
        plt.title(f'strain y at Last Time Step (Case {case_str})')
        plt.subplot(1, 3, 3)
        plt.pcolor(X[:-1,:-1,0], X[:-1,:-1,1], diff, cmap='viridis')
        plt.colorbar(label='MPS')
        plt.title(f'max shear strain Last Time Step (Case {case_str})')
        plt.tight_layout()
        plt.savefig(f"mechanical_testing/debug_images/{case_str}_disp_magnitude.png")
        plt.close()
        print(f"Processed case {case_str}")
    except Exception as e:
        print(f"Error processing case {case_str}: {e}")

        print('appended')
        failed[case_str] = str(e)

print("Failed cases (pretty):")
print(json.dumps(failed, indent=4))
print("Failed cases:", failed.keys())





# %%
