# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import os

import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import multivariate_normal


#%% generate_map function
def generate_map(locs, x_max=64, y_max=64, sigma=1):
    """Generates a density map from an array of fluorophore positions

    Parameters
    ----
    locs : numpy array of double
        N-by-2 ndarray containing the horizontal and vertical positions of the
        fluorophores in the frame, in pixels (but with subpixel precision)
    x_max : int
        Number of horizontal pixels of the density map (default 64)
    y_max : int
        Number of vertical pixels of the density map (default 64)

    Returns
    ----
    genMap : numpy array of double
        x_max-by-y_max array containing the values for each pixel of the
        density map
    """

    # meshgrid
    x, y = np.meshgrid(range(x_max), range(y_max))
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    grid = np.hstack((x,y))

    # create a map
    genMap = np.zeros((grid.shape[0],1))

    for k in range(locs.shape[0]):
            # Sum the mvn at each position for each pixel
            mvn = multivariate_normal.pdf(grid + 0.5, locs[k,:], cov=sigma)
            genMap += mvn.reshape(-1,1)
    genMap = genMap.reshape((x_max, y_max))

    return genMap

#%% generate_stack function
def gen_map_stack(frame_gt,
                  n_frames,
                  x_max, y_max,
                  threshold = 250,
                  sigma=1,
                  output_file=None,
                  dataset='data'):
    """Generates a HDF5 file with the density maps given in inputFile

    Parameters
    ----------
    frame_gt : str or pandas.DataFrame
            CSV file or pandas.DataFrame containing the ground truth positions,
            with the first column being the frame, and the second and third
            columns the fluorophore positions, in pixels (with sub-pixel
            precision)
    outputFile : str
        Name of the output HDF5 file containing the density maps. It
        contains a unique dataset named 'data', of shape
        (nFrames, x_max, y_max)
    nFrames : int
        The total number of frames in the stack
    x_max : int
        Number of horizontal pixels of the density map (default 64)
    y_max : int
        Number of vertical pixels of the density map (default 64)
    threshold : float (0<x<1)
        Minimal brightness to register on the image, in photons
    """

    if isinstance(frame_gt, str):
        ground_truth = pd.read_csv(frame_gt)
    elif isinstance(frame_gt, pd.DataFrame):
        ground_truth = frame_gt
    else:
        class GroundTruthFormatError(Exception):
            pass
        raise GroundTruthFormatError
        ('The ground truth must be either a csv file or a pandas.DataFrame')

    # Dont take into account the fluorophores with weak signals.
    print('Removing low-brightness molecules')

    # TODO Add unit test for this.
    try:
        ground_truth = ground_truth[ground_truth['brightness'] > threshold]
        print('Done.')
    except KeyError:
        print('No column named "brightness" detected. No thresholding on '
              'fluorophore brightness will be performed.')

    # Group by frame
    idx = ground_truth.groupby('frame')

    # Parse all the frames
    density_stack = np.zeros((n_frames, x_max, y_max))
    for index, f in idx:
        locs = f[['x', 'y']].values
        genMap = generate_map(locs, x_max, y_max, sigma=sigma)

        # The frames are in 1-indexing (ImageJ)
        numFrame = int(f.iloc[0]['frame'])
        density_stack[numFrame-1,:,:] = genMap

        if numFrame % 500 == 0:
            print('Frame {0}/{1}...' .format(numFrame, n_frames))
    print('Map generation complete.')

    # Export to HDF5
    if output_file is not None:
        h5File = h5py.File(output_file, 'a')
        h5File.create_dataset(dataset, data=density_stack)
        h5File.close()
    return density_stack

#%% Generate a tiff stack from the density HDF5 file
def h5_to_tiff(inputFile, dataset, outputFile=None):
    """Generate a tiff stack from a HDF5 dataset

    """
    inputName = os.path.splitext(inputFile)[0]
    if outputFile is None:
        outputFile = inputName + '.tif'
    file = h5py.File(inputFile, 'r')
    dMap = file[dataset]
    dMap = np.array(dMap)
    dMap *=10000
    dMap = dMap.astype('uint16')
    tifffile.imsave(outputFile, dMap)
    file.close()

def array_to_tiff(arr, outputTiff):
    """Transform a density map array into a tiff image, to be visualized."""
    arr = np.array(arr)
    arr *=10000
    arr = arr.astype('uint16')
    arr[arr > 60000] = 0
    tifffile.imsave(outputTiff, arr)

#%% frame ground truth (gt) from state logger
# Deprecated
def get_frame_gt(position_logger, state_logger, output_file=None):
    """Given the position logger and the state logger of a SASS simulation,
    builds a ['frame', 'time_on', 'x', 'y', 'id'] pandas DataFrame
    giving for each frame the positions and on time of fluorophores that are
    in a visible state

    """

    print('Extracting the ground truth positions per frame from the state logger...')
    def df_frame_time(on_frame, off_frame, on_time, off_time):
        # A column with all the frames during which the fluo is on
        out_row = pd.DataFrame(np.arange(on_frame, off_frame+1), columns=['frame'])

        # A column with the time the fluo is on for each frame
        if out_row.shape[0] == 1:
            time_on = off_time - on_time
        else:
            time_on = np.ones(out_row.shape)
            time_on[0] = on_frame+1 - on_time
            time_on[-1] = off_time - off_frame
        out_row['time_on'] = time_on
        out_row['time_on'].astype(float)
        return out_row

    # Load the state logger
    logger = pd.read_csv(state_logger)
    # Load the positions
    positions = pd.read_csv(position_logger)
    positions['id'] = positions['id'].astype('int64')

    # Some global variables
    on_state = 0
    nFrames = np.ceil(logger.time_elapsed.max())

    # Add a column with the frame number
    logger['frame'] = np.floor(logger['time_elapsed'])
    logger.frame = logger.frame.astype(int)
    # Sort by time elapsed and group by fluorophore
    id_index = logger.groupby(['id'])
    # Create the output empty data frame
    frameGT = pd.DataFrame(columns = ['frame', 'time_on', 'x', 'y',
                                      'z', 'id'])

    # Parse for each frame
    for fluoID, m in id_index:
        # Check whether molecule starts in the ON state
        if m.iloc[0]['initial_state'] == on_state:
            on_frame = 0
            on_time = 0.0

        for index, row in m.iterrows():
            if row['next_state'] == on_state:
                on_frame = row['frame']
                on_time = row['time_elapsed']

            if row['initial_state'] == on_state:
                off_frame = row['frame']
                off_time = row['time_elapsed']

                #print(positions['id']==fluoID)
                out_df = df_frame_time(on_frame, off_frame, on_time, off_time)
                out_df['x'] = positions.loc[positions['id']==fluoID].x.values[0]
                out_df['y'] = positions.loc[positions['id']==fluoID].y.values[0]
                out_df['z'] = positions.loc[positions['id']==fluoID].z.values[0]
                out_df['id'] = fluoID

                # Append to the output dataframe
                frameGT = frameGT.append(out_df)

        # Check if some molecules are on in the end
        if m.iloc[-1]['next_state'] == on_state:
            off_frame = nFrames-1
            off_time = float(nFrames)

            out_df = df_frame_time(on_frame, off_frame, on_time, off_time)
            out_df['x'] = positions.loc[positions['id']==fluoID].x.values[0]
            out_df['y'] = positions.loc[positions['id']==fluoID].y.values[0]
            out_df['z'] = positions.loc[positions['id']==fluoID].z.values[0]
            out_df['id'] = fluoID
            # Append to the output dataframe
            frameGT = frameGT.append(out_df)
        # Timer
        if fluoID % 1000 == 0:
            print('Fluorophore %d / %d...' % (fluoID, positions.shape[0]))

    # Convert the frame column to integers
    frameGT['frame'] = frameGT['frame'].astype(int)

    # If a fluo is activated twice during the same frame, the time_on must be
    # added: for a same frame-ID pair, add the times, and take the min
    # of the positions (they should be the same anyway)
    aggFunc = {'time_on': 'sum', 'x': 'min', 'y' : 'min', 'z' : 'min'}
    # Perform the aggregation on every frame-ID pair
    frameGT = frameGT.groupby(['frame', 'id']).agg(aggFunc)
    # Store the frame and ID as columns and not as index
    frameGT = frameGT.reset_index()
    frameGT['brightness'] = frameGT['time_on']*2500.0

    if output_file is not None:
        # Save the dataframe to a csv file
        csv_file = output_file
        frameGT.to_csv(csv_file, index=False)
        print("File saved at '{0}'" .format(csv_file))
    return frameGT


