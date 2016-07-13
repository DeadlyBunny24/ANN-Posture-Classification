import os
import numpy as np
import pandas as pd
from collections import OrderedDict


# This program developes the feature vector used to train our neural network.
# It obtains 10 angles formed by junctions, and associates them to a posture.
#The angles are taken with respect to 3 different basis vectors.

# get coordinate from header labels
def coordinates(header_row):
    coordinates_string = '{x}_x {x}_y {x}_z'.format(x=header_row)
    return coordinates_string.split()

def cross_product_matrix(vector):
    # array must be a R3 vector. 0=x, 1=y, 2=z
    a = vector
    result = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return result

def rotation_matrix_2d(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    matrix = np.array([[cos, -sin],[sin, cos]])
    return matrix

def rotation_matrix(vector, angle):
    # array must be a R3 vector
    constant = 1 / np.linalg.norm(vector)
    unit_vector = constant * vector
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)
    identity_matrix = np.identity(3)
    tensor_dot_vector = np.tensordot(unit_vector, unit_vector, axes=0)
    cpm  = cross_product_matrix(unit_vector)

    # formula components
    angle_cos_id_matrix = angle_cos * identity_matrix
    angle_sin_cpm = angle_sin * cpm
    angle_cos_tensor = (1 - angle_cos) * tensor_dot_vector

    # formula
    result = angle_cos_id_matrix + angle_sin_cpm + angle_cos_tensor

    return result

def angle_between_vector(vector1, vector2):
    # both vectors are R3. Return an angle
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    magnitude = magnitude1 * magnitude2
    dot_product = np.dot(vector1, vector2)
    dpm = dot_product / magnitude
    arc_cos_dpm = np.arccos(dpm)

    return arc_cos_dpm

def rotated_matrix_new(matrix, vector_new):
    # DOT PRODUCT ON ROTATED MATRIX (DPORM) - 2 (ER_WR) R3
    # Rotates matrix over the cross product of the matrix second columnd
    # and vector_new
    vector1 = matrix[0,:]
    vector2 = matrix[1,:]
    vector3 = matrix[2,:]
    rotation_axis = np.cross(vector_new, vector2)
    rotation_angle = angle_between_vector(vector_new, vector2)
    rm = rotation_matrix(rotation_axis, rotation_angle)

    # dporm1 NEW AXIS COORDINATES (NEW PLANE or BASE)
    new_axis_1 = np.dot(rm, vector1)
    new_axis_2 = np.dot(rm, vector2)
    new_axis_3 = np.dot(rm, vector3)

    # new matrix
    new_matrix = np.array([new_axis_1, new_axis_2, new_axis_3])

    return new_matrix

def get_feature_angles(new_vector, matrix):
    # new vector is a vector represented in the coordinates of matrix
    vector1 = matrix[0,:]
    vector2 = matrix[1,:]
    thetha = angle_between_vector(new_vector, vector1)
    new_vector_proj = new_vector * [0,1,1]
    phi = angle_between_vector(new_vector_proj, vector2)

    return (thetha, phi)


# assuming script is run from within project folder
project_dir = os.getcwd()

# complete path of csv directory
csv_dir = project_dir + '/csv/'

# create list of names of csv files in folder
csv_list = os.listdir(csv_dir)

# list of all features found in all files within csv directory
features_list = []

# features file
my_features = open('features.csv', 'w+')

# loop over files in directory
for csv_file in csv_list:
    # assuming we are in bamba_feature_vector directory
    file_path = 'csv/' + csv_file
    df = pd.read_csv(file_path)

    # see headers
    # df.columns.values

    # header combos we need
    headers = ['HPC_SP',
             'SL_SR',
             'SR_ER',
             'ER_WR',
             'SL_EL',
             'EL_WL',
             'HPR_SR',
             'HPR_KR',
             'HPL_SL',
             'HPL_KL',
             'Posture',
             ]

    joint_vectors = OrderedDict()

    # calculate vectors using headers
    for header in headers:
        if header != 'Posture':
            h1, h2 = header.split('_')
            joint1 = coordinates(h1)
            joint2 = coordinates(h2)

            vectors_array = df[joint2].values - df[joint1].values  

            joint_vectors[header] = vectors_array

        else:
            joint_vectors[header] = df[header].values

    # get length of any vector columns as row count
    row_count = len(joint_vectors.values()[0])

    # LOOP THROUGH THE ROWS OF EACH FILE

    num_rows_in_csv = len(joint_vectors['HPC_SP'])

    for row_num in range(num_rows_in_csv):
        HPC_SP = joint_vectors['HPC_SP'][row_num]
        SL_SR = joint_vectors['SL_SR'][row_num]
        SR_ER = joint_vectors['SR_ER'][row_num]
        ER_WR = joint_vectors['ER_WR'][row_num]
        SL_EL = joint_vectors['SL_EL'][row_num]
        EL_WL = joint_vectors['EL_WL'][row_num]
        HPR_SR = joint_vectors['HPR_SR'][row_num]
        HPR_KR = joint_vectors['HPR_KR'][row_num]
        HPL_SL = joint_vectors['HPL_SL'][row_num]
        HPL_KL = joint_vectors['HPL_KL'][row_num]
        Posture = joint_vectors['Posture'][row_num]

        matrix_base = [HPC_SP, SL_SR] # Hipcenter_spine, shoulder_left_and_right
        first_level_vectors = [SR_ER, SL_EL] # shoulder_elbow left and right / vectors for dot product on matrix
        second_level_vectors = [ER_WR, EL_WL] # elbow_wrist left and right / vectors for dot product on rotated matrix

        vector1 = HPC_SP
        vector2 = SL_SR
        vector3 = np.cross(vector1, vector2)

        # MATRIX
        matrix = np.array([vector1, vector2, vector3])

        # MATRIX INVERSE
        matrixinv = np.linalg.inv(matrix)

        # DOT PRODUCT ON MATRIX (DPOM) - 1 (SR_ER) R3
        new_vector_SR_ER = np.dot(matrixinv.transpose(), SR_ER) # SR_ER in new base (matrix is the base)

        # DOT PRODUCT ON MATRIX (DPOM) - 2 (SL_EL) R3
        new_vector_SL_EL = np.dot(matrixinv.transpose(), SL_EL) # SL_EL in new base (matrix is the base)

        # DOT PRODUCT ON ROTATED MATRICES
        new_base_ER_WR = rotated_matrix_new(matrix, new_vector_SR_ER)
        new_base_EL_WL = rotated_matrix_new(matrix, new_vector_SL_EL)

        # NEW INVERSE MATRICES OF NEW BASES
        new_base_ER_WR_inv = np.linalg.inv(new_base_ER_WR)
        new_base_EL_WL_inv = np.linalg.inv(new_base_EL_WL)

        # NEW VECTORS ER_WR and EL_WL
        new_vector_ER_WR = np.dot(new_base_ER_WR_inv.transpose(), ER_WR)
        new_vector_EL_WL = np.dot(new_base_EL_WL_inv.transpose(), EL_WL)

        # represent all vectors in angles thetha and phi
        SR_ER_theta, SR_ER_phi = get_feature_angles(new_vector_SR_ER, matrix)
        SL_EL_theta, SL_EL_phi = get_feature_angles(new_vector_SL_EL, matrix)
        ER_WR_theta, ER_WR_phi = get_feature_angles(new_vector_ER_WR, new_base_ER_WR)
        EL_WL_theta, EL_WL_phi = get_feature_angles(new_vector_EL_WL, new_base_EL_WL)

        # get angles between leg vectors
        SR_HPR_KR_angle = angle_between_vector(HPR_SR, HPR_KR)
        SL_HPL_KL_angle = angle_between_vector(HPL_SL, HPL_KL)

        # create list with resulting features from row and append them to features_list
        export_row_data = [SR_ER_theta, SR_ER_phi, SL_EL_theta, SL_EL_phi, ER_WR_theta,
        ER_WR_phi, EL_WL_theta, EL_WL_phi, SR_HPR_KR_angle, SL_HPL_KL_angle, Posture]

        features_list.append(export_row_data)

        # line to write in file
        row_text = ",".join([str(feature) for feature in export_row_data]) + '\n'

        # write line to file
        my_features.write(row_text)

my_features.close()



# {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
