###### Scrip to take read and write mda and a folder and
### save all of the dicom files within it as an mda file


import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from numpy.lib.type_check import real
from numpy.matrixlib.defmatrix import matrix

# from show_3d_images import show_3d_images
import sys
import pydicom


def dicom_to_mda(parser):
    folder = args.folder
    print("INPUT FOLDER IS " + str(folder) + "\n\n")

    if folder == "cwd" or folder == "pwd":
        folder = os.getcwd()

    error_finding_files = False

    print(folder)

    if args.regex:

        regex_to_files = os.path.join(folder, args.regex)
        print(regex_to_files)
        files_in_folder = glob.glob(regex_to_files)
        print(files_in_folder)

        if len(files_in_folder) == 0:
            error_finding_files = True
            print("ERROR IN args.regex")

    if not args.regex or error_finding_files:
        print("ERROR IN args.regex. Trying normal dicom extensions . . . ")

        regex_to_files = os.path.join(folder, "*.dcm")
        files_in_folder = glob.glob(regex_to_files)

        if len(files_in_folder) == 0:
            print("\n\nError in *.dcm. Trying *.DCM")

            regex_to_files = os.path.join(folder, "*.DCM")
            files_in_folder = glob.glob(regex_to_files)

        if len(files_in_folder) == 0:
            print("\n\nError in *.DCM. Trying *.ima")

            regex_to_files = os.path.join(folder, "*.ima")
            files_in_folder = glob.glob(regex_to_files)

        if len(files_in_folder) == 0:
            print("\n\nError in *.ima. Trying *.IMA")

            regex_to_files = os.path.join(folder, "*.IMA")
            files_in_folder = glob.glob(regex_to_files)

        if len(files_in_folder) == 0:
            error_message = "\n\n COULD NOT FIND FILES FROM " + folder
            if args.regex:
                error_message = error_message + "" + args.regex
            raise ValueError(error_message)

    num_files = len(files_in_folder)

    print("\n\n FOUND TOTAL " + str(num_files) + " NUMBER FILES . . . ")

    instance_number_pointer = (0x0020, 0x0013)

    # image_hdr = pydicom.dcmread(files_in_folder[0])
    # print(image_hdr)
    # print(image_hdr[instance_number_pointer])
    # print(aa)
    # print('\n\n\n')
    # image_hdr2 = pydicom.dcmread(files_in_folder[3])
    # print(image_hdr2)
    # print(image_hdr2[0x0020,0x0013].value)

    tmp_image_hdr = pydicom.dcmread(files_in_folder[0])
    tmp_image = tmp_image_hdr.pixel_array

    (n_rows, n_cols) = tmp_image.shape

    image_dtype = tmp_image.dtype

    # print(n_rows)
    # print(n_cols)
    # print(num_files)

    slice_order = np.zeros(num_files).astype(np.int16)

    for counter in range(num_files):
        tmp_image_hdr = pydicom.dcmread(files_in_folder[counter])
        tmp_image_index = tmp_image_hdr[instance_number_pointer].value
        slice_order[counter] = tmp_image_index

    output_matrix = np.zeros((n_rows, n_cols, num_files), dtype=image_dtype)

    print("READING IN FILES . . . \n\n\n")

    for counter in range(num_files):
        tmp_image_hdr = pydicom.dcmread(files_in_folder[slice_order[counter] - 1])

        output_matrix[:, :, counter] = tmp_image_hdr.pixel_array

    # show_3d_images(output_matrix)
    basepath = os.path.dirname(files_in_folder[0])

    print("WRITING TO MDA NOW . . . \n\n\n")
    output_file_name = os.path.join(basepath, "dataFile.mda")
    writemda(fname=output_file_name, mat=output_matrix)

    return output_file_name


def readmda(fname, folder=os.getcwd(), dtype=np.double):
    file_name = os.path.join(folder, fname)
    # print('\n\n\n\n READING MDA FILE \n\n'+ file_name+'\n\n')

    # First open up binary file and grab the
    # data type code contained within the header
    fid = open(file_name, "rb")
    tmp = np.fromfile(fid, np.intc)
    fid.close()

    # This corresponds to 4 for double, single,
    # complex, etc. and to -4 for uint8, int16, int64, etc.
    dtype_code = tmp[0]

    # The dtype code and the matrix dim size are 4 bytes each
    header_bit_depth = 4

    if dtype_code > 0:

        num_matrix_dims = dtype_code
        matrix_dims = tmp[1 : (1 + num_matrix_dims)]
        total_num_elements = np.prod(matrix_dims)
        dtype_code = -1
        bit_offset = (1 + num_matrix_dims) * header_bit_depth
        # bit offset is the number of bits (bytes?) that
        # one skips from header before reading the actual data

    else:
        num_matrix_dims = tmp[2]
        matrix_dims = tmp[3 : (3 + num_matrix_dims)]
        total_num_elements = np.prod(matrix_dims)
        bit_offset = (3 + num_matrix_dims) * header_bit_depth

    # print('MATRIX DIMENSIONS')
    # print(matrix_dims)
    # print('\n\n\n')

    if dtype_code == -1:
        # print('\n\n READING FILE NOW . . . \n\n')

        fid = open(file_name, "rb")
        # This is where we skip the initial header info
        fid.seek(bit_offset, os.SEEK_SET)

        data_stream = np.fromfile(fid, np.float32)
        fid.close()
        length_data_stream = data_stream.size

        inds_to_keep = np.arange(2 * total_num_elements)
        data_stream = data_stream[inds_to_keep]

        real_part = (
            data_stream[0:length_data_stream:2].copy().reshape(matrix_dims, order="F")
        )
        imag_part = (
            data_stream[1:length_data_stream:2].copy().reshape(matrix_dims, order="F")
        )

        if np.count_nonzero(imag_part.flatten()) > 0:
            dtype = np.complex
            raw_data = real_part + 1j * imag_part
        else:
            raw_data = real_part.astype(dtype=dtype)

    elif dtype_code == -4:
        # print('\n\n READING FILE NOW . . . \n\n')
        fid = open(file_name, "rb")
        fid.seek(bit_offset, os.SEEK_SET)

        data_stream = np.fromfile(fid, np.int16)
        fid.close()

        length_data_stream = data_stream.size

        raw_data = data_stream.copy().reshape(matrix_dims, order="F")

    return raw_data


def writemda(fname, mat):
    # print('\n\n\n\n WRITING MDA NOW .... \n\n\n')

    fid = open(fname, "wb")

    is_int = np.issubdtype(mat.dtype, np.integer)
    mat_size = mat.shape
    num_dims = len(mat_size)

    total_num_elements = np.prod(mat.shape)
    # print('TOTAL ELEMENTS')
    # print(total_num_elements)
    # print('MATRIX SHAPE')
    # print(mat.shape)
    # print('\n\n\n')

    if is_int:

        mat = mat.flatten(order="F")

        # print('IS INT')
        fid.write((-4).to_bytes(4, byteorder=sys.byteorder, signed=1))
        fid.write((2).to_bytes(4, byteorder=sys.byteorder, signed=1))
        fid.write(num_dims.to_bytes(4, byteorder=sys.byteorder, signed=1))
        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4, byteorder=sys.byteorder, signed=1))

        for iter_element in range(total_num_elements):
            fid.write(
                mat[iter_element]
                .tolist()
                .to_bytes(2, byteorder=sys.byteorder, signed=1)
            )

    else:

        # print('ISNT INT')
        fid.write(num_dims.to_bytes(4, byteorder=sys.byteorder, signed=1))

        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4, byteorder=sys.byteorder, signed=1))

        flatten_order = "F"
        mat_real = np.real(mat).flatten(order=flatten_order).astype(np.single)
        mat_imag = np.imag(mat).flatten(order=flatten_order).astype(np.single)

        for iter_element in range(total_num_elements):
            fid.write(mat_real[iter_element].tobytes())
            fid.write(mat_imag[iter_element].tobytes())

    fid.close()

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, required=False)
    parser.add_argument("--regex", type=str, required=False)

    args = parser.parse_args()

    output_file_name = dicom_to_mda(args)

    # tmp = readmda(output_file_name)

    # show_3d_images(tmp)
