from Matrix import Matrix
from Vector import Vector
from SingularValueDecomposition import SingularValueDecomposition
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.image import imread
from savingmatrices import store_matrices_to_file


def predict_digit(digit_matrix):
    residuals = []
    for residual_matrix in matrices_for_residual:
        residual = calculate_residual(digit_matrix, residual_matrix)
        residuals.append(residual)

    minimum_residual = min(residuals)
    digit = residuals.index(minimum_residual)

    return digit


def calculate_residual(digit_matrix, residual_matrix):
    residual_by_digit = Matrix.multiply(residual_matrix, digit_matrix)
    residual_by_digit_for_eigen_value = Matrix.multiply_left_by_transpose(residual_by_digit)

    eigen_vector = Vector.get_greatest_eigen_vector(residual_by_digit_for_eigen_value)
    eigen_value = Vector.get_eigen_value(eigen_vector, residual_by_digit_for_eigen_value)

    singular_value_matrix_norm = eigen_value ** (1/2)

    return singular_value_matrix_norm


def scale_processed_digit_to_size(digit_matrix):
    digit_matrix.strip()

    size_to_fit_to = 20

    greatest_dimension = max(digit_matrix.rows, digit_matrix.columns)
    difference_above_28 = greatest_dimension % size_to_fit_to
    difference_to_add = size_to_fit_to - difference_above_28
    new_dimension = greatest_dimension + difference_to_add

    padding_for_rows = new_dimension - digit_matrix.rows
    padding_for_columns = new_dimension - digit_matrix.columns
    digit_matrix.pad(padding_for_rows, padding_for_columns)

    resized_digit = Matrix.resize(digit_matrix, size_to_fit_to)

    total_rows_columns_to_add = 28 - size_to_fit_to
    x_centre, y_centre = resized_digit.calculate_centre_of_mass()

    centre = 13.5
    rows_for_top = round(centre - y_centre)
    rows_for_bottom = total_rows_columns_to_add - rows_for_top

    columns_for_left = round(centre - x_centre)
    columns_for_right = total_rows_columns_to_add - columns_for_left

    # print(x_centre, columns_for_left, columns_for_right, y_centre, rows_for_top, rows_for_bottom)
    resized_digit.pad_edges(rows_for_top, rows_for_bottom, columns_for_left, columns_for_right)

    return resized_digit


def get_processed_digit(file_name):
    hand_written_digit = imread(file_name)

    hand_written_digit_grey = np.mean(hand_written_digit, -1)

    grey_digit_array = hand_written_digit_grey.tolist()
    grey_digit_matrix = Matrix.create_matrix_from_data(grey_digit_array)

    grey_digit_matrix.make_integer()

    grey_digit_matrix.invert()

    grey_digit_matrix.black_or_white(200)

    grey_digit_matrix = Matrix.filter(grey_digit_matrix)

    resized_digit = scale_processed_digit_to_size(grey_digit_matrix)

    return resized_digit


def show_digit(matrix, title=''):
    image = plt.imshow(np.array(matrix.data))
    image.set_cmap('gray')
    plt.axis('off')
    plt.title(title)
    plt.show()


def predict_digit_matrix(matrix):
    digit_numpy_array = np.array(matrix.data)
    digit_flat = digit_numpy_array.flatten()
    flat_digit_array = digit_flat.tolist()
    current_digit_matrix_transposed = Matrix.create_matrix_from_data([flat_digit_array])
    current_digit_matrix = Matrix.transpose(current_digit_matrix_transposed)

    prediction = predict_digit(current_digit_matrix)
    return prediction



def get_next_digit(matrix):
    columns_of_next_digit = matrix.get_full_left_columns()
    # print(columns_of_next_digit, 'columns of first digit')
    digit_matrix = Matrix.get_first_columns(matrix, columns_of_next_digit)
    # show_digit(digit_matrix)
    empty_columns_after_digit = matrix.get_empty_columns_after_column(columns_of_next_digit)
    # print(empty_columns_after_digit, 'empty columns after first digit')
    if empty_columns_after_digit == 0:
        remaining_digits = None
    else:
        remaining_digits = Matrix.remove_left_columns(matrix, columns_of_next_digit + empty_columns_after_digit)
    return digit_matrix, remaining_digits


def seperate_into_digits(multiple_digit_matrix):
    digits = []
    while True:
        next_digit, multiple_digit_matrix = get_next_digit(multiple_digit_matrix)
        digits.append(next_digit)
        if not multiple_digit_matrix:
            break
    # digits.append(multiple_digit_matrix)
    return digits



def get_next_row(matrix):
    rows_of_next_row = matrix.get_full_top_rows()
    # print(rows_of_next_row, 'rows of first row')
    row_matrix = Matrix.get_first_rows(matrix, rows_of_next_row)
    # show_digit(digit_matrix)
    empty_rows_after_row = matrix.get_empty_rows_after_row(rows_of_next_row)
    # print(empty_rows_after_row, 'empty rows after first row')
    if empty_rows_after_row == 0:
        remaining_rows = None
    else:
        remaining_rows = Matrix.remove_bottom_rows(matrix, rows_of_next_row + empty_rows_after_row)
        remaining_rows.strip()
    row_matrix.strip()
    return row_matrix, remaining_rows


def seperate_into_rows(digit_page_matrix):
    rows = []
    while True:
        next_row, digit_page_matrix = get_next_row(digit_page_matrix)
        rows.append(next_row)
        if not digit_page_matrix:
            break
    return rows




def get_digits_from_image(file_name):
    digit_page = imread(file_name)

    digit_page_grey = np.mean(digit_page, -1)
    digit_page_array = digit_page_grey.tolist()
    digit_page_matrix = Matrix.create_matrix_from_data(digit_page_array)

    digit_page_matrix.make_integer()
    digit_page_matrix.invert()
    digit_page_matrix.black_or_white(175)
    multiple_digit_matrix = Matrix.filter(digit_page_matrix)

    multiple_digit_matrix.strip()

    show_digit(multiple_digit_matrix)

    seperate_rows = seperate_into_rows(multiple_digit_matrix)
    for row in seperate_rows:
        show_digit(row)

    seperate_digits = []
    for row in seperate_rows:
        seperate_digits_of_row = seperate_into_digits(row)

        for digit in seperate_digits_of_row:
            seperate_digits.append(digit)
    return seperate_digits

seperate_digits = []
files_of_handwriting = ['louietrainingimages/symboldigits.jpg']
for file in files_of_handwriting:
    digits_of_file = get_digits_from_image(file)
    seperate_digits = seperate_digits + digits_of_file



print('total digits', len(seperate_digits))


matrices_for_digits = [[], [], [], []]

for digit in seperate_digits:
    processed_digit = scale_processed_digit_to_size(digit)

    show_digit(processed_digit)

    correct_digit = input('what digit')

    if correct_digit == ' ':
        continue
    elif correct_digit == '+':
        matrices_for_digits[0].append(processed_digit)
    elif correct_digit == '-':
        matrices_for_digits[1].append(processed_digit)
    elif correct_digit == 'x':
        matrices_for_digits[2].append(processed_digit)
    elif correct_digit == '/':
        matrices_for_digits[3].append(processed_digit)

for m in matrices_for_digits:
    print(len(m))


training_matrices_for_handwriting = []

for i in range(4):
    training_digits = matrices_for_digits[i]

    training_images_for_digit_columns = [Matrix.flatten_along_rows(digit) for digit in training_digits]

    training_matrix_for_digit_transposed = Matrix(len(training_digits), 28 ** 2)

    training_matrix_for_digit_transposed.data = [flat_image.data[0] for flat_image in training_images_for_digit_columns]

    training_matrix_for_digit = Matrix.transpose(training_matrix_for_digit_transposed)

    training_matrices_for_handwriting.append(training_matrix_for_digit)

print('completed creating digit matrices for handwritten digits')
store_matrices_to_file(training_matrices_for_handwriting, 'louietrainingimages/symboldigitmatrices.txt')


#
# svd_rank = 10
#
# training_matrices_of_digits_svd = []
# for digit_training_matrix in training_matrices_for_handwriting:
#     digit_training_matrix_svd = SingularValueDecomposition(digit_training_matrix, svd_rank)
#     training_matrices_of_digits_svd.append(digit_training_matrix_svd)
#     print('completed svd of digit matrix for handwriting')
#
# matrices_for_residual = []
# for digit_svd in training_matrices_of_digits_svd:
#     u_matrix_transposed = Matrix.transpose(digit_svd.u)
#     u_u_transposed = Matrix.multiply_left_by_transpose(u_matrix_transposed)
#
#     identity_matrix = Matrix.identity(u_u_transposed.columns)
#     matrix_for_residual_for_digit = Matrix.subtract(identity_matrix, u_u_transposed)
#     matrices_for_residual.append(matrix_for_residual_for_digit)
#     print('completed residual matrix for handwritten digit')
#
# store_matrices_to_file(matrices_for_residual, 'louietrainingimages/louieresiduals.txt')




















