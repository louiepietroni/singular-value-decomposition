from Matrix import Matrix
from Vector import Vector
from SingularValueDecomposition import SingularValueDecomposition
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.image import imread
from savingmatrices import get_matrices_from_file
from Evaluate import evaluate_expression


matrices_for_residual = get_matrices_from_file('louietrainingimages/louiefullsymbolresiduals.txt')



def predict_digit(digit_matrix):
    residuals = []
    for residual_matrix in matrices_for_residual:
        residual = calculate_residual(digit_matrix, residual_matrix)
        residuals.append(residual)

    # plt.plot(residuals)
    # plt.xticks(range(17))
    # plt.show()

    minimum_residual = min(residuals)
    digit = residuals.index(minimum_residual)

    return digit


def calculate_residual(digit_matrix, residual_matrix):

    # print('residual matrix', residual_matrix.rows, residual_matrix.columns)
    # print('digital matrix', digit_matrix.rows, digit_matrix.columns)

    residual_by_digit = Matrix.multiply(residual_matrix, digit_matrix)
    residual_by_digit_for_eigen_value = Matrix.multiply_left_by_transpose(residual_by_digit)

    eigen_vector = Vector.get_greatest_eigen_vector(residual_by_digit_for_eigen_value)
    eigen_value = Vector.get_eigen_value(eigen_vector, residual_by_digit_for_eigen_value)

    singular_value_matrix_norm = eigen_value ** (1/2)

    return singular_value_matrix_norm


# total_correct = 0
# total = 200
# start_index = 1500
# for i in range(start_index, start_index + total):
#     print(i)
#     current_digit = testImages[i]
#     current_label = testLabels[i]
#
#     current_digit_flat = current_digit.flatten()
#     current_digit_array = current_digit_flat.tolist()
#     current_digit_matrix_transposed = Matrix.create_matrix_from_data([current_digit_array])
#     current_digit_matrix = Matrix.transpose(current_digit_matrix_transposed)
#
#     prediction = predict_digit(current_digit_matrix)
#     if prediction == current_label:
#         total_correct += 1
#     else:
#         image = plt.imshow(current_digit)
#         image.set_cmap('gray')
#         plt.axis('off')
#         plt.title('Digit:  ' + str(current_label) + ', Prediction:  ' + str(prediction))
#         plt.show()
#
# percent_correct = round((total_correct / total) * 100, 2)
#
# print(str(percent_correct) + ' with ' + str(limit_of_images_in_matrix) + ' images from each digit and rank of ' + str(svd_rank))



# while True:
#     index = random.randrange(10000)
#
#     current_digit = testImages[index]
#     current_label = testLabels[index]
#
#     current_digit_flat = current_digit.flatten()
#     current_digit_array = current_digit_flat.tolist()
#     current_digit_matrix_transposed = Matrix.create_matrix_from_data([current_digit_array])
#     current_digit_matrix = Matrix.transpose(current_digit_matrix_transposed)
#
#     prediction = predict_digit(current_digit_matrix)
#
#     image = plt.imshow(current_digit)
#     image.set_cmap('gray')
#     plt.axis('off')
#     plt.title('Digit:  ' + str(current_label) + ', Prediction:  ' + str(prediction))
#     plt.show()
#     a = input('Predict next digit')

def scale_processed_digit_to_size(digit_matrix):
    # show_digit(digit_matrix)

    digit_matrix.strip()

    # show_digit(digit_matrix)

    size_to_fit_to = 20

    greatest_dimension = max(digit_matrix.rows, digit_matrix.columns)
    difference_above_28 = greatest_dimension % size_to_fit_to
    difference_to_add = size_to_fit_to - difference_above_28
    new_dimension = greatest_dimension + difference_to_add

    padding_for_rows = new_dimension - digit_matrix.rows
    padding_for_columns = new_dimension - digit_matrix.columns
    digit_matrix.pad(padding_for_rows, padding_for_columns)
    # print('padded')

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

    # show_digit(resized_digit)

    return resized_digit


def get_processed_digit(file_name):
    hand_written_digit = imread(file_name)

    hand_written_digit_grey = np.mean(hand_written_digit, -1)

    grey_digit_array = hand_written_digit_grey.tolist()
    grey_digit_matrix = Matrix.create_matrix_from_data(grey_digit_array)

    # show_digit(grey_digit_matrix)

    grey_digit_matrix.make_integer()
    # show_digit(grey_digit_matrix)

    # print('made integer')
    grey_digit_matrix.invert()
    # show_digit(grey_digit_matrix)

    # show_digit(grey_digit_matrix)
    # print('inverted')
    grey_digit_matrix.black_or_white(200)
    # print('two colour')
    grey_digit_matrix = Matrix.filter(grey_digit_matrix)
    # print('filtered')



    # grey_digit_matrix.strip()
    # print('stripped')



    # size_to_fit_to = 20
    #
    # greatest_dimension = max(grey_digit_matrix.rows, grey_digit_matrix.columns)
    # difference_above_28 = greatest_dimension % size_to_fit_to
    # difference_to_add = size_to_fit_to - difference_above_28
    # new_dimension = greatest_dimension + difference_to_add
    #
    # padding_for_rows = new_dimension - grey_digit_matrix.rows
    # padding_for_columns = new_dimension - grey_digit_matrix.columns
    # grey_digit_matrix.pad(padding_for_rows, padding_for_columns)
    # # print('padded')
    #
    # resized_digit = Matrix.resize(grey_digit_matrix, size_to_fit_to)
    #
    # total_rows_columns_to_add = 28 - size_to_fit_to
    # x_centre, y_centre = resized_digit.calculate_centre_of_mass()
    #
    # centre = 13.5
    # rows_for_top = round(centre - y_centre)
    # rows_for_bottom = total_rows_columns_to_add - rows_for_top
    #
    # columns_for_left = round(centre - x_centre)
    # columns_for_right = total_rows_columns_to_add - columns_for_left
    #
    # print(x_centre, columns_for_left, columns_for_right, y_centre, rows_for_top, rows_for_bottom)
    # resized_digit.pad_edges(rows_for_top, rows_for_bottom, columns_for_left, columns_for_right)

    resized_digit = scale_processed_digit_to_size(grey_digit_matrix)

    # padding_for_each_edge = (28 - size_to_fit_to) / 2
    # left_padding = padding_for_each_edge - x_offset
    # resized_digit.pad(28 - size_to_fit_to, 28 - size_to_fit_to)
    # print('resized')
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


# processed_one = get_processed_digit('one.jpg')
# show_digit(processed_one, 'Pre processed 1')
# print('one', predict_digit_matrix(processed_one))
#
# processed_two = get_processed_digit('two.jpg')
# show_digit(processed_two, 'Pre processed 2')
# print('two', predict_digit_matrix(processed_two))
#
# processed_three = get_processed_digit('three.jpg')
# show_digit(processed_three, 'Pre processed 3')
# print('three', predict_digit_matrix(processed_three))
#
# processed_four = get_processed_digit('four.jpg')
# show_digit(processed_four, 'Pre processed 4')
# print('four', predict_digit_matrix(processed_four))
#
# processed_five = get_processed_digit('five.jpg')
# show_digit(processed_five, 'Pre processed 5')
# print('five', predict_digit_matrix(processed_five))
#
# processed_six = get_processed_digit('six.jpg')
# show_digit(processed_six, 'Pre processed 6')
# print('six', predict_digit_matrix(processed_six))
#
# processed_seven = get_processed_digit('seven.jpg')
# show_digit(processed_seven, 'Pre processed 7')
# print('seven', predict_digit_matrix(processed_seven))
#
# processed_eight = get_processed_digit('eight.jpg')
# show_digit(processed_eight, 'Pre processed 8')
# print('eight', predict_digit_matrix(processed_eight))


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


seperate_digits = get_digits_from_image('expression3.jpg')


# multiple_digit = imread('28457.jpg')
# multiple_digit = imread('bossnumbers.jpg')
# multiple_digit = imread('largenumbers.jpg')
# multiple_digit = imread('libbynumbers.jpg')
# multiple_digit = imread('newnumbers.jpg')
# multiple_digit = imread('expression2.jpg')


# multiple_digit_grey = np.mean(multiple_digit, -1)
# multiple_digit_array = multiple_digit_grey.tolist()
# multiple_digit_matrix = Matrix.create_matrix_from_data(multiple_digit_array)
#
# multiple_digit_matrix.make_integer()
# multiple_digit_matrix.invert()
# multiple_digit_matrix.black_or_white(175)
# multiple_digit_matrix = Matrix.filter(multiple_digit_matrix)
#
# multiple_digit_matrix.strip()
#
# show_digit(multiple_digit_matrix)
#
# seperate_digits = seperate_into_digits(multiple_digit_matrix)

expression_text = ''

print('Starting predicting digits now with my matrix library')

for index, digit in enumerate(seperate_digits):
    # title = 'Seperated digit of matrix, ' + str(index)
    # show_digit(digit, title)

    processed_digit = scale_processed_digit_to_size(digit)
    prediction = predict_digit_matrix(processed_digit)

    if prediction == 10:
        prediction = '+'
    elif prediction == 11:
        prediction = '-'
    elif prediction == 12:
        prediction = 'x'
    elif prediction == 13:
        prediction = '/'
    elif prediction == 14:
        prediction = '('
    elif prediction == 15:
        prediction = ')'
    elif prediction == 16:
        prediction = '='

    title = 'Seperated digit of matrix scaled, predicted as ' + str(prediction)
    # show_digit(processed_digit, title)

    if prediction == '=':
        result = evaluate_expression(expression_text)
        if int(result) == result:
            result = int(result)
        print(prediction + str(result))
        expression_text = ''
    else:
        expression_text += str(prediction)
        print(str(prediction), end='')

# result = evaluate_expression(expression_text[:-1])
# print(expression_text + str(result))






# while True:
#     index = random.randrange(10000)
#
#     current_digit = trainingImages[index]
#     current_label = trainingLabels[index]
#
#     current_digit_array = current_digit.tolist()
#     current_digit_matrix = Matrix(28, 28)
#     current_digit_matrix.data = current_digit_array
#
#     prediction = predict_digit_matrix(current_digit_matrix)
#     title = 'Digit:  ' + str(current_label) + ', Prediction:  ' + str(prediction)
#     show_digit(current_digit, title)
#     a = input('Predict next digit')



# print(trainingImages)
