from Matrix import Matrix
from Vector import Vector
from SingularValueDecomposition import SingularValueDecomposition
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.image import imread


def loadMNIST(prefix):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(prefix + '-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

    labels = np.fromfile(prefix + '-labels-idx1-ubyte', dtype='ubyte')[2 * intType.itemsize:]

    return data, labels


trainingImages, trainingLabels = loadMNIST("train")
testImages, testLabels = loadMNIST("t10k")

training_matrices_of_digits = []
limit_of_images_in_matrix = 100
for i in range(10):
    training_images_for_digit = [trainingImages[index] for index, label in enumerate(trainingLabels) if label == i]
    training_images_for_digit = training_images_for_digit[:limit_of_images_in_matrix]

    training_images_for_digit_columns = [numpy_array.flatten() for numpy_array in training_images_for_digit]

    training_matrix_for_digit_transposed = Matrix(len(training_images_for_digit), len(training_images_for_digit_columns[0]))

    training_matrix_for_digit_transposed.data = [flat_image.tolist() for flat_image in training_images_for_digit_columns]

    training_matrix_for_digit = Matrix.transpose(training_matrix_for_digit_transposed)

    training_matrices_of_digits.append(training_matrix_for_digit)

    # print(training_matrix_for_digit)
    # print(training_matrix_for_digit.rows, training_matrix_for_digit.columns)

print('completed creating digit matrices')

svd_rank = 10

training_matrices_of_digits_svd = []
for digit_training_matrix in training_matrices_of_digits:
    digit_training_matrix_svd = SingularValueDecomposition(digit_training_matrix, svd_rank)
    training_matrices_of_digits_svd.append(digit_training_matrix_svd)
    print('completed svd of digit matrix')

matrices_for_residual = []
for digit_svd in training_matrices_of_digits_svd:
    u_matrix_transposed = Matrix.transpose(digit_svd.u)
    u_u_transposed = Matrix.multiply_left_by_transpose(u_matrix_transposed)

    identity_matrix = Matrix.identity(u_u_transposed.columns)
    matrix_for_residual_for_digit = Matrix.subtract(identity_matrix, u_u_transposed)
    matrices_for_residual.append(matrix_for_residual_for_digit)
    print('completed residual matrix for digit')


def predict_digit(digit_matrix):
    residuals = []
    for residual_matrix in matrices_for_residual:
        residual = calculate_residual(digit_matrix, residual_matrix)
        residuals.append(residual)

    # plt.plot(residuals)
    # plt.xticks(range(10))
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

    print(x_centre, columns_for_left, columns_for_right, y_centre, rows_for_top, rows_for_bottom)
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
processed_three = get_processed_digit('three.jpg')
show_digit(processed_three, 'Pre processed 3')
print('three', predict_digit_matrix(processed_three))
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
    print(columns_of_next_digit, 'columns of first digit')
    digit_matrix = Matrix.get_first_columns(matrix, columns_of_next_digit)
    # show_digit(digit_matrix)
    empty_columns_after_digit = matrix.get_empty_columns_after_column(columns_of_next_digit)
    print(empty_columns_after_digit, 'empty columns after first digit')
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




# multiple_digit = imread('28457.jpg')
# multiple_digit = imread('bossnumbers.jpg')
# multiple_digit = imread('largenumbers.jpg')
# multiple_digit = imread('libbynumbers.jpg')
# multiple_digit = imread('newnumbers.jpg')
multiple_digit = imread('smallnumbers.jpg')


multiple_digit_grey = np.mean(multiple_digit, -1)
multiple_digit_array = multiple_digit_grey.tolist()
multiple_digit_matrix = Matrix.create_matrix_from_data(multiple_digit_array)

multiple_digit_matrix.make_integer()
multiple_digit_matrix.invert()
multiple_digit_matrix.black_or_white(200)
multiple_digit_matrix = Matrix.filter(multiple_digit_matrix)

multiple_digit_matrix.strip()

show_digit(multiple_digit_matrix)

seperate_digits = seperate_into_digits(multiple_digit_matrix)

for index, digit in enumerate(seperate_digits):
    # title = 'Seperated digit of matrix, ' + str(index)
    # show_digit(digit, title)

    processed_digit = scale_processed_digit_to_size(digit)
    prediction = predict_digit_matrix(processed_digit)
    # print('Predicted digit at index', index, 'as a', str(prediction))

    title = 'Seperated digit of matrix scaled, predicted as ' + str(prediction)
    show_digit(processed_digit, title)








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
