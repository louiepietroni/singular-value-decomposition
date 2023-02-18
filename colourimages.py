from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from Matrix import Matrix
from Vector import Vector


def get_eigen_vectors(original_matrix, matrix_rank):
    matrix = Matrix(original_matrix.rows, original_matrix.columns)
    matrix.data = [[item for item in row] for row in original_matrix.data]

    eigen_vectors = []
    eigen_values = []
    for i in range(matrix_rank):
        print(i)
        greatest_remaining_eigen_vector = Vector.get_greatest_eigen_vector(matrix)
        greatest_remaining_eigen_value = Vector.get_eigen_value(greatest_remaining_eigen_vector, matrix)
        if isinstance((greatest_remaining_eigen_value ** (1/2)), complex):
            print(greatest_remaining_eigen_value ** (1/2), greatest_remaining_eigen_value, 'complex')

        singular_value = greatest_remaining_eigen_value ** (1/2)
        # print(greatest_remaining_eigen_vector.data, greatest_remaining_eigen_value, singular_value)

        current_eigen_vector = Vector.create_vector_from_data(greatest_remaining_eigen_vector.data)

        if isinstance((greatest_remaining_eigen_value ** (1/2)), complex):
            eigen_values.append(1)
        else:
            eigen_values.append(greatest_remaining_eigen_value)

        eigen_vectors.append(current_eigen_vector)
        # eigen_values.append(greatest_remaining_eigen_value)

        scalar_for_matrix = greatest_remaining_eigen_value / greatest_remaining_eigen_vector.get_magnitude_squared()
        matrix_for_greatest_eigen_value = Matrix.multiply_right_vector_by_transpose(greatest_remaining_eigen_vector)

        matrix_for_greatest_eigen_value.multiply_by_scalar(scalar_for_matrix)

        matrix = Matrix.subtract(matrix, matrix_for_greatest_eigen_value)
    return eigen_values, eigen_vectors

def approximate_matrix(u, sigma, v_transpose, r):
    u = Matrix.create_matrix_from_data(u.data)
    sigma = Matrix.create_matrix_from_data(sigma.data)
    v_transpose = Matrix.create_matrix_from_data(v_transpose.data)

    u.reduce_columns(r)
    v_transpose.reduce_rows(r)
    sigma.reduce_diagonal(r)

    approximation = Matrix.multiply(Matrix.multiply(u, sigma), v_transpose)
    return approximation



# puppy = imread('images/puppy.jpg')
puppy = imread('images/house.jpg')
# puppy = imread('images/city.jpg')
# puppy = imread('images/beach.jpg')
# puppy = imread('images/person.jpg')
# puppy = imread('images/country.jpg')
# puppy = imread('images/bird.jpg')
# puppy = imread('images/car.jpg')



puppy_grey_for_edges = np.mean(puppy, -1)
puppy_grey_for_edges = puppy_grey_for_edges.tolist()

puppy_grey_for_edges = Matrix.create_matrix_from_data(puppy_grey_for_edges)
puppy_grey_for_edges = Matrix.detect_edges(puppy_grey_for_edges)

puppy_grey_np = np.array(puppy_grey_for_edges.data)

image = plt.imshow(puppy_grey_np)
image.set_cmap('gray')
plt.axis('off')
plt.title('original image edges')
plt.show()



puppy_colour_original = puppy.tolist()
puppy_red, puppy_green, puppy_blue = Matrix.split_three_dimension_array(puppy_colour_original)
original_size = puppy_red.rows * puppy_red.columns * 3
puppy_colour = Matrix.concatenate_matrices_into_list(puppy_red, puppy_green, puppy_blue)

puppy_red_average = puppy_red.get_average()
puppy_green_average = puppy_green.get_average()
puppy_blue_average = puppy_blue.get_average()

print('Original red average', puppy_red_average)
print('Original green average', puppy_green_average)
print('Original blue average', puppy_blue_average)

colour_matrices = [puppy_red, puppy_green, puppy_blue]
colour_matrices_svd = [[], [], []]

image = plt.imshow(puppy)
plt.axis('off')
plt.title('original image, values ' + str(original_size))
plt.show()

rank_to_calculate_to = 40

for i, colour_matrix in enumerate(colour_matrices):
    print('starting for new colour matrix')
    a = Matrix.multiply_left_by_transpose(colour_matrix)
    print('done multiply')

    a_rank = a.get_rank()
    print('matrix rank was', a_rank)
    a_rank = rank_to_calculate_to
    print(a_rank, 'matrix rank')

    a_eigen_values, a_eigen_vectors = get_eigen_vectors(a, a_rank)
    singular_values = [eigen_value ** (1/2) for eigen_value in a_eigen_values]
    print('singular values', a_rank, singular_values)

    for j in range(a_rank):
        if isinstance(singular_values[j], complex):
            print('complex singular value', j, singular_values[j], a_eigen_values[j])

    sigma = Matrix(a_rank, a_rank)
    sigma.add_values_on_leading_diagonal(singular_values)
    print('done sigma')
    v = Matrix.create_matrix_from_vectors(a_eigen_vectors)
    v_transpose = Matrix.transpose(v)
    print('done v')
    inverse_sigma = Matrix.inverse_of_leading_diagonal_matrix(sigma)
    print('done inverse sigma')
    u = Matrix.multiply(Matrix.multiply(colour_matrix, v), inverse_sigma)
    print('done u')
    print('done svd')
    colour_matrices_svd[i].append(u)
    colour_matrices_svd[i].append(sigma)
    colour_matrices_svd[i].append(v_transpose)




for r in range(5, rank_to_calculate_to + 5, 5):
    size_stored = (puppy_red.rows * r + puppy_red.columns * r + r) * 3
    colour_approximations = []
    for i in range(3):
        u = colour_matrices_svd[i][0]
        sigma = colour_matrices_svd[i][1]
        v_transpose = colour_matrices_svd[i][2]
        approximation = approximate_matrix(u, sigma, v_transpose, r)
        colour_approximations.append(approximation)
    approximation_red = colour_approximations[0]
    approximation_green = colour_approximations[1]
    approximation_blue = colour_approximations[2]

    # if r == rank_to_calculate_to:
    #     final_approximation_red = Matrix.create_matrix_from_data(approximation_red.data)
    #     final_approximation_green = Matrix.create_matrix_from_data(approximation_green.data)
    #     final_approximation_blue = Matrix.create_matrix_from_data(approximation_blue.data)

    approximation_red.make_integer()
    approximation_green.make_integer()
    approximation_blue.make_integer()

    puppy_approximation = Matrix.concatenate_matrices_into_list(approximation_red, approximation_green, approximation_blue)

    approximation_red_average = approximation_red.get_average()
    approximation_green_average = approximation_green.get_average()
    approximation_blue_average = approximation_blue.get_average()

    print('Approximation red average', approximation_red_average)
    print('Approximation green average', approximation_green_average)
    print('Approximation blue average', approximation_blue_average)

    new_puppy = np.array(puppy_approximation)
    image = plt.imshow(new_puppy)
    plt.axis('off')
    plt.title('r = ' + str(r) + ', values: ' + str(size_stored) + ', ' + str(int((size_stored/original_size) * 100)) + '%')
    plt.show()

    # for the general sharpening

    approximation_red_sharpen = Matrix.sharpen(approximation_red)
    approximation_green_sharpen = Matrix.sharpen(approximation_green)
    approximation_blue_sharpen = Matrix.sharpen(approximation_blue)

    approximation_red_sharpen.make_integer()
    approximation_green_sharpen.make_integer()
    approximation_blue_sharpen.make_integer()

    puppy_approximation_sharpened = Matrix.concatenate_matrices_into_list(approximation_red_sharpen,
                                                                          approximation_green_sharpen,
                                                                          approximation_blue_sharpen)

    new_puppy_sharpened = np.array(puppy_approximation_sharpened)
    image = plt.imshow(new_puppy_sharpened)
    plt.axis('off')
    plt.title('r = ' + str(r) + ' with sharpening matrix')
    plt.show()


    # With sharpening on the edges only

    new_puppy_grey = np.mean(new_puppy, -1)
    puppy_list = new_puppy_grey.tolist()
    grey_puppy_matrix = Matrix.create_matrix_from_data(puppy_list)
    grey_puppy_edges = Matrix.detect_edges(grey_puppy_matrix)


    new_puppy_edges_grey = np.array(grey_puppy_edges.data)
    image = plt.imshow(new_puppy_edges_grey)
    image.set_cmap('gray')
    plt.axis('off')
    plt.title('edge detection of r values')
    plt.show()


    approximation_red_sharpen = Matrix.sharpen_edges(approximation_red, grey_puppy_edges)
    approximation_green_sharpen = Matrix.sharpen_edges(approximation_green, grey_puppy_edges)
    approximation_blue_sharpen = Matrix.sharpen_edges(approximation_blue, grey_puppy_edges)

    approximation_red_sharpen.make_integer()
    approximation_green_sharpen.make_integer()
    approximation_blue_sharpen.make_integer()

    puppy_approximation_sharpened = Matrix.concatenate_matrices_into_list(approximation_red_sharpen,
                                                                          approximation_green_sharpen,
                                                                          approximation_blue_sharpen)

    new_puppy_sharpened = np.array(puppy_approximation_sharpened)
    image = plt.imshow(new_puppy_sharpened)
    plt.axis('off')
    plt.title('r = ' + str(r) + ' with sharpening matrix on edges')
    plt.show()


    # Sharpen where image says original edges are

    approximation_red_sharpen = Matrix.sharpen_edges(approximation_red, puppy_grey_for_edges)
    approximation_green_sharpen = Matrix.sharpen_edges(approximation_green, puppy_grey_for_edges)
    approximation_blue_sharpen = Matrix.sharpen_edges(approximation_blue, puppy_grey_for_edges)

    approximation_red_sharpen.make_integer()
    approximation_green_sharpen.make_integer()
    approximation_blue_sharpen.make_integer()

    puppy_approximation_sharpened = Matrix.concatenate_matrices_into_list(approximation_red_sharpen,
                                                                          approximation_green_sharpen,
                                                                          approximation_blue_sharpen)

    new_puppy_sharpened = np.array(puppy_approximation_sharpened)
    image = plt.imshow(new_puppy_sharpened)
    plt.axis('off')
    plt.title('r = ' + str(r) + ' with sharpening matrix on edges of original image')
    plt.show()





    # approximation_red_filter = Matrix.filter(approximation_red)
    # approximation_green_filter = Matrix.filter(approximation_green)
    # approximation_blue_filter = Matrix.filter(approximation_blue)
    #
    # approximation_red_filter.make_integer()
    # approximation_green_filter.make_integer()
    # approximation_blue_filter.make_integer()
    #
    # puppy_approximation_filtered = Matrix.concatenate_matrices_into_list(approximation_red_filter,
    #                                                                      approximation_green_filter,
    #                                                                      approximation_blue_filter)
    #
    # new_puppy = np.array(puppy_approximation_filtered)
    # image = plt.imshow(new_puppy)
    # plt.axis('off')
    # plt.title('r = ' + str(r) + ' with filtering matrix')
    # plt.show()




    # approximation_red_sharpen_filter = Matrix.filter(approximation_red_sharpen)
    # approximation_green_sharpen_filter = Matrix.filter(approximation_green_sharpen)
    # approximation_blue_sharpen_filter = Matrix.filter(approximation_blue_sharpen)
    #
    # approximation_red_sharpen_filter.make_integer()
    # approximation_green_sharpen_filter.make_integer()
    # approximation_blue_sharpen_filter.make_integer()
    #
    # puppy_approximation_sharpened_filtered = Matrix.concatenate_matrices_into_list(approximation_red_sharpen_filter,
    #                                                                                approximation_green_sharpen_filter,
    #                                                                                approximation_blue_sharpen_filter)
    #
    # new_puppy = np.array(puppy_approximation_sharpened_filtered)
    # image = plt.imshow(new_puppy)
    # plt.axis('off')
    # plt.title('r = ' + str(r) + ' with sharpening matrix then filtering matrix')
    # plt.show()



    # approximation_red_sharpen_filter = Matrix.sharpen(approximation_red_filter)
    # approximation_green_sharpen_filter = Matrix.sharpen(approximation_green_filter)
    # approximation_blue_sharpen_filter = Matrix.sharpen(approximation_blue_filter)
    #
    # approximation_red_sharpen_filter.make_integer()
    # approximation_green_sharpen_filter.make_integer()
    # approximation_blue_sharpen_filter.make_integer()
    #
    # puppy_approximation_filtered_sharpened = Matrix.concatenate_matrices_into_list(approximation_red_sharpen_filter,
    #                                                                                approximation_green_sharpen_filter,
    #                                                                                approximation_blue_sharpen_filter)
    #
    # new_puppy = np.array(puppy_approximation_filtered_sharpened)
    # image = plt.imshow(new_puppy)
    # plt.axis('off')
    # plt.title('r = ' + str(r) + ' with filtering matrix then sharpening matrix')
    # plt.show()







