from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from Matrix import Matrix
from Vector import Vector


# NOTE: Choose image from line ~55
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



puppy = imread('images/puppy.jpg')
# puppy = imread('images/house.jpg')
# puppy = imread('images/city.jpg')
# puppy = imread('images/beach.jpg')
# puppy = imread('images/person.jpg')
# puppy = imread('images/country.jpg')
# puppy = imread('images/bird.jpg')
# puppy = imread('images/car.jpg')

puppy_grey = np.mean(puppy, -1)
image = plt.imshow(puppy_grey)
image.set_cmap('gray')
plt.axis('off')
plt.title('original image')
plt.show()

puppy_list = puppy_grey.tolist()

original_matrix = Matrix.create_matrix_from_data(puppy_list)

original_matrix.make_integer()

original_size = original_matrix.rows * original_matrix.columns

integer_puppy = np.array(original_matrix.data)
image = plt.imshow(integer_puppy)
image.set_cmap('gray')
plt.axis('off')
plt.title('image with integer values, values: ' + str(original_size))
plt.show()

# approximation_edges = Matrix.detect_edges(original_matrix)
# new_puppy = np.array(approximation_edges.data)
# image = plt.imshow(new_puppy)
# image.set_cmap('gray')
# plt.axis('off')
# plt.title('with edge detection matrix')
# plt.show()


# print(original_matrix, 'original matrix')
a = Matrix.multiply_left_by_transpose(original_matrix)
# print(a, 'multiply front by transpose')
print('done multiply')

a_rank = a.get_rank()

print('matrix rank was', a_rank)
a_rank = 50


print(a_rank, 'matrix rank')
# print(a, 'now find eigen values and vectors')
a_eigen_values, a_eigen_vectors = get_eigen_vectors(a, a_rank)
# for i in range(a_rank):
#     print('Eigen value, eigen vector ', a_eigen_values[i], a_eigen_vectors[i])
singular_values = [eigen_value ** (1/2) for eigen_value in a_eigen_values]
print('singular values', a_rank, singular_values)

for i in range(a_rank):
    if isinstance(singular_values[i], complex):
        print('complex singular value', i, singular_values[i], a_eigen_values[i])

sigma = Matrix(a_rank, a_rank)
sigma.add_values_on_leading_diagonal(singular_values)
print('done sigma')
v = Matrix.create_matrix_from_vectors(a_eigen_vectors)
v_transpose = Matrix.transpose(v)
print('done v')
inverse_sigma = Matrix.inverse_of_leading_diagonal_matrix(sigma)
print('done inverse sigma')
u = Matrix.multiply(Matrix.multiply(original_matrix, v), inverse_sigma)
print('done u')
# reconstruction = Matrix.multiply(Matrix.multiply(u, sigma), v_transpose)
print('done svd')


for r in range(5, a_rank + 5, 5):

    size_stored = original_matrix.rows * r + original_matrix.columns * r + r

    approximation = approximate_matrix(u, sigma, v_transpose, r)
    new_puppy = np.array(approximation.data)
    image = plt.imshow(new_puppy)
    image.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r) + ', values: ' + str(size_stored) + ', ' + str(int((size_stored/original_size) * 100)) + '%')
    plt.show()

    # approximation_edges = Matrix.detect_edges(approximation)
    #
    # new_puppy = np.array(approximation_edges.data)
    # image = plt.imshow(new_puppy)
    # image.set_cmap('gray')
    # plt.axis('off')
    # plt.title('r = ' + str(r) + ' with edge detection matrix')
    # plt.show()


