# from Matrix import Matrix
# from Vector import Vector
#
#
# def get_eigen_vectors(original_matrix, matrix_rank):
#     matrix = Matrix(original_matrix.rows, original_matrix.columns)
#     matrix.data = [[item for item in row] for row in original_matrix.data]
#
#     eigen_vectors = []
#     eigen_values = []
#     for i in range(matrix_rank):
#         greatest_remaining_eigen_vector = Vector.get_greatest_eigen_vector(matrix)
#         greatest_remaining_eigen_value = Vector.get_eigen_value(greatest_remaining_eigen_vector, matrix)
#
#         singular_value = greatest_remaining_eigen_value ** (1/2)
#         # print(greatest_remaining_eigen_vector.data, greatest_remaining_eigen_value, singular_value)
#
#         current_eigen_vector = Vector.create_vector_from_data(greatest_remaining_eigen_vector.data)
#
#         eigen_vectors.append(current_eigen_vector)
#         eigen_values.append(greatest_remaining_eigen_value)
#
#         scalar_for_matrix = greatest_remaining_eigen_value / greatest_remaining_eigen_vector.get_magnitude_squared()
#         matrix_for_greatest_eigen_value = Matrix.multiply_right_vector_by_transpose(greatest_remaining_eigen_vector)
#
#         matrix_for_greatest_eigen_value.multiply_by_scalar(scalar_for_matrix)
#
#         matrix = Matrix.subtract(matrix, matrix_for_greatest_eigen_value)
#     return eigen_values, eigen_vectors
#
#
# # original_matrix = Matrix(7, 5)
# #
# # original_matrix.data = [[1, 0, 1, 0, 7],
# #                         [3, 0, 3, 0, 0],
# #                         [4, 0, 4, 0, 2],
# #                         [5, 0, 5, 0, 0],
# #                         [0, 0, 0, 0, 4],
# #                         [0, 0, 5, 0, 5],
# #                         [0, 0, 0, 0, 2]]
#
#
# # original_matrix = Matrix(7, 5)
# #
# # original_matrix.data = [[1, 1, 1, 0, 0],
# #                         [3, 3, 3, 0, 0],
# #                         [4, 4, 4, 0, 0],
# #                         [5, 5, 5, 0, 0],
# #                         [0, 2, 0, 4, 4],
# #                         [0, 0, 0, 5, 5],
# #                         [0, 1, 0, 2, 2]]
#
# original_matrix = Matrix(2, 2)
# original_matrix.data = [[5, 5],
#                         [-1, 7]]
#
# # original_matrix = Matrix(3, 3)
# # original_matrix.data = [[2, 0, 0],
# #                         [2, 1, 0],
# #                         [0, -2, 0]]
#
# # original_matrix = Matrix(3, 4)
# # original_matrix.data = [[1, 0, -3, 1],
# #                         [2, 4, 1, 3],
# #                         [3, -2, -2, 0]]
#
# print(original_matrix, 'original matrix')
# a = Matrix.multiply_left_by_transpose(original_matrix)
# print(a, 'multiply front by transpose')
#
# a_rank = a.get_rank()
# print(a_rank, 'matrix rank')
# # print(a, 'now find eigen values and vectors')
#
# a_eigen_values, a_eigen_vectors = get_eigen_vectors(a, a_rank)
# for i in range(a_rank):
#     print('Eigen value, eigen vector ', a_eigen_values[i], a_eigen_vectors[i])
#
# singular_values = [eigen_value ** (1/2) for eigen_value in a_eigen_values]
#
# sigma = Matrix(a_rank, a_rank)
# sigma.add_values_on_leading_diagonal(singular_values)
# print(sigma, 'sigma matrix made of roots of eigen values')
#
# # v = Matrix(a.columns, a_rank)
# v = Matrix.create_matrix_from_vectors(a_eigen_vectors)
# print(v, 'v matrix made of eigen vectors')
#
# inverse_sigma = Matrix.inverse_of_leading_diagonal_matrix(sigma)
# print(inverse_sigma, 'inverse sigma')
#
# u = Matrix.multiply(Matrix.multiply(original_matrix, v), inverse_sigma)
# print(u, 'u matrix calculated by A V inverse sigma')
#
# reconstruction = Matrix.multiply(Matrix.multiply(u, sigma), Matrix.transpose(v))
# print(reconstruction, 'reconstructed matrix')


from Matrix import Matrix
from SingularValueDecomposition import SingularValueDecomposition, GreyScaleImageSVD, ColourImageSVD

matrix = Matrix(2, 2)
matrix.data = [[5, 5],
               [-1, 7]]

matrix_svd = SingularValueDecomposition(matrix)
print(matrix_svd.approximate_matrix())
print(matrix_svd.u)

# the_image = GreyScaleImageSVD('puppy.jpg', 50)
# the_image.show_full_image()
# the_image.show_image_approximation(10)
# the_image.show_image_approximation(20)
# the_image.show_image_approximation(30)
# the_image.show_image_approximation(45)


# the_image = GreyScaleImageSVD('house.jpg', 40)
# the_image.show_full_image()
# the_image.show_image_approximation(10)

the_image = ColourImageSVD('puppy.jpg', 20)
the_image.show_full_image()
the_image.show_image_approximation(5)
the_image.show_image_approximation(15)
the_image.show_image_approximation(20)
