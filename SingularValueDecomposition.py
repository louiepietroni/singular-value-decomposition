from Matrix import Matrix
from Vector import Vector
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt


class SingularValueDecomposition:
    def __init__(self, matrix, maximum_rank=None):
        self.matrix = matrix
        if maximum_rank:
            # self.maximum_rank = min(self.rank, maximum_rank)
            self.maximum_rank = maximum_rank
            self.rank = None
        else:
            self.rank = self.matrix.get_rank()
            self.maximum_rank = self.rank

        svd_u, svd_sigma, svd_v = self.calculate_svd()

        self.u = svd_u
        self.sigma = svd_sigma
        self.v = svd_v

    def calculate_svd(self):
        self.a_transposed_a = Matrix.multiply_left_by_transpose(self.matrix)

        eigen_values, eigen_vectors = self.get_eigen_vectors_values()

        singular_values = [eigen_value ** (1 / 2) for eigen_value in eigen_values]

        sigma_matrix = Matrix(self.maximum_rank, self.maximum_rank)
        sigma_matrix.add_values_on_leading_diagonal(singular_values)

        v_matrix = Matrix.create_matrix_from_vectors(eigen_vectors)

        inverse_sigma_matrix = Matrix.inverse_of_leading_diagonal_matrix(sigma_matrix)

        u_matrix = Matrix.multiply(Matrix.multiply(self.matrix, v_matrix), inverse_sigma_matrix)

        return u_matrix, sigma_matrix, v_matrix

    def get_eigen_vectors_values(self):
        eigen_vector_matrix = Matrix.create_matrix_from_data(self.a_transposed_a.data)
        eigen_vectors = []
        eigen_values = []

        for _ in range(self.maximum_rank):
            greatest_eigen_vector = Vector.get_greatest_eigen_vector(eigen_vector_matrix)
            greatest_eigen_value = Vector.get_eigen_value(greatest_eigen_vector, eigen_vector_matrix)

            eigen_vector = Vector.create_vector_from_data(greatest_eigen_vector.data)

            eigen_vectors.append(eigen_vector)

            if isinstance((greatest_eigen_value ** (1 / 2)), complex):
                eigen_values.append(1)
            else:
                eigen_values.append(greatest_eigen_value)

            eigen_vector_matrix = SingularValueDecomposition.get_matrix_without_eigen_vector(eigen_vector_matrix, greatest_eigen_vector, greatest_eigen_value)

        return eigen_values, eigen_vectors

    @staticmethod
    def get_matrix_without_eigen_vector(matrix, eigen_vector, eigen_value):
        scalar_for_matrix = eigen_value / eigen_vector.get_magnitude_squared()
        matrix_without_eigen_vector = Matrix.multiply_right_vector_by_transpose(eigen_vector)

        matrix_without_eigen_vector.multiply_by_scalar(scalar_for_matrix)

        matrix_without_eigen_vector = Matrix.subtract(matrix, matrix_without_eigen_vector)
        return matrix_without_eigen_vector

    def approximate_matrix(self, r=None):
        if not r:
            r = self.maximum_rank
        u_matrix_reduced = Matrix.create_matrix_from_data(self.u.data)
        sigma_matrix_reduced = Matrix.create_matrix_from_data(self.sigma.data)
        v_matrix_reduced = Matrix.create_matrix_from_data(self.v.data)

        u_matrix_reduced.reduce_columns(r)
        sigma_matrix_reduced.reduce_diagonal(r)
        v_matrix_reduced.reduce_columns(r)

        u_by_sigma = Matrix.multiply(u_matrix_reduced, sigma_matrix_reduced)
        v_matrix_reduced_transposed = Matrix.transpose(v_matrix_reduced)

        matrix_approximation = Matrix.multiply(u_by_sigma, v_matrix_reduced_transposed)
        return matrix_approximation


class GreyScaleImageSVD:
    def __init__(self, file_name, maximum_rank=None):
        self.file = file_name
        image_numpy_matrix = imread(file_name)
        grey_image_numpy_matrix = np.mean(image_numpy_matrix, -1)

        grey_image_array = grey_image_numpy_matrix.tolist()
        self.grey_image_matrix = Matrix.create_matrix_from_data(grey_image_array)
        self.grey_image_matrix.make_integer()

        self.original_matrix_size = self.grey_image_matrix.rows * self.grey_image_matrix.columns

        self.image_svd = SingularValueDecomposition(self.grey_image_matrix, maximum_rank)

    def show_full_image(self):
        image_title = 'Greyscale full image of ' + self.file
        GreyScaleImageSVD.show_matrix_as_image(self.grey_image_matrix, image_title)

    def approximate_image(self, r=None):
        if not r:
            r = self.image_svd.maximum_rank
        return self.image_svd.approximate_matrix(r)

    def show_image_approximation(self, r=None):
        if not r:
            r = self.image_svd.maximum_rank
        image_approximation = self.approximate_image(r)
        image_title = 'Greyscale image approximation of ' + self.file + ' at rank ' + str(r)
        GreyScaleImageSVD.show_matrix_as_image(image_approximation, image_title)

    @staticmethod
    def show_matrix_as_image(matrix, title=''):
        numpy_matrix = np.array(matrix.data)
        full_image = plt.imshow(numpy_matrix)
        full_image.set_cmap('gray')
        plt.axis('off')
        plt.title(title)
        plt.show()


class ColourImageSVD:
    def __init__(self, file_name, maximum_rank=None):
        self.file = file_name
        image_numpy_matrix = imread(file_name)

        self.colour_image_array = image_numpy_matrix.tolist()

        red_matrix, green_matrix, blue_matrix = Matrix.split_three_dimension_array(self.colour_image_array)
        self.red_matrix = red_matrix
        self.green_matrix = green_matrix
        self.blue_matrix = blue_matrix

        self.original_matrix_size = red_matrix.rows * red_matrix.columns * 3

        self.red_matrix_svd = SingularValueDecomposition(self.red_matrix, maximum_rank)
        self.green_matrix_svd = SingularValueDecomposition(self.green_matrix, maximum_rank)
        self.blue_matrix_svd = SingularValueDecomposition(self.blue_matrix, maximum_rank)

    def show_full_image(self):
        image_title = 'Full colour image of ' + self.file
        ColourImageSVD.show_array_as_image(self.colour_image_array, image_title)

    def show_image_approximation(self, r=None):
        if not r:
            r = self.red_matrix_svd.maximum_rank

        red_matrix_approximation = self.red_matrix_svd.approximate_matrix(r)
        green_matrix_approximation = self.green_matrix_svd.approximate_matrix(r)
        blue_matrix_approximation = self.blue_matrix_svd.approximate_matrix(r)

        red_matrix_approximation.make_integer()
        green_matrix_approximation.make_integer()
        blue_matrix_approximation.make_integer()

        colour_array_approximation = Matrix.concatenate_matrices_into_list(red_matrix_approximation, green_matrix_approximation, blue_matrix_approximation)

        image_title = 'Colour image approximation of ' + self.file + ' at rank ' + str(r)
        ColourImageSVD.show_array_as_image(colour_array_approximation, image_title)

    @staticmethod
    def show_array_as_image(array, title=''):
        numpy_matrix = np.array(array)
        full_image = plt.imshow(numpy_matrix)
        plt.axis('off')
        plt.title(title)
        plt.show()

