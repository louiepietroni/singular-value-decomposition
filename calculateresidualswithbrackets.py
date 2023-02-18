from Matrix import Matrix
from Vector import Vector
from SingularValueDecomposition import SingularValueDecomposition
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.image import imread
from savingmatrices import store_matrices_to_file, get_matrices_from_file


training_matrices_1 = get_matrices_from_file('louietrainingimages/page1digitmatrices.txt')
training_matrices_2 = get_matrices_from_file('louietrainingimages/page2digitmatrices.txt')
training_matrices_symbols = get_matrices_from_file('louietrainingimages/symboldigitmatrices.txt')
training_matrices_other_symbols = get_matrices_from_file('louietrainingimages/bracketdigitmatrices.txt')


training_matrices_for_handwriting = []

for i in range(10):
    digit_handwriting_matrix = Matrix.concatenate_matrices_side(training_matrices_1[i], training_matrices_2[i])
    training_matrices_for_handwriting.append(digit_handwriting_matrix)

training_matrices_for_handwriting = training_matrices_for_handwriting + training_matrices_symbols + training_matrices_other_symbols


svd_rank = 10

training_matrices_of_digits_svd = []
for digit_training_matrix in training_matrices_for_handwriting:
    digit_training_matrix_svd = SingularValueDecomposition(digit_training_matrix, svd_rank)
    training_matrices_of_digits_svd.append(digit_training_matrix_svd)
    print('completed svd of digit matrix for handwriting')

matrices_for_residual = []
for digit_svd in training_matrices_of_digits_svd:
    u_matrix_transposed = Matrix.transpose(digit_svd.u)
    u_u_transposed = Matrix.multiply_left_by_transpose(u_matrix_transposed)

    identity_matrix = Matrix.identity(u_u_transposed.columns)
    matrix_for_residual_for_digit = Matrix.subtract(identity_matrix, u_u_transposed)
    matrices_for_residual.append(matrix_for_residual_for_digit)
    print('completed residual matrix for handwritten digit')

store_matrices_to_file(matrices_for_residual, 'louietrainingimages/louiefullsymbolresiduals.txt')




















