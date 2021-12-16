import src.ci_utils
import unit_tests.matrix_utils_test
import unit_tests.ci_utils_test

print("All tests have passed.")

[[print(str(k) + " " + str(l) + " (" + str(src.ci_utils.addressing_array(k, l, 3, 5)) + ")") for k in range(3)] for l in range(5)]