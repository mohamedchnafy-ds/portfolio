# Write pyton function that takes a list of numbers and returns the sum of the squares of all the numbers in the list.
# The function should be named sum_of_squares and should take a single argument, a list of numbers.
# The function should return a single number, the sum of the squares of the input numbers.
# The function should not print anything.

def sum_of_squares(numbers):
    total = 0
    for number in numbers:
        total += number ** 2
    return total

# Ecris moi une fonction qui ajoute deux nombres  premiers
def add_two_primes(prime1, prime2):
    return prime1 + prime2

